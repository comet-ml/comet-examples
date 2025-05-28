"""Vector store service for storing and querying call summaries using ChromaDB with Opik tracing."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

import chromadb

# RetrievalQA is deprecated, will be replaced by create_retrieval_chain
# from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# from langchain.chains.question_answering import load_qa_chain # No longer needed
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_core.documents import Document as LangchainDocument
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from opik import track
from opik.integrations.langchain import OpikTracer

from ..config import settings
from ..models.models import CallSummary, VectorStoreConfig

# Define the custom prompt for the QA chain
PROMPT_TEMPLATE_STR = """You are a helpful assistant for querying call summaries.
Use only the following pieces of context (call transcripts and their metadata) to answer the question.
The metadata for each call includes 'id', 'summary', 'action_items', 'category', and 'created_at'.
If the provided context does not contain the answer to the question, state that the information was not found in the call history for that query.
Do not make up information or answer questions outside of the provided context.
Always answer in the same language as the question.
Always answer in markdown format.

Context:
{context}

Question: {input} 
Helpful Answer:"""  # Note: Changed {question} to {input}
CUSTOM_PROMPT = PromptTemplate(template=PROMPT_TEMPLATE_STR, input_variables=["context", "input"])  # Note: Changed 'question' to 'input'


class VectorStoreService:
    """Service for managing call summaries in a ChromaDB vector store."""

    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """Initialize the vector store service with Opik tracing."""
        self.config = config or VectorStoreConfig(persist_dir=settings.vector_store_path, collection_name="call_summaries")

        # Initialize the embedding model with Opik tracing
        self.opik_tracer = OpikTracer() if settings.opik_api_key else None
        self.embedding_model = OpenAIEmbeddings()

        # Initialize ChromaDB client and collection
        self.chroma_client = chromadb.PersistentClient(path=str(self.config.persist_dir))

        # Create or get the collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

        # Initialize the LangChain Chroma vector store with Opik tracing
        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name=self.config.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=str(self.config.persist_dir),
        )

        # Initialize LLM for QA with Opik tracing
        self.llm = ChatOpenAI(model_name="gpt-4-turbo-preview", callbacks=[self.opik_tracer] if self.opik_tracer else None)

    @track(name="add_call_summary_to_vector_store", flush=True)
    def add_call_summary(self, call_summary: CallSummary) -> str:
        """Add a call summary to the vector store with Opik tracing."""
        # Create a document from the call summary
        doc_id = str(uuid4())
        document = LangchainDocument(
            page_content=call_summary.transcript,
            metadata={
                "id": doc_id,
                "summary": call_summary.summary,
                "action_items": "\n".join(call_summary.action_items),
                "category": call_summary.category.value,
                "created_at": call_summary.created_at.isoformat(),
                **call_summary.metadata,
            },
        )

        # Add the document to the vector store with tracing
        self.vector_store.add_documents([document], ids=[doc_id], callbacks=[self.opik_tracer] if self.opik_tracer else None)

        return doc_id

    @track(name="search_call_summaries", flush=True)
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for call summaries similar to the query with Opik tracing."""
        # Search for similar documents with tracing
        docs = self.vector_store.similarity_search_with_score(query, k=top_k, callbacks=[self.opik_tracer] if self.opik_tracer else None)

        # Convert to result format
        results = []
        for doc, score in docs:
            results.append({"id": doc.metadata.get("id", ""), "text": doc.page_content, "score": float(score), "metadata": doc.metadata})

        return results

    @track(name="chat_with_call_history_query", flush=True)
    def query(self, query: str, response_mode: str = "compact") -> str:
        """Query the vector store with a natural language question."""
        callbacks = [self.opik_tracer] if self.opik_tracer else []
        config = {}
        if callbacks:
            config["callbacks"] = callbacks

        # Use the original query for the LLM part of the chain
        llm_question = query

        if "today" in query.lower():
            today_date_str = datetime.now().strftime("%Y-%m-%d")

            # For 'today' queries, we retrieve more documents and filter them by date.
            # The query for retrieval might be the original query, or a modified one if needed.
            retrieval_query = query
            try:
                all_potentially_relevant_docs = self.vector_store.similarity_search(
                    retrieval_query,
                    k=20,  # Retrieve more documents to filter from
                )

                filtered_docs = [
                    doc for doc in all_potentially_relevant_docs if doc.metadata.get("created_at", "").startswith(today_date_str)
                ]
            except Exception as e:
                # Consider logging this error properly
                print(f"Error during similarity search for 'today' query: {e}")
                filtered_docs = []

            if not filtered_docs:
                return "I did not find any call summaries recorded for today."

            # Use create_stuff_documents_chain with the filtered documents and custom prompt
            # This chain directly processes documents and a question (as 'input')
            document_chain = create_stuff_documents_chain(self.llm, CUSTOM_PROMPT)

            # The input to this chain is a dictionary with 'context' (the documents)
            # and 'input' (the question/query based on CUSTOM_PROMPT)
            result_str = document_chain.invoke({"context": filtered_docs, "input": llm_question}, config=config)
            return result_str if result_str else "I could not generate a response based on today's calls."

        else:
            # Modern approach for non-"today" queries using create_retrieval_chain
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})  # Standard k for general queries

            # Create the document chain that will process the retrieved documents
            document_chain = create_stuff_documents_chain(self.llm, CUSTOM_PROMPT)

            # Create the retrieval chain
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            response = retrieval_chain.invoke({"input": llm_question}, config=config)  # New chains expect "input"
            # The answer is in response['answer']
            # Retrieved documents are in response['context']
            return response.get("answer", "I could not generate a response based on the available call history.")

    def get_all_summaries(self) -> List[Dict[str, Any]]:
        """Get all call summaries from the vector store."""
        # Get all documents from the collection
        results = self.chroma_client.get_collection(self.config.collection_name).get()

        # Convert to summary format
        summaries = []
        for i, doc_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i]
            document = results["documents"][i]

            summary = {
                "id": doc_id,
                "text": (document[:500] + "...") if len(document) > 500 else document,
                "summary": metadata.get("summary", ""),
                "category": metadata.get("category", "other"),
                "created_at": metadata.get("created_at", ""),
            }
            summaries.append(summary)

        return summaries

    def delete_summary(self, summary_id: str) -> bool:
        """Delete a call summary from the vector store."""
        try:
            self.vector_store._collection.delete(ids=[summary_id])
            return True
        except Exception as e:
            print(f"Error deleting summary {summary_id}: {e}")
            return False
