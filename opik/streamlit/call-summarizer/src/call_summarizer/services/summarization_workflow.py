"""LangGraph workflow for call summarization with Opik tracing."""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import Graph, StateGraph
from opik.integrations.langchain import OpikTracer

from ..config import settings
from ..models.models import CallCategoryConfig, CallSummary


class WorkflowState(TypedDict):
    """State for the summarization workflow."""

    transcript: str
    category: str
    category_config: CallCategoryConfig
    summary: str
    action_items: List[str]
    metadata: Dict


def create_summarization_workflow() -> Graph:
    """Create the LangGraph workflow for call summarization with Opik tracing."""
    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
    )

    # Define the nodes
    def route_to_category(state: WorkflowState) -> WorkflowState:
        """Route the call to the appropriate category handler."""
        # If no category is provided, use the default category
        if not state.get("category"):
            state["category"] = "other"

        # Get the category config
        category_config = state.get("category_config")
        if not category_config:
            raise ValueError(f"No configuration found for category: {state['category']}")

        # This node doesn't modify the state further, just ensures category is set
        return state

    def summarize_call(state: WorkflowState) -> WorkflowState:
        """Summarize the call transcript based on the category."""
        transcript = state["transcript"]
        category_config = state["category_config"]

        prompt = f"""{category_config.prompt_template}\n\n"
            f"Call Transcript:\n{transcript}\n\n"
            f"Please provide a concise summary of the call based on the category '{category_config.name}'.\n"
            f"IMPORTANT: Your main task here is to generate the narrative summary. "
            f"Do NOT include a list or section detailing specific action items in THIS summary. "
            f"Action items will be extracted and listed entirely separately. "
            f"Focus only on the overall discussion, key points, decisions, and outcomes as guided by the category template.\n"
            f"Summary:\n"
            f"""

        messages = [
            SystemMessage(content=f"You are an expert call summarizer for the category: {category_config.name}."),
            HumanMessage(content=prompt),
        ]

        response = llm.invoke(messages)
        state["summary"] = response.content.strip()
        return state

    def extract_action_items(state: WorkflowState) -> WorkflowState:
        """Extract action items from the call transcript."""
        transcript = state["transcript"]
        summary = state["summary"]  # Use the generated summary as context

        prompt = f"""
        Given the following call transcript and its summary, please extract key action items.
        If no specific action items are mentioned, state 'No action items identified'.
        Format the action items as a list.

        Call Transcript:
        {transcript}

        Summary:
        {summary}

        Action Items:
        - [Action Item 1]
        - [Action Item 2]
        ...
        """

        messages = [
            SystemMessage(content="You are an expert in identifying action items from call transcripts."),
            HumanMessage(content=prompt),
        ]

        response = llm.invoke(messages)
        action_items_text = response.content.strip()

        if "No action items identified" in action_items_text:
            state["action_items"] = []
        else:
            # Basic parsing, assuming action items are listed with '-'
            parsed_items = [item.strip() for item in action_items_text.split("-") if item.strip()]
            # Remove duplicates, preserving order
            state["action_items"] = list(dict.fromkeys(parsed_items))
        return state

    def add_metadata(state: WorkflowState) -> WorkflowState:
        """Add metadata to the call summary."""
        state["metadata"] = {
            "summarization_date": datetime.utcnow().isoformat(),
            "llm_model_used": "gpt-4o-mini",  # Or dynamically get from llm object if possible
            "workflow_version": "1.0",
        }
        return state

    # Define the graph
    graph = StateGraph(WorkflowState)

    graph.add_node("route_to_category", route_to_category)
    graph.add_node("summarize_call", summarize_call)
    graph.add_node("extract_action_items", extract_action_items)
    graph.add_node("add_metadata", add_metadata)

    # Define the edges
    graph.set_entry_point("route_to_category")
    graph.add_edge("route_to_category", "summarize_call")
    graph.add_edge("summarize_call", "extract_action_items")
    graph.add_edge("extract_action_items", "add_metadata")
    graph.set_finish_point("add_metadata")

    # Compile the graph
    compiled_workflow = graph.compile()

    return compiled_workflow


class CallSummarizer:
    """Service for summarizing call transcripts using LangGraph."""

    def __init__(self, category_manager):
        """Initialize the call summarizer."""
        self.workflow = create_summarization_workflow()  # This is the compiled graph
        self.category_manager = category_manager
        self.opik_tracer = None  # Initialize opik_tracer attribute

        if settings.opik_api_key:
            try:
                # Ensure self.workflow has get_graph method and xray is a valid param for it in this context
                # The Opik documentation implies app.get_graph(xray=True) is standard.
                graph_for_tracer = self.workflow.get_graph(xray=True)
                self.opik_tracer = OpikTracer(
                    graph=graph_for_tracer, tags=["langchain", "call-summarizer", "langgraph"], metadata={"use-case": "call-summarizer"}
                )
            except Exception as e:
                # Fallback if get_graph(xray=True) or OpikTracer init with graph fails
                print(f"Warning: Could not initialize OpikTracer with graph details: {e}. Falling back to basic OpikTracer.")
                self.opik_tracer = OpikTracer(
                    tags=["langchain", "call-summarizer", "langgraph"], metadata={"use-case": "call-summarizer"}
                )  # Basic tracer as a fallback

    def summarize_transcript(
        self,
        transcript: str,
        category_name: Optional[str] = None,
    ) -> CallSummary:
        """Summarize a call transcript."""
        # Get the category config
        if not category_name:
            category = self.category_manager.get_default_category()
        else:
            category = self.category_manager.get_category(category_name)

        if not category:
            raise ValueError(f"No category found: {category_name}")

        # Prepare the initial state
        initial_state = {
            "transcript": transcript,
            "category": category.name.lower(),
            "category_config": category,
            "summary": "",
            "action_items": [],
            "metadata": {},
        }

        # Prepare the run configuration for Langchain callbacks
        run_config = {"recursion_limit": 25}
        if self.opik_tracer:  # Use the instance's opik_tracer
            run_config["callbacks"] = [self.opik_tracer]  # Use the instance's opik_tracer

        # Run the workflow synchronously, passing the config
        result = self.workflow.invoke(initial_state, config=run_config)

        # Create and return the call summary
        return CallSummary(
            id=str(uuid.uuid4()),
            transcript=transcript,
            summary=result["summary"],
            action_items=result["action_items"],
            category=result["category"],
            metadata=result["metadata"],
        )
