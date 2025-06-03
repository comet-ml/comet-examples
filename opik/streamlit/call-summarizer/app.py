"""Streamlit app for call summarization."""

import sys
from pathlib import Path

# Add the src directory to the Python path BEFORE attempting to import from src
# This assumes app.py is in the project root, and 'src' is a direct subdirectory.
sys.path.append(str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
import opik  # For opik.configure
import streamlit as st
from opik import track  # For the decorator

# Now import local modules from 'src'
from src.call_summarizer.config import settings
from src.call_summarizer.models.models import CallSummary
from src.call_summarizer.services.category_manager import CategoryManager
from src.call_summarizer.services.summarization_workflow import CallSummarizer
from src.call_summarizer.services.vector_store import VectorStoreConfig, VectorStoreService

# Load environment variables after all imports are declared
load_dotenv()

opik = opik.configure(
    api_key=settings.opik_api_key,
    workspace=settings.opik_workspace,
    use_local=False,
)

# Page configuration
st.set_page_config(
    page_title="Call Summarizer",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Initialize services
@st.cache_resource
def get_category_manager():
    return CategoryManager()


@st.cache_resource
def get_vector_store():
    return VectorStoreService(VectorStoreConfig(persist_dir=settings.vector_store_path, collection_name="call_summaries"))


@st.cache_resource
def get_summarizer():
    return CallSummarizer(category_manager=get_category_manager())


category_manager = get_category_manager()
vector_store = get_vector_store()
summarizer = get_summarizer()

# Ensure default categories exist
category_manager.ensure_default_categories_exist()


def main():
    """Main application function."""
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Summarize Call", "View History", "Manage Categories", "Chat with Data"],
    )

    if page == "Summarize Call":
        show_summarize_call()
    elif page == "View History":
        show_history()
    elif page == "Manage Categories":
        show_manage_categories()
    elif page == "Chat with Data":
        show_chat()


@track(flush=True)
def process_transcript(
    transcript: str, category_name: str, summarizer: CallSummarizer = summarizer, vector_store: VectorStoreService = vector_store
) -> CallSummary:
    """Process the transcript and generate a summary."""
    # Generate the summary
    call_summary = summarizer.summarize_transcript(transcript=transcript, category_name=category_name)

    # Save to vector store
    vector_store.add_call_summary(call_summary)

    return call_summary


def show_summarize_call():
    """Show the call summarization interface."""
    st.title("📞 Call Summarizer")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload a call transcript", type=["txt", "md"], help="Upload a text or markdown file containing the call transcript"
    )

    # Or paste text
    transcript = st.text_area("Or paste the call transcript here", height=200, placeholder="Paste the call transcript here...")

    # Category selection
    categories = category_manager.list_categories()
    category_names = [cat.name for cat in categories]
    selected_category = st.selectbox(
        "Select a category", category_names, index=category_names.index("other") if "other" in [c.lower() for c in category_names] else 0
    )

    # Summarize button
    if st.button("Summarize Call", type="primary"):
        if not (uploaded_file or transcript.strip()):
            st.error("Please upload a file or paste a transcript.")
            return

        with st.spinner("Generating summary..."):
            try:
                # Read the transcript
                if uploaded_file:
                    transcript = uploaded_file.read().decode("utf-8")

                # Process the transcript and generate summary
                call_summary = process_transcript(transcript, selected_category, summarizer, vector_store)

                # Display the results
                display_call_summary(call_summary)

            except Exception as e:
                st.error(f"Error generating summary: {str(e)}")


def display_call_summary(call_summary: CallSummary):
    """Display a call summary."""
    st.subheader("Call Summary")
    st.write(call_summary.summary)

    st.subheader("Action Items")
    for item in call_summary.action_items:
        st.write(f"- {item}")

    st.subheader("Details")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Category", call_summary.category)
    with col2:
        st.metric("Created", call_summary.created_at.strftime("%Y-%m-%d %H:%M"))


def show_history():
    """Show the call history with delete functionality."""
    st.title("📜 Call History")

    # Get all summaries
    summaries = vector_store.get_all_summaries()

    if not summaries:
        st.info("No call summaries found.")
        return

    # Filter by category
    categories = list({s["category"] for s in summaries})
    selected_category = st.selectbox("Filter by category", ["All"] + sorted(categories))

    if selected_category != "All":
        summaries = [s for s in summaries if s["category"] == selected_category]

    # Display summaries with delete buttons
    for summary in summaries:
        col1, col2 = st.columns([0.9, 0.1])

        with col1:
            with st.expander(f"{summary['summary'][:100]}..."):
                st.write(summary["summary"])
                st.caption(f"Category: {summary['category']} | Created: {summary['created_at']}")

        with col2:
            # Create a unique key for the delete button
            delete_key = f"delete_{summary['id']}"
            if st.button("🗑️", key=delete_key, help="Delete this summary"):
                # Store the ID to delete in the session state
                st.session_state["delete_summary_id"] = summary["id"]

    # Confirmation dialog
    if "delete_summary_id" in st.session_state:
        st.warning("Are you sure you want to delete this summary? This action cannot be undone.")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Yes, delete it"):
                summary_id = st.session_state["delete_summary_id"]
                if vector_store.delete_summary(summary_id):
                    st.success("Summary deleted successfully!")
                    # Clear the delete state
                    del st.session_state["delete_summary_id"]
                    # Rerun to refresh the list
                    st.rerun()
                else:
                    st.error("Failed to delete summary.")

        with col2:
            if st.button("❌ No, keep it"):
                # Clear the delete state
                del st.session_state["delete_summary_id"]
                st.rerun()


def show_manage_categories():
    """Show the category management interface."""
    st.title("🗂️ Manage Categories")

    # List existing categories
    st.subheader("Existing Categories")
    categories = category_manager.list_categories()

    if not categories:
        st.info("No categories found. Create one below.")
    else:
        for category in categories:
            with st.expander(f"{category.name}"):
                st.write(category.description)
                st.code(category.prompt_template, language="text")

                if st.button(f"Delete {category.name}", key=f"del_{category.name}"):
                    if category_manager.delete_category(category.name):
                        st.success(f"Category '{category.name}' deleted.")
                        st.rerun()

    # Add new category
    st.subheader("Add New Category")
    with st.form("add_category"):
        name = st.text_input("Name")
        description = st.text_area("Description")
        prompt_template = st.text_area("Prompt Template", help="Use {transcript} as a placeholder for the call transcript.")

        if st.form_submit_button("Add Category"):
            if not all([name, description, prompt_template]):
                st.error("All fields are required.")
            else:
                try:
                    category_manager.create_category(name=name, description=description, prompt_template=prompt_template)
                    st.success(f"Category '{name}' created successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error creating category: {str(e)}")


def show_chat():
    """Show the chat interface for querying call data."""
    st.title("💬 Chat with Call Data")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your call history..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Query the vector store
                    response = vector_store.query(prompt)
                    st.markdown(response)

                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")


if __name__ == "__main__":
    main()
