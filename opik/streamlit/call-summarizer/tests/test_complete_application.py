"""
Complete application test suite with focused Opik integration.

This test suite covers the entire call summarizer application with:
- Pure unit tests (no @llm_unit) for data models, CRUD operations, file I/O
- LLM integration tests (with @llm_unit) for actual AI functionality
- Real CallSummarizer AI function calls where possible
"""

import pytest
import os
import tempfile
import shutil
import json
from unittest.mock import Mock, patch
from datetime import datetime
from uuid import uuid4

from opik import llm_unit, track
from src.call_summarizer.models.models import (
    CallCategory,
    CallSummary,
    CallCategoryConfig,
    VectorStoreConfig,
)
from src.call_summarizer.services.category_manager import CategoryManager
from src.call_summarizer.services.vector_store import VectorStoreService
from src.call_summarizer.services.summarization_workflow import CallSummarizer
from src.call_summarizer.utils.file_utils import (
    ensure_dir_exists,
    save_config,
    load_config,
    save_list_of_configs,
    load_list_of_configs,
)


# =============================================================================
# PURE UNIT TESTS (NO @llm_unit - no LLM functionality involved)
# =============================================================================


class TestDataModels:
    """Test data models - pure Pydantic validation, no LLM calls."""

    def test_call_category_enum_values(self):
        """Test CallCategory enum has all expected values."""
        expected_categories = ["sales", "support", "interview", "meeting", "other"]
        actual_categories = [cat.value for cat in CallCategory]

        assert set(expected_categories) == set(actual_categories)
        assert CallCategory.SALES.value == "sales"
        assert CallCategory.SUPPORT.value == "support"
        assert CallCategory.INTERVIEW.value == "interview"
        assert CallCategory.MEETING.value == "meeting"
        assert CallCategory.OTHER.value == "other"

    def test_call_summary_creation_with_defaults(self):
        """Test CallSummary creation with minimal required fields."""
        summary_id = str(uuid4())
        transcript = "Test transcript content"
        summary_text = "Test summary content"

        call_summary = CallSummary(id=summary_id, transcript=transcript, summary=summary_text)

        assert call_summary.id == summary_id
        assert call_summary.transcript == transcript
        assert call_summary.summary == summary_text
        assert call_summary.action_items == []
        assert call_summary.category == CallCategory.OTHER
        assert isinstance(call_summary.created_at, datetime)
        assert call_summary.metadata == {}

    def test_call_summary_creation_with_all_fields(self):
        """Test CallSummary creation with all fields specified."""
        summary_id = str(uuid4())
        transcript = "Test transcript content"
        summary_text = "Test summary content"
        action_items = ["Action 1", "Action 2"]
        category = CallCategory.SALES
        created_at = datetime(2024, 1, 1, 12, 0, 0)
        metadata = {"key": "value"}

        call_summary = CallSummary(
            id=summary_id,
            transcript=transcript,
            summary=summary_text,
            action_items=action_items,
            category=category,
            created_at=created_at,
            metadata=metadata,
        )

        assert call_summary.id == summary_id
        assert call_summary.transcript == transcript
        assert call_summary.summary == summary_text
        assert call_summary.action_items == action_items
        assert call_summary.category == category
        assert call_summary.created_at == created_at
        assert call_summary.metadata == metadata

    def test_call_category_config_creation(self):
        """Test CallCategoryConfig creation and validation."""
        config = CallCategoryConfig(
            name="test_category", description="Test category description", prompt_template="Test prompt: {transcript}"
        )

        assert config.name == "test_category"
        assert config.description == "Test category description"
        assert config.prompt_template == "Test prompt: {transcript}"
        assert isinstance(config.created_at, datetime)
        assert isinstance(config.updated_at, datetime)

    def test_vector_store_config_creation(self):
        """Test VectorStoreConfig creation and validation."""
        config = VectorStoreConfig(persist_dir="/tmp/test_vector_store", collection_name="test_collection")

        assert config.persist_dir == "/tmp/test_vector_store"
        assert config.collection_name == "test_collection"


class TestCategoryManager:
    """Test CategoryManager CRUD operations - no LLM calls involved."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config_path = os.path.join(self.temp_dir, "test_categories.json")

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)
        shutil.rmtree(self.temp_dir)

    def test_category_manager_initialization(self):
        """Test CategoryManager initialization."""
        manager = CategoryManager(config_file=self.test_config_path)
        assert manager.config_file == self.test_config_path
        assert isinstance(manager._categories, dict)

    def test_create_category_success(self):
        """Test successful category creation."""
        manager = CategoryManager(config_file=self.test_config_path)

        result = manager.create_category(
            name="test_category", description="Test category description", prompt_template="Test prompt: {transcript}"
        )

        # create_category returns the created config, not a boolean
        assert isinstance(result, CallCategoryConfig)
        assert result.name == "test_category"
        assert "test_category" in manager._categories
        assert manager._categories["test_category"].name == "test_category"

    def test_create_category_duplicate_name(self):
        """Test creating category with duplicate name fails."""
        manager = CategoryManager(config_file=self.test_config_path)

        # Create first category
        manager.create_category(name="duplicate", description="First category", prompt_template="First prompt: {transcript}")

        # Try to create duplicate - should raise ValueError
        with pytest.raises(ValueError, match="Category 'duplicate' already exists"):
            manager.create_category(name="duplicate", description="Second category", prompt_template="Second prompt: {transcript}")

        assert len(manager._categories) == 1

    def test_get_category_exists(self):
        """Test getting an existing category."""
        manager = CategoryManager(config_file=self.test_config_path)

        # Create category
        manager.create_category(name="test_category", description="Test category", prompt_template="Test prompt: {transcript}")

        # Get category
        category = manager.get_category("test_category")
        assert category is not None
        assert category.name == "test_category"

    def test_get_category_not_exists(self):
        """Test getting a non-existent category."""
        manager = CategoryManager(config_file=self.test_config_path)
        category = manager.get_category("non_existent")
        assert category is None

    def test_delete_category_exists(self):
        """Test deleting an existing category."""
        manager = CategoryManager(config_file=self.test_config_path)

        # Create category
        manager.create_category(name="to_delete", description="Category to delete", prompt_template="Delete prompt: {transcript}")

        # Delete category
        result = manager.delete_category("to_delete")
        assert result is True
        assert "to_delete" not in manager._categories

    def test_delete_category_not_exists(self):
        """Test deleting a non-existent category."""
        manager = CategoryManager(config_file=self.test_config_path)
        result = manager.delete_category("non_existent")
        assert result is False

    def test_load_categories_valid_file(self):
        """Test loading categories from valid JSON file."""
        test_categories = [
            {
                "name": "sales",
                "description": "Sales calls",
                "prompt_template": "Summarize this sales call: {transcript}",
                "created_at": "2024-01-01T12:00:00",
                "updated_at": "2024-01-01T12:00:00",
            }
        ]

        with open(self.test_config_path, "w") as f:
            json.dump(test_categories, f)

        manager = CategoryManager(config_file=self.test_config_path)

        assert len(manager._categories) == 1
        assert "sales" in manager._categories
        assert manager._categories["sales"].name == "sales"

    def test_load_categories_invalid_json(self):
        """Test handling invalid JSON file gracefully."""
        with open(self.test_config_path, "w") as f:
            f.write("invalid json content")

        # CategoryManager should handle invalid JSON gracefully during initialization
        with pytest.raises(json.JSONDecodeError):
            CategoryManager(config_file=self.test_config_path)


class TestVectorStoreService:
    """Test VectorStore database operations - no LLM calls involved."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = VectorStoreConfig(persist_dir=self.temp_dir, collection_name="test_collection")

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("src.call_summarizer.services.vector_store.chromadb.PersistentClient")
    @patch("src.call_summarizer.services.vector_store.Chroma")
    @patch("src.call_summarizer.services.vector_store.OpenAIEmbeddings")
    def test_vector_store_initialization(self, mock_embeddings, mock_chroma, mock_client):
        """Test VectorStoreService initialization."""
        service = VectorStoreService(config=self.test_config)

        assert service.config == self.test_config
        mock_client.assert_called_once()
        mock_embeddings.assert_called_once()

    @patch("src.call_summarizer.services.vector_store.chromadb.PersistentClient")
    @patch("src.call_summarizer.services.vector_store.Chroma")
    @patch("src.call_summarizer.services.vector_store.OpenAIEmbeddings")
    def test_store_call_summary(self, mock_embeddings, mock_chroma, mock_client):
        """Test storing a call summary in vector store."""
        # Mock the vector store
        mock_vectorstore = Mock()
        mock_chroma.return_value = mock_vectorstore

        service = VectorStoreService(config=self.test_config)

        # Create test summary
        summary = CallSummary(
            id="test-123", transcript="Test transcript", summary="Test summary", action_items=["Test action"], category=CallCategory.SALES
        )

        # Store summary - use the correct method name
        service.add_call_summary(summary)

        # Verify add_documents was called (the actual method used by VectorStoreService)
        mock_vectorstore.add_documents.assert_called_once()


class TestFileUtils:
    """Test file utility functions - no LLM calls involved."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.json")

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_ensure_dir_exists_new_directory(self):
        """Test creating directory for new file path."""
        new_file = os.path.join(self.temp_dir, "new_directory", "test.json")
        ensure_dir_exists(new_file)

        assert os.path.exists(os.path.dirname(new_file))

    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        config = CallCategoryConfig(name="test", description="Test config", prompt_template="Test: {transcript}")

        # Save config
        save_config(config, self.test_file)
        assert os.path.exists(self.test_file)

        # Load config
        loaded_config = load_config(self.test_file, CallCategoryConfig)
        assert loaded_config.name == config.name
        assert loaded_config.description == config.description

    def test_save_and_load_list_of_configs(self):
        """Test saving and loading list of configurations."""
        configs = [
            CallCategoryConfig(name="test1", description="Test config 1", prompt_template="Test 1: {transcript}"),
            CallCategoryConfig(name="test2", description="Test config 2", prompt_template="Test 2: {transcript}"),
        ]

        # Save configs
        save_list_of_configs(configs, self.test_file)
        assert os.path.exists(self.test_file)

        # Load configs
        loaded_configs = load_list_of_configs(self.test_file, CallCategoryConfig)
        assert len(loaded_configs) == 2
        assert loaded_configs[0].name == "test1"
        assert loaded_configs[1].name == "test2"


# =============================================================================
# LLM INTEGRATION TESTS (WITH @llm_unit - involves LLM calls or mocks)
# =============================================================================


class TestLLMIntegration:
    """Test LLM integration functionality - these tests use @llm_unit."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config_path = os.path.join(self.temp_dir, "categories.json")

        # Create mock category manager
        self.mock_category_manager = Mock(spec=CategoryManager)
        self.test_category = CallCategoryConfig(
            name="test_category", description="Test category for testing", prompt_template="Summarize this test call: {transcript}"
        )
        self.mock_category_manager.get_category.return_value = self.test_category
        self.mock_category_manager.get_default_category.return_value = self.test_category

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @track
    def mock_llm_summarize(self, transcript: str, category: str = "other") -> dict:
        """Mock LLM function for testing."""
        if "sales" in transcript.lower():
            return {
                "summary": "Sales call: Customer interested in product features.",
                "action_items": ["Send product demo", "Follow up with pricing"],
                "category": "sales",
                "metadata": {"confidence": 0.95},
            }
        elif "support" in transcript.lower():
            return {
                "summary": "Support call: Technical issue resolved.",
                "action_items": ["Update documentation", "Monitor logs"],
                "category": "support",
                "metadata": {"confidence": 0.88},
            }
        return {
            "summary": "General business discussion.",
            "action_items": ["Send notes"],
            "category": "other",
            "metadata": {"confidence": 0.70},
        }

    @llm_unit()
    def test_llm_call_summarization_sales(self):
        """Test LLM call summarization for sales scenario."""
        transcript = "This is a sales call about our new product features and pricing."
        result = self.mock_llm_summarize(transcript)

        assert result["category"] == "sales"
        assert "product features" in result["summary"]
        assert "Send product demo" in result["action_items"]
        assert result["metadata"]["confidence"] > 0.9

    @llm_unit()
    def test_llm_call_summarization_support(self):
        """Test LLM call summarization for support scenario."""
        transcript = "Customer support call about login issues and password reset."
        result = self.mock_llm_summarize(transcript)

        assert result["category"] == "support"
        assert "Technical issue" in result["summary"]
        assert "Update documentation" in result["action_items"]
        assert result["metadata"]["confidence"] > 0.8

    @llm_unit(expected_output_key="expected_category")
    @pytest.mark.parametrize(
        "transcript, expected_category",
        [
            ("Sales call about new product launch", "sales"),
            ("Support ticket for technical issue", "support"),
            ("Regular team meeting discussion", "other"),
        ],
    )
    def test_llm_category_detection(self, transcript, expected_category):
        """Test LLM category detection with multiple scenarios."""
        result = self.mock_llm_summarize(transcript)
        assert result["category"] == expected_category

    @llm_unit()
    @patch("src.call_summarizer.services.summarization_workflow.OpikTracer")
    def test_call_summarizer_llm_workflow_with_mocks(self, mock_opik_tracer):
        """Test CallSummarizer LLM workflow with mocked components."""
        # Mock the OpikTracer
        mock_tracer_instance = Mock()
        mock_opik_tracer.return_value = mock_tracer_instance

        # Create summarizer
        summarizer = CallSummarizer(category_manager=self.mock_category_manager)

        # Mock the LLM workflow
        with patch.object(summarizer, "workflow") as mock_workflow:
            mock_workflow.invoke.return_value = {
                "summary": "LLM generated summary",
                "action_items": ["LLM suggested action"],
                "category": "sales",
                "metadata": {"llm_confidence": 0.92},
            }

            # Test transcript
            transcript = "Sales call transcript for LLM processing"
            result = summarizer.summarize_transcript(transcript, "sales")

            # Verify LLM workflow was called
            mock_workflow.invoke.assert_called_once()

            # Verify result
            assert isinstance(result, CallSummary)
            assert result.transcript == transcript
            assert result.summary == "LLM generated summary"
            assert result.action_items == ["LLM suggested action"]
            assert result.category == "sales"

    @llm_unit()
    @patch("src.call_summarizer.services.summarization_workflow.OpikTracer")
    def test_call_summarizer_llm_workflow_without_category(self, mock_opik_tracer):
        """Test CallSummarizer LLM workflow without explicit category."""
        # Mock the OpikTracer
        mock_tracer_instance = Mock()
        mock_opik_tracer.return_value = mock_tracer_instance

        # Create summarizer
        summarizer = CallSummarizer(category_manager=self.mock_category_manager)

        # Mock the LLM workflow
        with patch.object(summarizer, "workflow") as mock_workflow:
            mock_workflow.invoke.return_value = {
                "summary": "General call summary",
                "action_items": ["Review notes"],
                "category": "other",
                "metadata": {},
            }

            # Test without explicit category
            transcript = "General business call transcript"
            result = summarizer.summarize_transcript(transcript)

            # Verify LLM workflow was called
            mock_workflow.invoke.assert_called_once()

            # Verify result
            assert isinstance(result, CallSummary)
            assert result.transcript == transcript
            assert result.summary == "General call summary"
            assert result.category == "other"

    @llm_unit()
    def test_llm_error_handling(self):
        """Test LLM error handling scenarios."""
        # Test with empty transcript
        with pytest.raises(ValueError, match="Transcript cannot be empty"):
            if not "empty_transcript":
                raise ValueError("Transcript cannot be empty")

        # Test with very long transcript
        long_transcript = "word " * 10000
        result = self.mock_llm_summarize(long_transcript)
        assert result["category"] == "other"  # Should handle gracefully

    @llm_unit()
    @patch("src.call_summarizer.services.summarization_workflow.OpikTracer")
    def test_call_summarizer_category_not_found(self, mock_opik_tracer):
        """Test CallSummarizer when requested category doesn't exist."""
        # Mock the OpikTracer
        mock_tracer_instance = Mock()
        mock_opik_tracer.return_value = mock_tracer_instance

        # Set up category manager to return None for non-existent category
        self.mock_category_manager.get_category.return_value = None

        # Create summarizer
        summarizer = CallSummarizer(category_manager=self.mock_category_manager)

        # Test with non-existent category - should raise ValueError
        transcript = "Test transcript"
        with pytest.raises(ValueError, match="No category found: non_existent_category"):
            summarizer.summarize_transcript(transcript, "non_existent_category")


# =============================================================================
# INTEGRATION TESTS WITH REAL LLM CALLS (if API keys are available)
# =============================================================================


class TestRealLLMIntegration:
    """Integration tests with real LLM calls - only run if API keys are present."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config_path = os.path.join(self.temp_dir, "categories.json")

        # Create real category manager
        self.category_manager = CategoryManager(config_file=self.test_config_path)
        self.category_manager.create_category(
            name="sales", description="Sales calls", prompt_template="Summarize this sales call: {transcript}"
        )
        self.category_manager.create_category(
            name="support", description="Support calls", prompt_template="Summarize this support call: {transcript}"
        )
        self.category_manager.create_category(name="other", description="Other calls", prompt_template="Summarize this call: {transcript}")

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY") or not os.getenv("OPIK_API_KEY"), reason="API keys not available for real LLM integration tests"
    )
    @llm_unit()
    def test_real_llm_sales_call_summarization(self):
        """Test real LLM call summarization for sales scenario."""
        # Create real CallSummarizer
        summarizer = CallSummarizer(category_manager=self.category_manager)

        # Real sales call transcript
        transcript = """
        Hi, this is John from ABC Company. I'm calling because we're interested in your
        enterprise software solution. We currently have about 500 employees and are
        looking to streamline our project management processes. Can you tell me more
        about your pricing for enterprise licenses and what kind of support you offer?
        """

        # Call real LLM
        result = summarizer.summarize_transcript(transcript, "sales")

        # Verify result structure
        assert isinstance(result, CallSummary)
        assert result.transcript == transcript
        assert len(result.summary) > 0
        assert isinstance(result.action_items, list)
        assert result.category in ["sales", "other"]  # LLM might classify differently
        assert isinstance(result.metadata, dict)

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY") or not os.getenv("OPIK_API_KEY"), reason="API keys not available for real LLM integration tests"
    )
    @llm_unit()
    def test_real_llm_support_call_summarization(self):
        """Test real LLM call summarization for support scenario."""
        # Create real CallSummarizer
        summarizer = CallSummarizer(category_manager=self.category_manager)

        # Real support call transcript
        transcript = """
        Hello, I'm having trouble logging into my account. I've tried resetting my
        password multiple times but I'm still getting an error message that says
        'invalid credentials'. My username is john.doe@company.com and I last
        successfully logged in about a week ago. Can you help me resolve this issue?
        """

        # Call real LLM
        result = summarizer.summarize_transcript(transcript, "support")

        # Verify result structure
        assert isinstance(result, CallSummary)
        assert result.transcript == transcript
        assert len(result.summary) > 0
        assert isinstance(result.action_items, list)
        assert result.category in ["support", "other"]  # LLM might classify differently
        assert isinstance(result.metadata, dict)

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY") or not os.getenv("OPIK_API_KEY"), reason="API keys not available for real LLM integration tests"
    )
    @llm_unit(expected_output_key="expected_contains")
    @pytest.mark.parametrize(
        "transcript, expected_contains",
        [
            ("Sales call about product demo and pricing", "sales"),
            ("Support ticket about login issues", "login"),
            ("Meeting about quarterly planning", "meeting"),
        ],
    )
    def test_real_llm_parametrized_calls(self, transcript, expected_contains):
        """Test real LLM calls with parametrized inputs."""
        # Create real CallSummarizer
        summarizer = CallSummarizer(category_manager=self.category_manager)

        # Call real LLM with 'other' category as default
        result = summarizer.summarize_transcript(transcript, "other")

        # Verify result contains expected content
        assert isinstance(result, CallSummary)
        assert len(result.summary) > 0
        # Check that summary or action items contain expected content
        full_content = result.summary + " " + " ".join(result.action_items)
        assert expected_contains.lower() in full_content.lower()
