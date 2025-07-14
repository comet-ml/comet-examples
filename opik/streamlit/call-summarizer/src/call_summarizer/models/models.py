"""Data models for the call summarizer application."""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List

from pydantic import BaseModel, Field


class CallCategory(str, Enum):
    """Categories for call summaries."""

    SALES = "sales"
    SUPPORT = "support"
    INTERVIEW = "interview"
    MEETING = "meeting"
    OTHER = "other"


class CallSummary(BaseModel):
    """Model for call summary data."""

    id: str = Field(..., description="Unique identifier for the call summary")
    transcript: str = Field(..., description="The full text of the call transcript")
    summary: str = Field(..., description="Generated summary of the call")
    action_items: List[str] = Field(default_factory=list, description="List of action items from the call")
    category: CallCategory = Field(default=CallCategory.OTHER, description="Category of the call")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When the summary was created")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata about the call")


class CallCategoryConfig(BaseModel):
    """Configuration for a call category including its prompt template."""

    name: str = Field(..., description="Name of the category")
    description: str = Field(..., description="Description of when to use this category")
    prompt_template: str = Field(..., description="Template for generating summaries")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When the category was created")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When the category was last updated")


class VectorStoreConfig(BaseModel):
    """Configuration for the vector store."""

    persist_dir: str = Field(..., description="Directory to persist the vector store")
    collection_name: str = Field("call_summaries", description="Name of the collection in the vector store")
