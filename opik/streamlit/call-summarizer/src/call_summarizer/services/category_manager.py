"""Manages call categories and their configurations."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..config import settings
from ..models.models import CallCategory, CallCategoryConfig
from ..utils.file_utils import load_list_of_configs, save_list_of_configs


class CategoryManager:
    """Manages call categories and their configurations."""

    def __init__(self, config_file: Optional[str] = None):
        """Initialize the category manager."""
        self.config_file = config_file or str(Path(settings.data_dir) / "categories.json")
        self._categories: Dict[str, CallCategoryConfig] = self._load_categories()

    def _load_categories(self) -> Dict[str, CallCategoryConfig]:
        """Load categories from the configuration file."""
        categories = load_list_of_configs(self.config_file, CallCategoryConfig)
        return {category.name.lower(): category for category in categories}

    def _save_categories(self) -> None:
        """Save categories to the configuration file."""
        save_list_of_configs(list(self._categories.values()), self.config_file)

    def create_category(self, name: str, description: str, prompt_template: str) -> CallCategoryConfig:
        """Create a new call category."""
        if name.lower() in self._categories:
            raise ValueError(f"Category '{name}' already exists")

        category = CallCategoryConfig(name=name, description=description, prompt_template=prompt_template)

        self._categories[name.lower()] = category
        self._save_categories()
        return category

    def update_category(self, name: str, **updates) -> Optional[CallCategoryConfig]:
        """Update an existing call category."""
        if name.lower() not in self._categories:
            return None

        category = self._categories[name.lower()]

        # Update fields that were provided
        for key, value in updates.items():
            if hasattr(category, key):
                setattr(category, key, value)

        # Always update the updated_at timestamp
        category.updated_at = datetime.utcnow()

        self._save_categories()
        return category

    def delete_category(self, name: str) -> bool:
        """Delete a call category."""
        if name.lower() not in self._categories:
            return False

        del self._categories[name.lower()]
        self._save_categories()
        return True

    def get_category(self, name: str) -> Optional[CallCategoryConfig]:
        """Get a call category by name."""
        return self._categories.get(name.lower())

    def list_categories(self) -> List[CallCategoryConfig]:
        """List all call categories."""
        return list(self._categories.values())

    def get_category_prompt(self, name: str) -> Optional[str]:
        """Get the prompt template for a category."""
        category = self.get_category(name)
        return category.prompt_template if category else None

    def get_default_category(self) -> Optional[CallCategoryConfig]:
        """Get the default category."""
        return self._categories.get(CallCategory.OTHER.value)

    def ensure_default_categories_exist(self) -> None:
        """Ensure that default categories exist."""
        default_categories = [
            {
                "name": CallCategory.SALES.value,
                "description": "Sales calls with potential or existing customers",
                "prompt_template": (
                    "Summarize the following sales call transcript. "
                    "Include key discussion points, customer needs, and next steps.\n\n"
                    "Transcript:\n{transcript}"
                ),
            },
            {
                "name": CallCategory.SUPPORT.value,
                "description": "Customer support or service calls",
                "prompt_template": (
                    "Summarize the following support call transcript. "
                    "Include the customer's issue, troubleshooting steps, and resolution.\n\n"
                    "Transcript:\n{transcript}"
                ),
            },
            {
                "name": CallCategory.INTERVIEW.value,
                "description": "Job interviews or candidate screenings",
                "prompt_template": (
                    "Summarize the following interview transcript. "
                    "Include key qualifications discussed, strengths, and areas for follow-up.\n\n"
                    "Transcript:\n{transcript}"
                ),
            },
            {
                "name": CallCategory.MEETING.value,
                "description": "General business meetings",
                "prompt_template": (
                    "Summarize the following meeting transcript. "
                    "Include key decisions, action items, and next steps.\n\n"
                    "Transcript:\n{transcript}"
                ),
            },
            {
                "name": CallCategory.OTHER.value,
                "description": "Other types of calls",
                "prompt_template": (
                    "Summarize the following call transcript. " "Include key points and action items.\n\n" "Transcript:\n{transcript}"
                ),
            },
        ]

        for category_data in default_categories:
            if category_data["name"].lower() not in self._categories:
                self.create_category(
                    name=category_data["name"], description=category_data["description"], prompt_template=category_data["prompt_template"]
                )
