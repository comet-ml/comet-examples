"""Utility functions for file operations."""

import json
import os
from typing import List, Optional, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def ensure_dir_exists(file_path: str) -> None:
    """Ensure the directory of the given file path exists."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)


def save_config(data: BaseModel, file_path: str) -> None:
    """Save a Pydantic model to a JSON file."""
    ensure_dir_exists(file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data.dict(), f, indent=2, default=str)


def load_config(file_path: str, model_class: Type[T]) -> Optional[T]:
    """Load a Pydantic model from a JSON file."""
    if not os.path.exists(file_path):
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return model_class(**data)


def save_list_of_configs(data_list: List[BaseModel], file_path: str) -> None:
    """Save a list of Pydantic models to a JSON file."""
    ensure_dir_exists(file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump([item.dict() for item in data_list], f, indent=2, default=str)


def load_list_of_configs(file_path: str, model_class: Type[T]) -> List[T]:
    """Load a list of Pydantic models from a JSON file."""
    if not os.path.exists(file_path):
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    return [model_class(**item) for item in data_list]


def read_text_file(file_path: str) -> str:
    """Read text content from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def write_text_file(file_path: str, content: str) -> None:
    """Write text content to a file."""
    ensure_dir_exists(file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
