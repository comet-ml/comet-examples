"""Application configuration and settings."""
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings."""
    
    # OpenAI
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    
    # Application
    vector_store_path: str = Field("./data/vector_store", alias="VECTOR_STORE_PATH")
    data_dir: str = Field("./data", alias="DATA_DIR")
    
    # Opik
    opik_api_key: Optional[str] = Field(default=None, alias="OPIK_API_KEY")
    opik_workspace: Optional[str] = Field(default=None, alias="OPIK_WORKSPACE")
    opik_project_name: Optional[str] = Field(default=None, alias="OPIK_PROJECT_NAME")
    
    # Pydantic v2 config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    def ensure_dirs_exist(self) -> None:
        """Ensure that all required directories exist."""
        os.makedirs(self.vector_store_path, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)


# Initialize settings
settings = Settings()
settings.ensure_dirs_exist()
