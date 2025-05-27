#!/usr/bin/env python3
"""Initialize the application by creating necessary directories and default categories."""
import os
from pathlib import Path

from src.call_summarizer.config import settings
from src.call_summarizer.services.category_manager import CategoryManager

def main():
    """Initialize the application."""
    print("🚀 Initializing Call Summarizer...")
    
    # Ensure data directories exist
    os.makedirs(settings.data_dir, exist_ok=True)
    os.makedirs(settings.vector_store_path, exist_ok=True)
    
    # Initialize category manager and ensure default categories exist
    print("📂 Setting up default categories...")
    category_manager = CategoryManager()
    category_manager.ensure_default_categories_exist()
    
    print("✅ Initialization complete!")
    print(f"\nTo start the application, run:")
    print("  streamlit run app.py\n")

if __name__ == "__main__":
    main()
