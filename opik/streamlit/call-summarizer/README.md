# 📞 Call Summarizer

A Streamlit application for summarizing and categorizing call transcripts using LangGraph and LangChain. This application helps you quickly extract key information, action items, and insights from call recordings or transcripts.

## Features

- **Automatic Call Summarization**: Generate concise summaries of call transcripts
- **Smart Categorization**: Automatically categorize calls into predefined categories
- **Action Item Extraction**: Identify and extract action items from calls
- **Vector Search**: Semantic search through past call summaries
- **Call History**: View, search, and manage your call history
- **Customizable Categories**: Define custom call categories with specific summarization prompts
- **Export Options**: Export summaries in various formats (TXT, JSON, CSV)
- **Responsive Design**: Works on both desktop and mobile devices
- **Opik Integration**: Comprehensive tracing of all LLM interactions for monitoring and debugging
- **Natural Language Query**: Chat with your call data using natural language
- **Persistent Storage**: All data is stored locally for privacy
- **Call History Management**: View and manage your call history with the ability to delete entries

## 🚀 Getting Started

### Prerequisites

- Python 3.12 or higher (recommended)
- [Poetry](https://python-poetry.org/) for dependency management
- [OpenAI API key](https://platform.openai.com/api-keys)
- [Opik API key](https://www.comet.com/signup?utm_source=call-summarizer&utm_medium=referral&utm_campaign=github) (optional but recommended for tracing)
- (Optional) [pyenv](https://github.com/pyenv/pyenv) for Python version management
- (Recommended) [direnv](https://direnv.net/) for environment variable management

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/call-summarizer.git
   cd call-summarizer
   ```

2. Install Python dependencies with Poetry:

   ```bash
   poetry install
   ```

3. Copy the example environment file and update with your API keys:

   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. Activate the virtual environment:

   ```bash
   poetry shell
   ```

## Configuration

The application can be configured using environment variables. Create a `.env` file in the project root with the following variables:

```ini
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional - Opik for LLM tracing
OPIK_API_KEY=your_opik_api_key_here

# Storage paths
VECTOR_STORE_PATH=./data/vector_store
DATA_DIR=./data
```

## Opik Integration

The application includes built-in support for Opik tracing, which provides:

- Detailed tracing of all LLM calls
- Performance metrics and token usage
- Debugging information for prompt engineering
- Conversation history and chain visualization

To enable Opik tracing:

1. Sign up for a free account at [Opik](https://www.comet.com/signup)
2. Get your API key from the dashboard
3. Add it to your `.env` file
4. Restart the application

All LLM interactions will now be traced and visible in your Opik dashboard.

### Running the Application

1. Start the Streamlit application:

   ```bash
   streamlit run app.py
   ```

2. Open your browser to the URL shown in the terminal (usually http://localhost:8501)

### Development

This project uses [Poetry](https://python-poetry.org/) for dependency management. For code quality and consistency, we use:

- **[Ruff](https://github.com/astral-sh/ruff)**: For extremely fast Python linting and formatting (replaces Black, isort, Flake8).
- **[pre-commit](https://pre-commit.com/)**: For managing and maintaining multi-language pre-commit hooks.
- **[Commitizen](https://commitizen-tools.github.io/commitizen/)**: For standardized commit messages (following Conventional Commits) and automated version bumping.

**1. Install Development Dependencies:**
   This includes tools like Ruff, pre-commit, Commitizen, and pytest.

   ```bash
   poetry install --with dev
   ```

**2. Set Up Pre-commit Hooks:**
After installing dependencies, activate the pre-commit hooks in your local repository. These hooks will run automatically before each commit.

```bash
poetry run pre-commit install --hook-type commit-msg --hook-type pre-commit
```

The configured hooks will:

- Format your code using Ruff.
- Lint your code using Ruff.
- Check your commit message format using Commitizen.
- Perform other general checks (e.g., for trailing whitespace, valid YAML/TOML).

**3. Running Linters and Formatters Manually:**
While pre-commit hooks handle this automatically, you can also run these tools manually:

- **Format code with Ruff:**

```bash
poetry run ruff format .
```

- **Check for linting issues with Ruff (and autofix where possible):**

```bash
poetry run ruff check . --fix
```

**4. Running Tests:**

```bash
poetry run pytest
```

**5. Commit Guidelines (Conventional Commits):**
This project adheres to the [Conventional Commits](https://www.conventionalcommits.org/) specification. This practice helps in creating an explicit commit history, which is useful for automated version bumping and changelog generation.

To make a commit:

1. Stage your changes (`git add <files>...`).
2. Run `git commit`.
   - The pre-commit hooks will execute first. If any hook fails (e.g., Ruff finds formatting issues it can't fix automatically, or your commit message doesn't conform), the commit will be aborted. Address the issues and try committing again.
   - If all pre-commit hooks pass, Commitizen will prompt you interactively to build a conventional commit message.
   Alternatively, you can directly use Commitizen's interactive prompt for committing:

```bash
poetry run cz c
```

This command also respects pre-commit hooks if they are installed.

**6. Release Process (Versioning):**
When you're ready to release a new version of the project (e.g., after merging significant features or fixes):

1. Ensure your local main branch (or your primary development branch) is up-to-date with the remote repository and that your working directory is clean (no uncommitted changes).
2. Run the Commitizen bump command:

   ```bash
   poetry run cz bump --changelog
   ```

   This command performs several actions:

   - Analyzes your commit history since the last version tag.
   - Determines the appropriate new version (patch, minor, or major) based on your conventional commits.
   - Updates the version string in `pyproject.toml` (under `tool.poetry.version`) and `src/call_summarizer/__init__.py` (the `__version__` attribute).
   - Creates a new commit with these version changes.
   - Tags the new commit with the new version number (e.g., `v0.2.0`).
   - The `--changelog` flag will also attempt to generate or update a changelog file. (Note: For more advanced changelog generation, you might need to configure `commitizen` further, for example, by setting `update_changelog_on_bump = true` and specifying a changelog file in `pyproject.toml` if you want it automatically updated).

3. Push the new commit and the new tag to the remote repository:

```bash
git push --follow-tags
```

This process ensures that versioning is consistent and automated.

## 🛠️ Usage

### Summarize a Call

1. Click on "Summarize Call" in the sidebar
2. Upload a text file with the call transcript or paste it directly
3. Select a category for the call
4. Click "Summarize Call" to generate the summary and action items

### View Call History

1. Click on "View History" in the sidebar
2. Browse through previous call summaries
3. Filter by category if needed

### Manage Categories

1. Click on "Manage Categories" in the sidebar
2. View existing categories and their prompt templates
3. Add new categories with custom prompts
4. Delete categories you no longer need

### Chat with Call Data

1. Click on "Chat with Data" in the sidebar
2. Ask questions about your call history in natural language
3. The AI will search through your call summaries to provide relevant answers

## 🏗️ Project Structure

```text
call-summarizer/
├── .env.example           # Example environment variables
├── app.py                 # Main Streamlit application
├── init_app.py           # Application initialization script
├── pyproject.toml        # Project dependencies and configuration (Poetry)
├── README.md             # This file
└── src/
    └── call_summarizer/
        ├── __init__.py
        ├── config.py      # Application configuration
        ├── models/        # Data models
        │   └── models.py
        ├── services/      # Business logic
        │   ├── category_manager.py
        │   ├── summarization_workflow.py
        │   └── vector_store.py
        └── utils/         # Utility functions
            └── file_utils.py
```

## 🤖 Technologies Used

- **Streamlit**: Web application framework
- **LangGraph**: For creating the summarization workflow
- **LangChain**: For LLM integration and prompt engineering
- **OpenAI**: For generating summaries and action items
- **ChromaDB**: Vector database for storing and searching call summaries
- **LlamaIndex**: For semantic search and retrieval
- **Opik**: For LLM tracing and debugging
- **Poetry**: Dependency management
- **Pydantic**: Data validation and settings management

## 🙏 Acknowledgments

- Built with ❤️ using amazing open-source libraries
- Inspired by the need for better call analysis tools
- Special thanks to the LangChain and ChromaDB communities for their great tools

---

Happy summarizing! 🎉
