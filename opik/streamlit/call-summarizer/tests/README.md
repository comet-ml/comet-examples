# Call Summarizer Test Suite

This directory contains comprehensive pytest tests for the call summarizer application with Opik integration for LLM test tracking.

## Quick Start

```bash
# Run all tests with Opik integration (uses your .env credentials)
poetry run python run_tests.py

# Run tests locally without Opik
poetry run python run_tests.py --no-opik --profile local

# Run with coverage report
poetry run python run_tests.py --coverage

# Run specific test types
poetry run python run_tests.py --type unit
poetry run python run_tests.py --type integration

# Run tests in parallel
poetry run python run_tests.py --parallel
```

## Configuration

All test configuration is now centralized in `pyproject.toml` under the `[tool.pytest.ini_options]` section. This follows Poetry best practices and eliminates the need for separate `.ini` files.

### Test Profiles

- **default**: Standard test run with balanced verbosity
- **local**: Development-friendly with verbose output and disabled warnings
- **ci**: CI/CD optimized with JUnit XML output and minimal verbosity

### Environment Variables

The test runner uses `python-dotenv` to automatically load your `.env` file, ensuring your real Opik and OpenAI API keys are used for integration testing.

## Test Organization

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_models.py           # Pydantic model tests
├── test_category_manager.py # Category management tests
├── test_summarization_workflow.py # LLM workflow tests
├── test_vector_store.py     # ChromaDB vector store tests
├── test_utils.py           # Utility function tests
└── README.md               # This file
```

## Opik Integration

All tests use the `@llm_unit()` decorator from Opik for LLM test tracking. Test results are automatically sent to your Opik dashboard at https://www.comet.com/opik/.

### Opik Configuration

Tests use your real Opik credentials from `.env`:
- `OPIK_API_KEY`: Your Opik API key
- `OPIK_WORKSPACE`: Your workspace name
- `OPIK_PROJECT_NAME`: Project name for test tracking

### Opik Decorators

- `@llm_unit`: Decorator for LLM unit tests
- `@track`: Decorator for tracking LLM calls

## Coverage

Coverage configuration is in `pyproject.toml` under `[tool.coverage.*]` sections:
- **Source**: `src/call_summarizer`
- **Reports**: HTML (htmlcov/), Terminal, XML
- **Target**: 80% minimum coverage

## Running Tests with Poetry

Since all configuration is in `pyproject.toml`, you can also run tests directly with Poetry:

```bash
# Basic test run
poetry run pytest

# With coverage
poetry run pytest --cov=src/call_summarizer --cov-report=html

# Specific markers
poetry run pytest -m unit
poetry run pytest -m integration
poetry run pytest -m "not slow"

# Parallel execution
poetry run pytest -n auto
```

## Test Markers

- `unit`: Fast unit tests with mocked dependencies
- `integration`: Integration tests with real services
- `slow`: Tests that take longer to run
- `llm`: Tests involving LLM functionality

## Troubleshooting

### Opik Authentication Errors
If you see 401 errors from Opik, verify your `.env` file contains valid credentials:
```bash
OPIK_API_KEY="your-real-api-key"
OPIK_WORKSPACE="your-workspace"
```

### Test Failures
Use different profiles for debugging:
```bash
# More verbose output for debugging
poetry run python run_tests.py --profile local --verbose

# Run specific failing test
poetry run pytest tests/test_specific.py::TestClass::test_method -v
```

### Coverage Issues
Generate detailed coverage reports:
```bash
poetry run python run_tests.py --coverage
open htmlcov/index.html  # View detailed HTML report
```

## Best Practices

1. **Use Poetry**: All commands should use `poetry run` for consistency
2. **Environment Variables**: Always use `.env` for real credentials
3. **Test Isolation**: Each test should be independent and use proper mocking
4. **Opik Integration**: Use `@llm_unit()` decorator for all LLM-related tests
5. **Markers**: Tag tests appropriately for selective execution
6. **Coverage**: Aim for 80%+ coverage with meaningful tests

## CI/CD Integration

For continuous integration, use the CI profile:
```bash
poetry run python run_tests.py --profile ci --coverage --parallel
```

This generates JUnit XML output and optimized reporting for CI systems.
