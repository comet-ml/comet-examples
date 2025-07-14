#!/usr/bin/env python3
"""Test runner script for the call summarizer application."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n {description}...")
    print(f"Running: {' '.join(command)}")

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f" {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f" {description} failed")
        print(f"Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def setup_environment(use_opik: bool = True):
    """Set up environment variables for testing."""
    # Load environment variables from .env file using python-dotenv
    load_dotenv()

    if not use_opik:
        # Disable Opik for local testing by clearing the API key
        os.environ["OPIK_API_KEY"] = ""
        os.environ["OPIK_WORKSPACE"] = ""

    # Set additional test-specific environment variables
    os.environ["PYTHONPATH"] = str(Path.cwd())
    os.environ["TESTING"] = "true"

    # Set paths for testing (use test directories to avoid conflicts)
    if not os.environ.get("VECTOR_STORE_PATH"):
        os.environ["VECTOR_STORE_PATH"] = "./test_data/vector_store"
    if not os.environ.get("DATA_DIR"):
        os.environ["DATA_DIR"] = "./test_data"


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run tests for the call summarizer application")
    parser.add_argument("--type", choices=["unit", "integration", "all"], default="all", help="Type of tests to run (default: all)")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--opik", action="store_true", default=True, help="Enable Opik integration (default: True)")
    parser.add_argument("--no-opik", action="store_true", help="Disable Opik integration")
    parser.add_argument("--profile", choices=["default", "ci", "local"], default="default", help="Test profile to use (default: default)")

    args = parser.parse_args()

    # Handle opik flag logic
    if args.no_opik:
        args.opik = False

    # Change to project root
    project_root = Path(__file__).parent
    os.chdir(project_root)

    print(" Call Summarizer Test Runner")
    print("=" * 50)

    # Check if poetry is available
    if not run_command(["poetry", "--version"], "Checking Poetry installation"):
        print(" Poetry is not installed. Please install Poetry first.")
        sys.exit(1)

    # Install dependencies
    if not run_command(["poetry", "install", "--with", "dev"], "Installing dependencies"):
        sys.exit(1)

    # Set up environment variables
    setup_environment(use_opik=args.opik)

    # Build pytest command
    pytest_cmd = ["poetry", "run", "pytest"]

    # Add test directory
    pytest_cmd.append("tests/")

    # Add test type markers
    if args.type == "unit":
        pytest_cmd.extend(["-m", "unit"])
    elif args.type == "integration":
        pytest_cmd.extend(["-m", "integration"])
    # For "all", no marker filter is needed

    # Add coverage if requested
    if args.coverage:
        pytest_cmd.extend(
            ["--cov=src/call_summarizer", "--cov-report=html", "--cov-report=term", "--cov-report=xml", "--cov-fail-under=80"]
        )

    # Add parallel execution if requested
    if args.parallel:
        pytest_cmd.extend(["-n", "auto"])

    # Add profile-specific options
    if args.profile == "ci":
        # CI-specific options: less verbose, no color, JUnit XML
        pytest_cmd.extend(["--tb=short", "--no-header", "--junit-xml=test-results.xml"])
    elif args.profile == "local":
        # Local development: more verbose, disable warnings
        pytest_cmd.extend(["--tb=long", "--disable-warnings", "-v"])

    # Add verbose output if requested (overrides profile settings)
    if args.verbose:
        pytest_cmd.append("-vv")

    print(f"\n Running tests: {' '.join(pytest_cmd)}")
    print(f" Profile: {args.profile}")
    print(f" Opik integration: {'enabled' if args.opik else 'disabled'}")

    # Run tests
    try:
        _ = subprocess.run(pytest_cmd, check=True)
        print("\n All tests passed!")

        if args.coverage:
            print("\n Coverage report generated:")
            print("  - Terminal: Coverage summary shown above")
            print("  - HTML: Open htmlcov/index.html in your browser")
            print("  - XML: coverage.xml file generated")

    except subprocess.CalledProcessError as e:
        print(f"\n Tests failed with exit code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
