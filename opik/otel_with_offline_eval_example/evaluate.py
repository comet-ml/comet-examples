"""
FILE 2: evaluate.py
Separate offline evaluation script — uses the Opik Python SDK.
Import your application function directly and wrap it as an Opik task.

This does NOT replace OTel tracing in your app. It runs offline experiments
against a dataset and records scored results as an Experiment in Opik.

Setup:
    pip install opik

Environment variables:
    OPIK_API_KEY        - your Opik API key
    OPIK_WORKSPACE      - your Opik workspace name
    OPIK_URL_OVERRIDE   - your self-hosted Opik URL
                          (e.g. https://your-opik-host/opik/api)
"""

import opik
from opik.evaluation.metrics import Equals, Hallucination
# Note: Hallucination is an LLM-judge metric — it requires an LLM provider to be
# configured (e.g. set OPENAI_API_KEY). Replace or remove it if you don't have one.

# Import the function from your application code.
# opik.evaluate() will call this function for each dataset item.
from llm_app import answer_question


# --- Step 1: Create (or get) a dataset ---

client = opik.Opik()

dataset = client.get_or_create_dataset("capital-cities-qa")
dataset.insert([
    {"question": "What is the capital of France?",   "expected": "Paris"},
    {"question": "What is the capital of Japan?",    "expected": "Tokyo"},
    {"question": "What is the capital of Germany?",  "expected": "Berlin"},
])


# --- Step 2: Define the evaluation task ---
# This is a thin wrapper that maps dataset item fields to your app function.
# The dict keys returned here are used by scoring metrics below.

def evaluation_task(dataset_item: dict) -> dict:
    answer = answer_question(dataset_item["question"])
    return {
        "output": answer,
        "reference": dataset_item["expected"],  # passed to Equals metric
        "input": dataset_item["question"],       # passed to Hallucination metric
    }


# --- Step 3: Run the experiment ---

opik.evaluate(
    dataset=dataset,
    task=evaluation_task,
    scoring_metrics=[
        Equals(name="exact_match"),
        Hallucination(name="hallucination"),
    ],
    experiment_name="capital-cities-baseline",
    project_name="mediatek-llm-eval",
)
