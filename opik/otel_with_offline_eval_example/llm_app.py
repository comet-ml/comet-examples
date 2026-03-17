"""
FILE 1: llm_app.py
Your application code — uses OpenTelemetry for tracing.
No Opik SDK required here.

This is the code that runs in production or dev environments.
OTel spans are exported to Opik via the OTLP endpoint.

Setup:
    pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http

Environment variables:
    OPIK_OTLP_ENDPOINT  - e.g. https://your-opik-host/opik/api/v1/private/otel/v1/traces
    OPIK_API_KEY        - your Opik API key
    OPIK_WORKSPACE      - your Opik workspace name
"""

import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter


def setup_otel():
    """Configure OTel to export traces to Opik."""
    otlp_endpoint = os.environ["OPIK_OTLP_ENDPOINT"]
    api_key = os.environ["OPIK_API_KEY"]
    workspace = os.environ["OPIK_WORKSPACE"]

    exporter = OTLPSpanExporter(
        endpoint=otlp_endpoint,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Comet-Workspace": workspace,
        },
    )

    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)


tracer = trace.get_tracer(__name__)


def call_llm(prompt: str) -> str:
    """
    Your LLM call — wrapped in an OTel span for tracing.
    Replace this with your actual model call (OpenAI, vLLM, etc.)
    """
    with tracer.start_as_current_span("llm_call") as span:
        span.set_attribute("input.prompt", prompt)

        # --- replace with your actual LLM call ---
        response = f"[LLM response to: {prompt}]"
        # -----------------------------------------

        span.set_attribute("output.response", response)
        return response


def answer_question(user_question: str) -> str:
    """
    Your application logic — also traced via OTel.
    This is the function you'll reuse in evaluate.py.
    """
    with tracer.start_as_current_span("answer_question") as span:
        span.set_attribute("input.question", user_question)
        answer = call_llm(user_question)
        span.set_attribute("output.answer", answer)
        return answer


if __name__ == "__main__":
    # Set up OTel when running as the application entry point
    setup_otel()
    result = answer_question("What is the capital of France?")
    print(result)
