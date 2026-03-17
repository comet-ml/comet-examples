# OTel Tracing + Opik Offline Evaluation

Demonstrates how to use OpenTelemetry for application tracing **alongside**
Opik's offline evaluation workflow. These are complementary tools, not alternatives.

## Architecture

```
┌─────────────────────────────────────┐    ┌───────────────────────────────────────┐
│         llm_app.py                  │    │           evaluate.py                 │
│  (your application code)            │    │   (offline evaluation script)         │
│                                     │    │                                       │
│  - OTel for tracing                 │    │  - Opik Python SDK                    │
│  - Runs in production/dev           │    │  - Runs separately (CI, scheduled)    │
│  - Sends spans → Opik OTLP endpoint │    │  - Imports answer_question() from app │
│                                     │    │  - Runs task against a dataset        │
│                                     │    │  - Creates an Experiment in Opik      │
└─────────────────────────────────────┘    └───────────────────────────────────────┘
         │                                              │
         └──────────────────────┬───────────────────────┘
                                ▼
                    ┌─────────────────────┐
                    │       Opik          │
                    │                    │
                    │  Traces (from OTel) │
                    │  Experiments (from  │
                    │  evaluate())        │
                    └─────────────────────┘
```

## Key Insight

| | OTel tracing | `opik.evaluate()` |
|---|---|---|
| Purpose | Observe your app in real-time | Test quality offline against a dataset |
| When it runs | During application execution | In a separate evaluation script |
| What you get in Opik | Traces / Spans | Experiments with scored results |
| SDK | OTel SDK | Opik Python SDK (`pip install opik`) |

**There is no OTel-native way to run offline experiments.** The offline evaluation
workflow requires the Opik Python SDK because it needs to:
1. Read items from an Opik Dataset
2. Run your LLM task against each item
3. Score outputs with metrics
4. Record results as an Experiment in Opik

## Setup

```bash
pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http opik
```

Environment variables for `llm_app.py`:
```bash
export OPIK_OTLP_ENDPOINT="https://your-opik-host/opik/api/v1/private/otel/v1/traces"
export OPIK_API_KEY="your-api-key"
export OPIK_WORKSPACE="your-workspace"
```

Environment variables for `evaluate.py`:
```bash
export OPIK_API_KEY="your-api-key"
export OPIK_WORKSPACE="your-workspace"
export OPIK_URL_OVERRIDE="https://your-opik-host/opik/api"
```

## Running

```bash
# Run your application normally (OTel tracing active)
python llm_app.py

# Run offline evaluation separately
python evaluate.py
```

## Notes for self-hosted Opik

The OTel OTLP endpoint for self-hosted Opik is:
```
https://<your-opik-host>/opik/api/v1/private/otel/v1/traces
```

The Opik SDK base URL override for self-hosted is:
```
https://<your-opik-host>/opik/api
```
