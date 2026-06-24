# Agent instructions for comet-examples

This repo is a reference library of examples for [Comet](https://www.comet.com/site/), the ML
experiment-tracking, model-management, and observability platform. Examples instrument popular ML
frameworks (PyTorch, Keras, fastai, scikit-learn, XGBoost, Transformers, …) with `comet_ml`. Read
this file before generating or editing any code.

## Repo structure

```
comet-examples/
├── integrations/   # Add Comet to a framework, grouped by ML task (see below)
├── guides/         # How-to notebooks for Comet workflows
├── panels/         # Custom Comet panel (visualization) examples
├── notebooks/      # General/standalone notebooks
└── templates/      # Starter template (integration-example)
```

`integrations/` is grouped by ML task, then by framework, then by example:

```
integrations/<category>/<framework>/<example-name>/
```

Categories in use: `model-training`, `model-evaluation`, `model-optimization`, `model-deployment`,
`workflow-orchestration`, `reinforcement-learning`, `llm`, `data-management`.

**Where does new code belong?**

| If you are… | Put it in |
|---|---|
| Instrumenting a framework with Comet (the common case) | `integrations/<category>/<framework>/<example-name>/` |
| Writing a how-to notebook for a Comet workflow | `guides/` |
| Building a custom Comet panel | `panels/` |

New examples go under `integrations/`. The top-level legacy dirs (`pytorch/`, `fastai/`,
`xgboost/`, `keras/`, `tensorflow/`) are historical — **do not add new examples there.** Example
folder names are **kebab-case** (`pytorch-mnist`, `fastai-hello-world`).

When adding an example, use the [`scaffold-example`](.claude/skills/scaffold-example/SKILL.md)
skill — it stamps [`templates/integration-example/`](templates/integration-example/) into the
target directory and renames it for you.

## Non-negotiable conventions

### Credentials — always from environment variables

```python
import os

COMET_API_KEY  = os.environ.get("COMET_API_KEY")
COMET_WORKSPACE = os.environ.get("COMET_WORKSPACE")
```

Never hardcode keys. Never read `.env` files in example code. In CI the workspace is
`cometexamples-tests` and `COMET_API_KEY` comes from a secret.

### The Comet SDK idiom

Match the house style used across the repo:

```python
import comet_ml

comet_ml.login(project_name="comet-example-<name>")
experiment = comet_ml.start()
# ... training / logging ...
experiment.end()
```

`comet_ml.login()` reads `COMET_API_KEY` from the environment. Project names follow
`comet-example-<framework>-<thing>`.

### Offline mode — the no-account path (recommended, not required)

Most examples log a real run, so they need an API key. Where it is practical, let a reader run
without an account using Comet's offline mode — no code change required:

```bash
COMET_MODE=offline python <example>.py
```

Note this in the README's run section when it works. Don't force it onto examples that genuinely
need a live run.

### Comments — only when the WHY is non-obvious

These are teaching examples: keep them short and readable. Don't add comments that restate what the
code does. Use a one-line `# WHY:` comment only when a behaviour would surprise a reader (a hidden
API constraint, a non-obvious ordering requirement, a known gotcha). No type-hint or docstring
mandate — match the surrounding file.

### Dependencies — a `requirements.txt` per example

Every example ships a `requirements.txt` **sibling to its entry point** (script or notebook
directory). This is the single source of truth and is exactly what CI installs
(`pip install -r requirements.txt`). Pin `comet_ml` like the rest of the repo (`comet_ml>=3.44.0`)
plus the framework deps.

Richer, multi-file projects **may** use Poetry with a committed `poetry.lock` (see
[`integrations/langgraph/`](integrations/langgraph/)). Do **not** introduce `uv`, and do **not**
gitignore `poetry.lock` — this repo commits it for reproducibility.

## Coding best practices

- **Principles:** DRY, KISS, YAGNI. Prefer reusing an existing helper over adding a new one.
- **Match the surrounding file's** style, naming, and comment density.
- **Git / PR safety:**
  - Never `git commit` or `git push` on `master`. Cut a feature branch (`<user>/<topic>`, e.g.
    `fschlz/pytorch-amp-example`), push there, open a PR, and let a human merge.
  - Commits follow **Conventional Commits**: `feat:`, `fix:`, `chore:`, `refactor:`, `docs:`.
  - Never `gh pr merge`, `gh pr close`, or `gh pr review --approve` — author/reviewer actions only.
    Blocked in [`.claude/settings.json`](.claude/settings.json).
  - No AI-attribution footers in commit messages or PR bodies.
  - **Update READMEs before opening a PR** — the example's own `README.md`, and the root
    [`readme.md`](readme.md) if you add a new top-level area.

## CI

Tests run via [`.github/workflows/test-examples.yml`](.github/workflows/test-examples.yml), which
holds **explicit matrices** of notebooks and scripts — examples are not auto-discovered. To get a
new example covered:

1. Make sure its `requirements.txt` is complete and the example runs from its own directory.
2. Add it to the matrix — the `notebooks` list (run with `ipython`) for a `.ipynb`, or the
   `example` list (run with `python <script> <arg>`) for a script.

An example without a matrix entry is valid but won't be tested.

## Every example must have a README.md

Follow the house structure used across the repo (see
[`templates/integration-example/README.md`](templates/integration-example/README.md)):

1. **Title + intro** — what the framework is and what instrumenting it with Comet gives you
2. **Documentation** — link to the relevant page on `comet.com/docs`
3. **See it** — link to a public Comet project, when one exists
4. **Setup** — `python -m pip install -r requirements.txt`
5. **Run the example** — the exact command (and the offline variant when it applies)

Keep the README in sync with the code and update it before every PR.

## Full contribution guide

See [CONTRIBUTING.md](CONTRIBUTING.md) for the complete standards, the recommended
plan → brainstorm → branch → implement → test → READMEs → PR → review workflow, and the PR checklist.
