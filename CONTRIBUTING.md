# Contributing to comet-examples

Thank you for contributing. This repo is a reference library for Comet users — the goal is examples
that are easy to find, easy to run, and easy to adapt. Contributions come from both humans and AI
agents; the same standards apply to both. Agents should also read [AGENTS.md](AGENTS.md).

## Recommended workflow

This is the loop we follow for non-trivial contributions. The slash-commands in brackets come from
Claude Code plugins (see below) and are optional but recommended.

1. **Plan first.** Switch Claude Code to plan mode on the best available model with reasoning effort
   maxed before writing any code.
2. **Brainstorm the scope** (`/brainstorming`) — agree on *what* to build before *how*.
3. **Write the plan** (`/writing-plans`) — turn the agreed scope into an implementation plan.
4. **Cut a feature branch** — `git switch -c <user>/<topic>` (never commit on `master`).
5. **Implement and commit frequently** — small [Conventional Commits](https://www.conventionalcommits.org/)
   (`feat:`, `fix:`, `chore:`, `refactor:`, `docs:`).
6. **Test and fix** — run the example from its own directory until it works.
7. **Update the READMEs** — the example's own README, and the root `readme.md` if you added a new area.
8. **Open a PR with a description** — what changed and why. A human merges it.
9. **Review the diff** (`/review`) before requesting human review.

**Recommended Claude Code plugins:** `superpowers` (provides `/brainstorming`, `/writing-plans`, and
`/review`) and `caveman` (terse output mode).

## Repo structure

```
comet-examples/
├── integrations/   # Add Comet to a framework, grouped by ML task
├── guides/         # How-to notebooks for Comet workflows
├── panels/         # Custom Comet panel examples
├── notebooks/      # General/standalone notebooks
└── templates/      # Starter template (integration-example)
```

New examples go under `integrations/<category>/<framework>/<example-name>/`. Categories in use:
`model-training`, `model-evaluation`, `model-optimization`, `model-deployment`,
`workflow-orchestration`, `reinforcement-learning`, `llm`, `data-management`.

**Which bucket does my example belong in?**

| Question | Bucket |
|---|---|
| I'm instrumenting a framework with Comet (the common case) | `integrations/<category>/<framework>/` |
| I'm showing how to do something with Comet itself in a notebook | `guides/` |
| I'm building a custom Comet panel | `panels/` |

The top-level dirs `pytorch/`, `fastai/`, `keras/`, `tensorflow/`, `xgboost/` are legacy — don't add
new examples there. If you're unsure, open an issue and ask.

## Adding an example

[`templates/integration-example/`](templates/integration-example/) is a minimal, runnable Comet
example: a single `comet_ml` script, a `requirements.txt`, and a README in the house structure. New
examples start from it.

**Easiest path (recommended):** use the `scaffold-example` skill — it copies the template and
renames it for you:

```bash
python .claude/skills/scaffold-example/scripts/scaffold.py my-framework-hello-world \
  --description "What it does" \
  --dest integrations/model-training/my-framework
```

**By hand:** copy the template and rename its `example_integration` / `example-integration`
identifiers across the files:

```bash
cp -r templates/integration-example integrations/<category>/<framework>/my-example
```

Then, either way:

1. `cd` into the new folder and `python -m pip install -r requirements.txt`.
2. Fill in the script (real logic) and the README sections.
3. Run it from the folder: `python <name>.py` (and check the offline variant if it applies).
4. If it should be tested in CI, add it to `.github/workflows/test-examples.yml` (see [CI](#ci)).
5. Open a PR and fill in the checklist below.

## Standards

### Every example must have

- A `README.md` in the house structure (below)
- A complete `requirements.txt` sibling to its entry point
- Credentials loaded from environment variables only — no hardcoded keys

### README structure

Follow the structure used across the repo (see the template's README):

- **Title + intro** — what the framework is and what Comet adds
- **Documentation** — link to the relevant `comet.com/docs` page
- **See it** — link to a public Comet project, when one exists
- **Setup** — `python -m pip install -r requirements.txt`
- **Run the example** — the exact command (and the offline variant when it applies)

### Credentials

Always load from environment variables:

```python
import os

api_key   = os.environ["COMET_API_KEY"]
workspace = os.environ["COMET_WORKSPACE"]
```

`comet_ml.login()` picks up `COMET_API_KEY` automatically. Never commit `.env` files or hardcoded
keys.

### Offline mode (optional)

Where it's practical, let a reader run without an account using Comet's offline mode and note it in
the README:

```bash
COMET_MODE=offline python <example>.py
```

Don't force it onto examples that genuinely need a live run.

### Dependencies

Each example declares its dependencies in a `requirements.txt` sibling to its entry point — this is
the single source of truth and what CI installs. Pin `comet_ml>=3.44.0` plus the framework deps.
Richer multi-file projects may use Poetry with a committed `poetry.lock` (e.g.
`integrations/langgraph/`). Don't introduce `uv`.

### Code style

- These are teaching examples — keep them short and readable
- No comments that explain what the code does — well-named variables and functions do that
- A short `# WHY:` comment is appropriate when a behaviour would surprise a reader
- No emojis

## CI

[`.github/workflows/test-examples.yml`](.github/workflows/test-examples.yml) tests an **explicit
list** of notebooks and scripts — examples are not auto-discovered. To get yours covered, add it to
the matrix:

- **Notebook** → add its path to the `notebooks` list (run with `ipython`).
- **Script** → add `{script: "<path>", arg: "<args>"}` to the `example` list (run with
  `python <script> <args>`).

Make sure the example installs cleanly from its own `requirements.txt` first.

## PR checklist

Before opening a PR, verify:

- [ ] Example is in the right place (`integrations/<category>/<framework>/`, or `guides/` / `panels/`)
- [ ] Folder name is kebab-case (e.g. `my-framework-hello-world`)
- [ ] `README.md` has the house sections
- [ ] READMEs updated — the example's `README.md`, and the root `readme.md` if you added a new area
- [ ] `requirements.txt` is complete and the example runs from its own directory
- [ ] Added to the CI matrix in `test-examples.yml` if it should be tested
- [ ] No credentials or `.env` files committed

## Questions

Open an issue or start a discussion. We're happy to help you figure out the right bucket or approach
before you write the code.
