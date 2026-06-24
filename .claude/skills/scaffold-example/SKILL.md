---
name: scaffold-example
description: Scaffold a brand-new Comet example in this repo from the canonical template under templates/integration-example/. Use whenever the user wants to create, add, or start a new example, demo, or integration on Comet — phrasings like "add a Comet example for X", "scaffold an integration example", "create a new example for <framework>", "start a hello-world for <library>", "set up a new example project". Do NOT hand-build the folder structure or copy files manually — this skill stamps out the example and renames it correctly. Require a name and a one-line description of what it does; refuse to invent an example from nothing.
---

# Scaffold a new Comet example

This repo has one canonical example shape, a real, runnable template under
[`templates/integration-example/`](../../../templates/integration-example): a single `comet_ml`
script (`example_integration.py`), a `requirements.txt`, and a README in the house structure
(Title → Documentation → See it → Setup → Run the example). Getting it right by hand is fiddly and
easy to drift on. This skill copies the template and renames it, so you start from something that
already runs.

## When to refuse

You need two things:

- A **name** (kebab-case, e.g. `pytorch-amp-example`, `keras-mnist-dnn`). It becomes the folder, the
  Python module, and the Comet project name (`comet-example-<name>`).
- A **one-line description** of what it does.

If either is missing, ask one short question. Don't invent an example — an empty scaffold with a
guessed purpose wastes the user's time.

## Step 1 — Generate the example

Run the bundled generator. It copies the template, renames its sentinel identity
(`example_integration` / `example-integration`) to the new name, and refuses to overwrite an
existing folder.

```bash
python .claude/skills/scaffold-example/scripts/scaffold.py <name> \
  --description "<one line>" \
  --dest <parent-dir>
```

`--dest` is the parent directory (default `integrations`). New examples live under
`integrations/<category>/<framework>/`, so pass e.g.
`--dest integrations/model-training/pytorch`. Categories in use: `model-training`,
`model-evaluation`, `model-optimization`, `model-deployment`, `workflow-orchestration`,
`reinforcement-learning`, `llm`, `data-management`. The script prints the folder / script / project
names it chose and a next-steps checklist.

The name is normalised: the folder and Comet project use kebab-case, the Python file uses
snake_case (so it stays importable). The description is printed for you to paste into the README
intro — there's no structured field for it.

## Step 2 — Make it real

The scaffold runs as-is but is generic. Fill the parts marked `TODO`:

- **`<name>.py`** — replace the stub loop with the real instrumentation: your framework's training
  or inference, plus the `comet_ml` logging that matters (`log_parameters`, `log_metrics`,
  `log_model`, …). Keep the `comet_ml.login(...)` + `comet_ml.start()` + `experiment.end()` frame.
- **`requirements.txt`** — add the framework dependency alongside the pinned `comet_ml`.
- **`README.md`** — fill the title, intro (paste the description), the docs link, and a public
  Comet project link if one exists. This is required for every example (see
  [AGENTS.md](../../../AGENTS.md)).

Keep the repo conventions: credentials from env vars only, `# WHY:`-only comments, no emojis.

## Step 3 — Verify

From the new example directory:

```bash
python -m pip install -r requirements.txt
python <name>.py                       # logs a real run (needs COMET_API_KEY)
COMET_MODE=offline python <name>.py    # runs without an account, if practical
```

To get the example tested, add it to the matrix in
[`.github/workflows/test-examples.yml`](../../../.github/workflows/test-examples.yml) — the
`notebooks` list for a `.ipynb`, or the `example` list for a script. See
[CONTRIBUTING.md](../../../CONTRIBUTING.md#ci).

## What this skill does NOT do

- **No git commit / PR.** Leave the new files in the working tree for the user to review.
- **No domain logic.** It scaffolds; you (with the user) write the actual instrumentation, deps, and README.
