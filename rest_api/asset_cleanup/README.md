### Comet Asset Cleanup Script

A safe, automatable Python utility to reduce cloud storage by identifying and deleting old experiment assets in Comet.

This script can:
- Iterate experiments in a workspace (single project or all projects)
- List assets per experiment and evaluate age
- Save a JSON "deletion plan" before any destructive action
- Delete assets older than a threshold (opt-in, with confirmation)

By default, the script runs in dry-run mode and never deletes anything unless you pass `--execute`.

---

### Requirements
- Python 3.9+
- Poetry (recommended)
- Comet API key with access to the target workspace/projects

Environment variables (recommended via `.env`):
- `COMET_API_KEY` (required)
- `COMET_WORKSPACE` (optional if passed via `--workspace`)
- `COMET_PROJECT` (optional; omit to process all projects)

---

### Setup
1) Install dependencies (project uses Poetry):

```bash
poetry install --no-root
```

2) Create a `.env` in the repo root (or export env vars in your shell):

```bash
echo "COMET_API_KEY=your_api_key_here" >> .env
echo "COMET_WORKSPACE=your_workspace_name" >> .env
# COMET_PROJECT is optional; omit to process all projects
```

---

### Usage
General form:

```bash
poetry run python scripts/comet_asset_cleanup.py \
  --workspace <WORKSPACE> \
  [--project <PROJECT>] \
  [--days-threshold 365] \
  [--include-archived] \
  [--execute] [--yes] \
  [--batch-size 10] [--delay 1.0]
```

Flags:
- `--workspace`: Workspace to process (required if not set in env)
- `--project`: Project to process; omit to process ALL projects in the workspace
- `--days-threshold`: Assets older than this many days are targeted (default: 365)
- `--include-archived`: Include archived experiments
- `--dry-run`: Force simulation mode (default behavior)
- `--execute`: Perform deletions (requires confirmation unless `--yes`)
- `--yes`: Skip interactive confirmation (non-interactive/scheduled runs)
- `--batch-size`: Number of deletions per batch (default: 10)
- `--delay`: Seconds to sleep between batches (default: 1.0)

Examples:

Dry-run, one project:
```bash
poetry run python comet_asset_cleanup.py \
  --workspace your-ws --project your-proj --days-threshold 365
```

Dry-run, all projects in a workspace:
```bash
poetry run python comet_asset_cleanup.py \
  --workspace your-ws --days-threshold 365
```

Execute, include archived, non-interactive:
```bash
poetry run python comet_asset_cleanup.py \
  --workspace your-ws --project your-proj --days-threshold 365 \
  --include-archived --execute --yes
```

---

### What gets saved
- A deletion plan JSON is always written before any deletion (also in dry runs):

  - File name pattern: `asset_deletion_plan_<workspace>_<project>_<timestamp>.json`
  - Structure:

```json
{
  "my-workspace": {
    "my-project": {
      "abcd1234_experimentkey": [
        "gradient_layer3.4_weight.json",
        "..."
      ]
    }
  }
}
```

- Logs are written to `comet_asset_cleanup.log` and printed to console.

---

### How it works (Comet REST API)
The script uses Comet REST endpoints to discover projects, experiments, and assets, and to delete assets:
- Get Projects (used when `--project` is omitted):
  - [Get Projects](https://www.comet.com/docs/v2/api-and-sdk/rest-api/read-endpoints/#get-projects)
- Get Experiments in a project:
  - [Get Experiments](https://www.comet.com/docs/v2/api-and-sdk/rest-api/read-endpoints/#get-experiments)
- List assets for an experiment:
  - [Get Asset List](https://www.comet.com/docs/v2/api-and-sdk/rest-api/read-endpoints/#get-asset-list)
- Delete an asset:
  - [Delete an Asset](https://www.comet.com/docs/v2/api-and-sdk/rest-api/write-endpoints/#delete-an-asset)

The script computes the threshold date/time and selects assets where `createdAt` is older than the given threshold.

---

### Safety and scheduling
- Default behavior is non-destructive dry run.
- In execute mode, the script saves a deletion plan JSON and requires a console confirmation (`DELETE`) unless `--yes`.
- Consider scheduling as a CRON job:

```cron
# Run daily at 2:00am in execute mode across all projects
0 2 * * * cd /path/to/repo && \
  /usr/local/bin/poetry run python scripts/comet_asset_cleanup.py \
  --workspace your-ws --days-threshold 365 --execute --yes >> cleanup.cron.log 2>&1
```

---

### Notes
- API rate limits: adjust `--batch-size` and `--delay` if needed.
- Authentication: the script expects `Authorization` header with your API key (set via `COMET_API_KEY`).
- Project discovery requires `Get Projects` permissions in the workspace.
- Test with a small `--days-threshold` and in dry run before enabling execute mode.

---

### Support
If you need help adapting the script to your environment or adding filters (e.g., by asset type), reach out to your Comet contact.
