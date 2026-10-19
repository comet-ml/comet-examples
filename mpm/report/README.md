# Long-range reports

The dashboard's date picker offers 7, 30 and 90 days. A longer range is
proportionally more expensive to query, so it belongs somewhere run
deliberately rather than on every dashboard load. `comet-mpm report` builds one.

Everything here is runnable. Replace the workspace and model and go.

## Everything at once

```bash
./showcase.sh WORKSPACE MODEL [OUTPUT_DIR] [DAYS] [FORMAT]
```

Builds eight reports covering every capability, over **180 days by default —
twice what the dashboard's date picker offers**, which is the point of the
exercise. Pass `360` for a full year; a range longer than the model has
history is not an error, it just draws an empty stretch before the data
starts. `FORMAT` is `pdf` by default; pass `html` for interactive charts and
no browser requirement.

Each command is printed before it runs, so the output doubles as a transcript. Step 8 runs the product's own
Custom panel examples from a `frontend` checkout beside this repo, and is
skipped if it is not there.

PDF needs the `report-pdf` extra, which the script checks for before fetching
anything:

```bash
pip install -e ".[report-pdf]" && playwright install chromium
```

## Setup

```bash
pip install -e ".[report]"          # charts + HTML
pip install -e ".[report-pdf]"      # and PDF
playwright install chromium         # once, for PDF only

export COMET_API_KEY=...
```

Self-hosted installs reached directly also need:

```bash
export COMET_URL_OVERRIDE=http://127.0.0.1:8080/
export COMET_MPM_API_BASE_PATH=mpm/    # the query API is at /mpm/, not /api/mpm/
```

## The shortest useful command

```bash
comet-mpm report -w WORKSPACE -m MODEL --out report.html
```

200 days, weekly, with volume plus drift, percentiles and category mix for the
first few features. Output features (`prediction_*`) come first, since drift on
what a model predicts is usually the question before drift on what it was fed.

## Choosing what goes in

```bash
# specific features, including the model's own output
comet-mpm report -w WORKSPACE -m MODEL \
    --feature prediction_probability --feature feature_age \
    --out report.html

# a year, monthly, dark, with distribution charts
comet-mpm report -w WORKSPACE -m MODEL \
    --days 365 --interval MS --theme dark --density \
    --out report.html

# keep the native daily buckets
comet-mpm report -w WORKSPACE -m MODEL --interval daily --out report.html

# a custom metric alongside the standard sections
comet-mpm report -w WORKSPACE -m MODEL \
    --metric-sql "SELECT AVG(prediction_probability) FROM MODEL" \
    --out report.html
```

`--interval` only applies to metrics that can be aggregated. Drift, percentiles
and distributions are computed from a distribution that no longer exists once
it has been reduced to a number, so they stay at their native daily interval
and say so on the chart rather than being quietly resampled.

## PDF

```bash
comet-mpm report -w WORKSPACE -m MODEL --out report.pdf
```

The format is taken from the extension, or set `--format pdf`. The PDF is the
HTML printed by a headless browser, so the two cannot drift apart. Print CSS
keeps a chart from being split across a page break.

## Specs

Pin a report down once and re-run it:

```bash
comet-mpm report -w WORKSPACE -m MODEL --days 365 --print-spec > myspec.json
comet-mpm report --spec myspec.json --out report.html
```

`specs/weekly_overview.json` and `specs/with_script.json` are starting points —
replace `REPLACE_ME` in each.

```bash
comet-mpm report --spec specs/weekly_overview.json --out report.html
```

A spec stores `days`, not a date range, so re-running it next month reports on
that month's last N days. For a frozen window add an explicit `"end"`:

```json
{"days": 90, "end": "2026-06-01T00:00:00+00:00"}
```

## Panel scripts as sections

A Custom panel script and a report want the same chart from the same data, so a
report can run one directly:

```bash
comet-mpm report -w WORKSPACE -m MODEL --days 60 \
    --script panels/output_drift.py \
    --script-params script_params.json \
    --out report.html
```

### The panels here

| Panel | What it shows |
| --- | --- |
| `output_drift.py` | Drift on the model's own output, with four controls |
| `volume_and_drift.py` | Two charts from one panel: traffic beside the drift it explains |
| `category_shift.py` | Grouped bars over categories, first bucket against last |
| `health_check.py` | No chart at all: what the model looks like, in words |

`volume_and_drift.py` shows that a section can hold several charts — every
`st.plotly_chart` call adds one. `health_check.py` shows that a section need not
draw: a panel that inspects the model and reports in words is a legitimate
section, and it is what the others fall back to when a model has nothing they
can plot.

### The product's own panels

The Custom panel examples in the app's frontend run here unmodified. With that
repo checked out beside this one:

```bash
PY=../../../frontend/src/workers/pyodidePanels/examples
comet-mpm report -w WORKSPACE -m MODEL --days 60 \
    --script $PY/drift_demo.py --script $PY/threshold_overlay.py \
    --script $PY/time_buckets.py --script $PY/multi_metric.py \
    --script-params script_params.json --out report.html
```

Seven of the eight run as they are. `bar_by_decile.py` does not: it needs
`get_label_names`, which is not on `main` yet.

`panels/output_drift.py` runs unchanged in a Comet Custom panel *and* here. The
difference is who answers the controls: a person in the panel, `script_params`
in a report, keyed by the script's filename — the same shape
`comet_config_override` uses to configure a panel:

```json
{"output_drift.py": {"Output": "prediction_probability", "Algorithm": "PSI"}}
```

Any widget the script offers can be set by its label. Unset ones fall back to
their own defaults, so a script runs with no params at all and produces what
the panel would show before anyone touched a control. The values used are
printed above the chart, because a report has no dropdowns to show.

The report captures `st.plotly_chart` for charts and `print` for narration.
`st.columns` and `st.expander` are accepted and ignored; anything else
unimplemented is a no-op rather than an error.

A script is ordinary Python run in this process, with no sandbox. A spec that
names one is as trusted as the script it names.

## What is not here yet

- **Classification curves and gains tables.** They need `get_classification_curve`
  and `get_gains_table`, which are not on `main` yet.
- **Coarser intervals for drift and percentiles.** The request-facing interval is
  DAILY or HOURLY, and the responses carry no window-level aggregate, so there is
  nothing correct to roll them up to. Reported at their native interval until the
  backend exposes the MONTHLY/YEARLY granularities it already defines internally.
- **Alerts.** `/mpm/v2/monitor-notification/count` is not in the SDK yet.

## Cost, briefly

`--chunk-days` splits the range into one request per window. Smaller windows
are **not cheaper** — the backend scans the same rows either way — they avoid a
single request large enough to time out, and let you watch progress. Query cost
is linear in rows scanned, so a 180-day report costs about twice a 90-day one.

Only the volume and custom-metric sections are split. Drift cannot be: the
backend draws its numerical bins from the whole requested range, so each window
would be binned differently and the series would not join up. A density is one
distribution over the range, so it cannot be split either. Percentiles and
categories are computed per bucket and could be, but are one request each for
now.
