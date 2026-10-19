#!/usr/bin/env bash
#
# Everything `comet-mpm report` can do, as eight reports you can open.
#
#   ./showcase.sh WORKSPACE MODEL [OUTPUT_DIR] [DAYS] [FORMAT]
#
# Needs COMET_API_KEY, plus COMET_URL_OVERRIDE and COMET_MPM_API_BASE_PATH for
# a self-hosted install reached directly. See README.md.
#
# Each step prints the command before running it, so the output doubles as a
# transcript of how each report was made.
#
# DAYS defaults to 180 -- twice what the dashboard's date picker offers, which
# is the point of the whole exercise. Raise it to 360 for a full year; a range
# longer than the model has history is not an error, it just draws an empty
# stretch before the data starts.

set -u

WORKSPACE="${1:-}"
MODEL="${2:-}"
OUT="${3:-./showcase-out}"
DAYS="${4:-180}"
# pdf (the default) or html. HTML keeps the charts interactive and needs no
# browser; PDF is the one to attach to a review.
FORMAT="${5:-pdf}"

if [ -z "$WORKSPACE" ] || [ -z "$MODEL" ]; then
    echo "usage: $0 WORKSPACE MODEL [OUTPUT_DIR] [DAYS] [FORMAT]" >&2
    echo "  DAYS defaults to 180; FORMAT is pdf (default) or html" >&2
    exit 2
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT"

# The Pyodide Custom panel examples, if the frontend repo is checked out beside
# this one. They are not vendored here: they belong to the app, and running
# them from their own copy is the point -- a panel that works in the product
# works in a report unchanged.
PYODIDE="$HERE/../../../frontend/src/workers/pyodidePanels/examples"

case "$FORMAT" in
    pdf|html) ;;
    *) echo "FORMAT must be pdf or html, got '$FORMAT'" >&2; exit 2 ;;
esac

# PDF needs a browser. Check once, before any fetching, so a missing one costs
# a second rather than eight reports' worth of queries.
if [ "$FORMAT" = "pdf" ] && ! python -c "
import sys
from comet_mpm.report import pdf
problem = pdf.is_available()
if problem:
    sys.stderr.write(problem + '\n')
    sys.exit(1)
" 2>&1; then
    echo "Cannot export PDF; see the message above." >&2
    exit 2
fi

failures=0

step() {
    echo
    echo "=============================================================="
    echo "$1"
    echo "=============================================================="
    shift
    echo "\$ $*"
    echo
    if ! "$@"; then
        echo "  ^ FAILED" >&2
        failures=$((failures + 1))
    fi
}

step "1. ${DAYS} days -- the dashboard's picker stops at 90. Volume plus drift,
   percentiles and category mix for the first few features, output columns
   first. Every step exports PDF: portrait, cover page, two charts to a page,
   figures under each." \
    comet-mpm report -w "$WORKSPACE" -m "$MODEL" --days "$DAYS" \
        --out "$OUT/01-default.$FORMAT"

step "2. Monthly buckets, dark. Only additive metrics honour the interval;
   drift and percentiles say on the chart why they cannot." \
    comet-mpm report -w "$WORKSPACE" -m "$MODEL" \
        --days "$DAYS" --interval MS --theme dark \
        --out "$OUT/02-annual-dark.$FORMAT"

step "3. Chosen features, including the model's own output, with distribution
   charts and the native daily interval -- ${DAYS} daily points." \
    comet-mpm report -w "$WORKSPACE" -m "$MODEL" \
        --days "$DAYS" --interval daily --density \
        --feature prediction_probability \
        --out "$OUT/03-daily-density.$FORMAT"

step "4. A custom metric. Its meaning is opaque to the report, so it is never
   aggregated automatically." \
    comet-mpm report -w "$WORKSPACE" -m "$MODEL" \
        --days "$DAYS" \
        --metric-sql "SELECT count(*) FROM MODEL" \
        --out "$OUT/04-custom-metric.$FORMAT"

step "5. Filtered and pinned to one model version." \
    comet-mpm report -w "$WORKSPACE" -m "$MODEL" \
        --days "$DAYS" --version 1.0.0 \
        --out "$OUT/05-filtered.$FORMAT"

step "6. The annual review shape: monthly buckets, two named features." \
    comet-mpm report -w "$WORKSPACE" -m "$MODEL" \
        --days "$DAYS" --interval MS \
        --feature prediction_probability --feature feature_latency_ms \
        --out "$OUT/06-annual-review.$FORMAT"

step "7. Panels written here, run as report sections. One draws two charts,
   one draws none and reports in words instead." \
    comet-mpm report -w "$WORKSPACE" -m "$MODEL" --days "$DAYS" \
        --script "$HERE/panels/health_check.py" \
        --script "$HERE/panels/volume_and_drift.py" \
        --script "$HERE/panels/category_shift.py" \
        --script "$HERE/panels/output_drift.py" \
        --script-params "$HERE/script_params.json" \
        --out "$OUT/07-panels.$FORMAT"

if [ -d "$PYODIDE" ]; then
    step "8. The product's own Custom panel examples, run unmodified from the
   frontend checkout. bar_by_decile is omitted: it needs get_label_names,
   which is not on main yet." \
        comet-mpm report -w "$WORKSPACE" -m "$MODEL" --days "$DAYS" \
            --script "$PYODIDE/chart_controls.py" \
            --script "$PYODIDE/multi_metric.py" \
            --script "$PYODIDE/threshold_overlay.py" \
            --script "$PYODIDE/time_buckets.py" \
            --script "$PYODIDE/custom_metric_panel.py" \
            --script "$PYODIDE/confusion_matrix.py" \
            --script "$PYODIDE/drift_demo.py" \
            --script-params "$HERE/script_params.json" \
            --out "$OUT/08-pyodide-panels.$FORMAT"
else
    echo
    echo "8. Skipped: the Custom panel examples were not found at"
    echo "   $PYODIDE"
    echo "   Check out the frontend repo beside this one to include them."
fi

step "9. Write the spec rather than the report, so it can be edited and re-run." \
    comet-mpm report -w "$WORKSPACE" -m "$MODEL" --days "$DAYS" --interval MS \
        --feature prediction_probability --density \
        --print-spec

echo
echo "=============================================================="
if [ "$failures" -eq 0 ]; then
    echo "All steps completed. Reports are in $OUT"
else
    echo "$failures step(s) failed. Reports that did build are in $OUT"
fi
ls -1sh "$OUT" 2>/dev/null
exit "$failures"
