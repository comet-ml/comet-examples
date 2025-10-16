from comet_ml import API, ui
from comet_ml.data_structure import Histogram
import random
import numpy as np
import plotly.express as px

# Options:

max_xbins = st.sidebar.slider(
    "Maximum X bins",  # Label for the slider
    min_value=5,                # Minimum allowed value
    max_value=20,              # Maximum allowed value
    value=10,                   # Default starting value
    step=1                      # Increment step for the slider
)
max_ybins = st.sidebar.slider(
    "Maximum Y bins",  # Label for the slider
    min_value=5,                # Minimum allowed value
    max_value=100,              # Maximum allowed value
    value=50,                   # Default starting value
    step=1                      # Increment step for the slider
)
start = None
stop = None
#max_ybins = 50
#max_xbins = 50
# Colors are scaled from highest to lowest. You can add
# additional values between 0 and 1 to add color ranges.
colorScale = [
    [0, "white"], # lower values
    [0.5, "gray"], # middle value
    [1, "blue"] # higher values
]
showScale = False
layout = {
    "title": "Histograms by Step",
    "xaxis": {
        "ticks": "",
        "side": "bottom",
        "title": "Steps"
    },
    "yaxis": {
        "ticks": "",
        "ticksuffix": " ",
        "autosize": True,
        "title": "Weights"
    }
}

def collect(histogram, start=None, stop=None, bins=50):
    """
    Collect the counts for the given range into bins.

    Args:
        start: optional, float, start of range to display
        stop: optional, float, end of range to display
        bins: optional, int, number of bins

    Returns a list of dicts containing details on each
    virtual bin.
    """
    counts_compressed = histogram.counts_compressed()
    if start is None:
        if len(counts_compressed) > 0:
            start = histogram.values[counts_compressed[0][0]]
        else:
            start = -1.0
    if stop is None:
        if len(counts_compressed) > 1:
            stop = histogram.values[counts_compressed[-1][0]]
        else:
            stop = 1.0

    step = (stop - start) / bins

    counts = histogram.get_counts(start, stop + step, step)
    current = start
    bins = []
    next_one = current + step
    i = 0
    while next_one <= stop + step and i < len(counts):
        start_bin = histogram.get_bin_index(current)
        stop_bin = histogram.get_bin_index(next_one)
        current_bin = {
            "value_start": current,
            "value_stop": next_one,
            "bin_index_start": start_bin,
            "bin_index_stop": stop_bin,
            "count": counts[i],
        }
        bins.append(current_bin)
        current = next_one
        next_one = current + step
        i += 1
    return bins

def get_histogram_indices(length, max_xbins):
    """
    Get indices from list of histograms, sampling if necessary.
    """
    if (length > max_xbins):
        return (
            [0] +
            random.sample(
                list(range(1, length - 1)),
                max_xbins - 2) +
            [length - 1])
    else:
        return list(range(length))

def get_histogram_data(
    experiment,
    asset,
):
    assetJSON = experiment.get_asset(asset["assetId"], return_type="json")
    histograms = []
    # {"histograms": [{"step": num, "histogram": {"index_values"}}, ...]
    selected = get_histogram_indices(len(assetJSON["histograms"]), max_xbins)
    for index in selected:
        hist = assetJSON["histograms"][index]
        histogram = Histogram.from_json(hist["histogram"])
        histogram.logged_at_step = hist["step"]
        histograms.append(histogram)

    # First, find the overall min/max of all histograms:
    min_val, max_val = float("+inf"), float("-inf")
    for histogram in histograms:
        # {"value_start", "value_stop", "bin_index_start", "bin_index_stop", "count"}
        data = collect(histogram, start=None, stop=None, bins=max_ybins)
        min_val = min(min_val, data[0]["value_start"])
        max_val = max(max_val, data[-1]["value_stop"])
        
    zValues = []
    yValues = []
    span = (max_val - min_val)/(max_ybins)
    xValues = np.arange(min_val, max_val + span, span)
    for h, histogram in enumerate(histograms):
        # {"value_start", "value_stop", "bin_index_start", "bin_index_stop", "count"}
        data = collect(histogram, start=min_val, stop=max_val, bins=max_ybins)
        zValues.append([bin["count"] for bin in data])
        yValues.append(histogram.logged_at_step)
    return [xValues, yValues, np.transpose(zValues), min_val, max_val]

def plot_histogram(experiment, asset):
    x, y, data, xmin, xmax = get_histogram_data(
        experiment,
        asset,
    )
    if (len(data[0]) == 0):
        print("<h1>No histogram data available<h1>")
        return None

    # Transposed, so x is y, y is x:
    fig = px.imshow(
        data,
        x=y,
        y=x[:len(data)],
        aspect="auto",
        color_continuous_scale=colorScale,
    )
    st.plotly_chart(fig)


api = API()
experiments = api.get_panel_experiments()
if len(experiments) == 1:
    selected_experiment = experiments[0]
else:
    selected_experiment = ui.dropdown("Experiments: ", experiments)

if selected_experiment:
    assets = sorted(
        selected_experiment.get_asset_list('histogram_combined_3d'),
        key=lambda item: item["fileName"]
    )
    selected_histogram = ui.dropdown(
        "Histogram: ",
        assets,
        format_func=lambda item: item["fileName"]
    )
    if selected_histogram:
        plot_histogram(selected_experiment, selected_histogram)
    else:
        print("No histograms available.")    
else:
    print("No experiments available.")
