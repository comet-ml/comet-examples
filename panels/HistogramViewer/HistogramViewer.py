%pip install streamlit_plotly_events

import comet_ml
from comet_ml.data_structure import Histogram
import streamlit as st
import numbers
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io
import base64
from streamlit_plotly_events import plotly_events

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

options = {
    "start": None,
    "stop": None,
    "ybins": None,
    "xbins": 50,
    ## Colors are scaled from highest to lowest. You can add
    ## additional values between 0 and 1 to add color ranges.
    "colorScale": [
        [0, "white"],  ## lower values
        [0.5, "gray"],  ## middle value
        [1, "blue"],  ## higher values
    ],
    "showScale": False,
    "layout": {
        "title": "",
        "xaxis": {"ticks": "", "side": "bottom", "title": "Steps"},
        "yaxis": {
            "ticks": "", 
            "ticksuffix": " ", 
            "title": "Weights"
        },
    },
}

@st.dialog("Histogram by Step")
def show_selected_data(selected_data, z, weight_labels, step_labels):
    y, x = selected_data[0]["pointNumber"]
    step = step_labels[x]
    data = [column[x] for column in z]
    fig = go.Figure(data=go.Bar(
        y=data,
        x=weight_labels,
    ))
    fig.update_layout(
        title='Step %s' % step,
        xaxis_title='Weight',
        yaxis_title='Count',
        barmode='group'
    )
    st.plotly_chart(fig)

def generate_column_plot(column, z):
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot([1, 2, 3, 2, 5, 2, 1])
    ax.axis('off')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    img_data = base64.b64encode(buf.getbuffer()).decode("utf8")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{img_data}">'

def transpose(matrix):
    return [list(row) for row in zip(*matrix)]

def my_range(start, stop, step):
    retval = []
    i = start
    while i <= stop:
        retval.append(i)
        i += step
    return retval

def getMinMax(histogram):
    min = None
    max = None
    # Find the first/min value:
    for i in range(len(histogram.counts)):
        if histogram.counts[i] > 0:
            min = histogram.values[i - 1]
            break

    # Find the last/max value:
    for i in range(len(histogram.counts) - 1, -1, -1):
        if histogram.counts[i] > 0:
            max = histogram.values[i + 1]
            break

    if min is None and max is None:
        min = -1.0
        max = 1.0
    #print(min, max)
    return [min, max]




def get_sample(length, max_steps):
    """
    Get selected/sampled indices
    """
    ## Start with all:
    selected = range(length)
    if length > max_steps:
        ## need to sample
        ## always include the first and last
        selected = [0] + random.sample(selected[1:-1], max_steps - 2) + [length - 1]
    return selected


def get_histogram_values(asset, start, stop, bins, maxSteps):
    ## First, collect them:
    histograms = []
    ## {'histograms': [{'step': num, 'histogram': {'index_values'}}, ...]
    index = 0
    selected_indices = get_sample(len(asset["histograms"]), maxSteps)
    xValues = []
    for hist in asset["histograms"]:
        if index in selected_indices:
            #print(hist)
            h = Histogram.from_json(hist["histogram"])
            xValues.append(hist["step"])
            histograms.append(h)
        index += 1
    ## Next, find the start/stop
    zValues = []
    #xValues = []
    if start is None or stop is None:
        minimum = None
        maximum = None
        for histogram in histograms:
            minmax = getMinMax(histogram)
            minimum = min(minimum if minimum is not None else float("+inf"), minmax[0])
            maximum = max(maximum if maximum is not None else float("-inf"), minmax[1])
        if start is None:
            start = minimum
        if stop is None:
            stop = maximum

    if bins == None:
        bins = 50

    for histogram in histograms:
        #print(start, stop)
        data = histogram.get_counts(start, stop, (stop - start) / bins)
        #print(data)
        zValues.append(data)
        #print(histogram)
        #xValues.append(histogram.step)

    yValues = my_range(start, stop, (stop - start) / bins)
    #print("zValues:", transpose(zValues))
    return [xValues, yValues, transpose(zValues)]


def drawHistogram(asset):
    [x, y, z] = get_histogram_values(
        asset, options["start"], options["stop"], options["ybins"], options["xbins"]
    )

    if len(z[0]) == 0:
        print("No histogram data available")
        return

    #sums = []
    #for step in range(len(z[0])):
    #    sum = 0.0
    #    for bin in range(len(z)):
    #        if isinstance(z[bin][step], numbers.Number):
    #            sum += z[bin][step]
    #    sums.append(sum)

    ## This is run on every cell:
    # hoverText = function(bin, step, item) {
    #  stepPercentage = (sums[step] === 0
    #    ? 0
    #    : (item / sums[step]) * 100
    #  ).toFixed(2)
    # return `Step: ${x[step]}<br>
    # Value: ${y[bin].toFixed(2)}</br>
    # Count: ${item.toFixed(2)}<br>
    # Count % of step: ${stepPercentage}%`
    # }

    #print(x)

    #column_plots = [generate_column_plot(column, z) for column in x]
    
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale=options["colorScale"],
        showscale=options["showScale"],
    ))

    fig.update_layout(**options["layout"])
    selected_data = plotly_events(fig)
    if selected_data:
        show_selected_data(selected_data, z, y, x)


api = comet_ml.API()

experiments = api.get_panel_experiments()

if len(experiments) == 0:
    print("No available experiments")
    st.stop()
elif len(experiments) == 1:
    experiment = experiments[0]
else:
    experiment = st.sidebar.selectbox(
        "Experiment:",
        experiments,
        format_func=lambda experiment: experiment.name or experiment.id,
    )

assets = experiment.get_asset_list("histogram_combined_3d")

if len(assets) == 0:
    print("No available histograms")
    st.stop()
elif len(assets) == 1:
    asset = assets[0]
else:
    asset = st.sidebar.selectbox(
        "Histogram:",
        sorted(assets, key=lambda item: item["fileName"]),
        format_func=lambda asset: asset["fileName"],
    )

histogram = experiment.get_asset(asset["assetId"], return_type="json")
drawHistogram(histogram)
