# A Custom panel, and a report section: the model's output drift beside its
# volume, with the quiet days called out.
#
# Runs in a Comet Custom panel as-is, and as a `comet-mpm report --script`
# section. The two differ only in who answers the controls: a person in the
# panel, `script_params` in a report.
#
# Trace types are limited to what the panel's Plotly bundle registers --
# scatter, bar, box, violin, parcoords -- so this uses scatter and bar.

from comet_mpm import API

model = API().get_model()

numerical = model.get_numerical_features()
outputs = [name for name in numerical if name.startswith("prediction_")]

if not outputs:
    print("This panel needs an output column (prediction_*).")
    print("Numerical columns on this model:", numerical)
else:
    with st.expander("Controls", expanded=False):
        picking, drawing = st.columns(2)
        with picking:
            feature = st.selectbox("Output", outputs)
            algorithm = st.selectbox("Algorithm", ["EMD", "PSI", "KL", "KS"])
        with drawing:
            as_bars = st.checkbox("Draw as bars", False)
            mark_quiet = st.checkbox("Mark quiet days", True)

    drift = model.get_feature_drift(feature_name=feature, algorithm=algorithm)
    volume = model.get_nb_predictions()

    days = [str(index) for index in drift.index]
    values = [None if value != value else value for value in drift.iloc[:, 0]]

    trace = {
        "x": days,
        "y": values,
        "name": f"{feature} ({algorithm})",
        "type": "bar" if as_bars else "scatter",
    }
    if not as_bars:
        trace["mode"] = "lines"
        trace["line"] = {"width": 2, "color": "#2a78d6"}

    data = [trace]

    # A day with unusually few predictions makes drift jumpy for a reason that
    # has nothing to do with the model, so say which days those are rather than
    # letting the spike be read as drift.
    if mark_quiet and not volume.empty:
        counts = volume.iloc[:, 0].dropna()
        if len(counts):
            floor = counts.median() * 0.5
            quiet = [str(index) for index, count in counts.items() if count < floor]
            print(f"{len(quiet)} day(s) below half the median volume: {quiet[:5]}")

    layout = {
        "xaxis": {"type": "category", "title": {"text": "day"}},
        "yaxis": {"title": {"text": "distance"}},
        "margin": {"l": 56, "r": 24, "t": 16, "b": 56},
    }

    st.plotly_chart({"data": data, "layout": layout})
