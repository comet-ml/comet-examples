# Volume and output drift together, as two charts from one panel.
#
# A section can hold more than one chart: every st.plotly_chart call becomes
# another figure in the same section. Useful when two views only make sense
# side by side -- drift is hard to read without knowing whether the traffic
# behind it was steady.

from comet_mpm import API

model = API().get_model()

outputs = [f for f in model.get_numerical_features() if f.startswith("prediction_")]

if not outputs:
    print("No output columns (prediction_*) on this model.")
else:
    feature = st.selectbox("Output", outputs)

    volume = model.get_nb_predictions()
    drift = model.get_feature_drift(feature_name=feature)

    days = [str(index) for index in volume.index]
    counts = [None if v != v else v for v in volume.iloc[:, 0]]

    st.plotly_chart(
        {
            "data": [
                {
                    "x": days,
                    "y": counts,
                    "type": "bar",
                    "name": "predictions",
                    "marker": {"color": "#2a78d6"},
                }
            ],
            "layout": {
                "title": {"text": "Predictions per day"},
                "xaxis": {"type": "category", "showticklabels": False},
                "yaxis": {"title": {"text": "predictions"}},
                "margin": {"l": 56, "r": 24, "t": 40, "b": 24},
                "height": 260,
            },
        }
    )

    drift_days = [str(index) for index in drift.index]
    distances = [None if v != v else v for v in drift.iloc[:, 0]]

    st.plotly_chart(
        {
            "data": [
                {
                    "x": drift_days,
                    "y": distances,
                    "type": "scatter",
                    "mode": "lines",
                    "name": feature,
                    "line": {"width": 2, "color": "#eb6834"},
                }
            ],
            "layout": {
                "title": {"text": f"Drift: {feature}"},
                "xaxis": {"type": "category", "showticklabels": False},
                "yaxis": {"title": {"text": "distance"}},
                "margin": {"l": 56, "r": 24, "t": 40, "b": 24},
                "height": 260,
            },
        }
    )

    usable = [v for v in counts if v is not None]
    if usable:
        steady = max(usable) <= min(usable) * 1.5
        print(
            f"Traffic was {'steady' if steady else 'uneven'} over the period "
            f"({min(usable):,.0f} to {max(usable):,.0f} per day), so the drift "
            f"above is {'comparable across days' if steady else 'noisier on the quiet days'}."
        )
