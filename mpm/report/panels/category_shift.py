# How a categorical feature's mix changed, as grouped bars: first bucket beside
# last bucket, one pair per category.
#
# A bar chart over categories rather than time, so the x axis must be declared
# as a category axis -- the runtime's default is a date axis and categories
# plot nowhere on it.

from comet_mpm import API

model = API().get_model()

categorical = model.get_categorical_features()

if not categorical:
    print("This panel needs a categorical feature.")
    print("Numerical columns on this model:", model.get_numerical_features())
else:
    feature = st.selectbox("Feature", categorical)
    sort_by_shift = st.checkbox("Order by size of change", True)

    distribution = model.get_feature_category_distribution(feature_name=feature)

    # The endpoint answers with a row per (timestamp, category), so gather the
    # first and last bucket rather than assuming a wide frame. A category absent
    # from a bucket had no share of it: one that disappeared ends at zero, one
    # that appeared starts there.
    label_column = distribution.columns[0]
    value_column = distribution.columns[-1]
    buckets = {}
    for timestamp, row in distribution.iterrows():
        share = row[value_column]
        if share != share:
            continue
        buckets.setdefault(timestamp, {})[str(row[label_column])] = share
    ordered = [buckets[timestamp] for timestamp in sorted(buckets)]
    first_seen = ordered[0] if ordered else {}
    last_seen = ordered[-1] if ordered else {}

    categories = sorted(set(first_seen) | set(last_seen))
    if sort_by_shift:
        categories.sort(
            key=lambda name: abs(last_seen.get(name, 0) - first_seen.get(name, 0)),
            reverse=True,
        )

    st.plotly_chart(
        {
            "data": [
                {
                    "x": categories,
                    "y": [first_seen.get(name, 0) for name in categories],
                    "type": "bar",
                    "name": "start of period",
                    "marker": {"color": "#9ec5f4"},
                },
                {
                    "x": categories,
                    "y": [last_seen.get(name, 0) for name in categories],
                    "type": "bar",
                    "name": "end of period",
                    "marker": {"color": "#2a78d6"},
                },
            ],
            "layout": {
                "barmode": "group",
                "xaxis": {"type": "category", "title": {"text": feature}},
                "yaxis": {"title": {"text": "share"}, "tickformat": ".0%"},
                "legend": {"orientation": "h", "y": 1.12, "x": 0},
                "margin": {"l": 56, "r": 24, "t": 40, "b": 56},
                "height": 340,
            },
        }
    )

    movers = sorted(
        ((last_seen.get(n, 0) - first_seen.get(n, 0), n) for n in categories),
        reverse=True,
    )
    if movers:
        up, down = movers[0], movers[-1]
        print(f"{up[1]} gained {up[0]:+.1%}; {down[1]} moved {down[0]:+.1%}.")
