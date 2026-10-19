# A text-only panel: what this model looks like before anyone reads a chart.
#
# A section does not have to draw. A panel that inspects the model and reports
# in words is a legitimate report section -- and it is what the other panels
# fall back to when a model has nothing they can plot.

from comet_mpm import API

model = API().get_model()

details = model.get_details().get("model", {})
numerical = model.get_numerical_features()
categorical = model.get_categorical_features()
outputs = [name for name in numerical + categorical if name.startswith("prediction_")]
inputs = [name for name in numerical + categorical if name.startswith("feature_")]

print(f"Model: {details.get('name', 'unknown')}")
print(f"Versions: {', '.join(details.get('versions') or ['unknown'])}")
print(f"Latest prediction: {details.get('latestPrediction', 'unknown')}")
print(f"Ground truth attached: {'yes' if details.get('isLabeled') else 'no'}")
print(f"Input columns: {len(inputs)} | Output columns: {len(outputs)}")

if not details.get("isLabeled"):
    print(
        "Without ground truth, accuracy and the classification charts cannot be "
        "computed; drift and distribution charts are unaffected."
    )

volume = model.get_nb_predictions()
if not volume.empty:
    counts = volume.iloc[:, 0].dropna()
    if len(counts):
        quiet = [
            str(index)[:10]
            for index, value in counts.items()
            if value < counts.median() * 0.5
        ]
        print(f"Buckets with data: {len(counts)}")
        if quiet:
            print(f"Unusually quiet buckets ({len(quiet)}): {', '.join(quiet[:8])}")
        else:
            print("No unusually quiet buckets: every bucket is within half the median.")
