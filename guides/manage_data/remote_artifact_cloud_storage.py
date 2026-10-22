# coding: utf-8
import argparse
import os
import sys
from urllib.parse import urlparse

import comet_ml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Log an S3 or GCS prefix as a synced remote Artifact, then download it."
    )
    parser.add_argument("uri", help="s3://bucket/prefix/ or gs://bucket/prefix/")
    parser.add_argument("--name", default="my-inference-model")
    parser.add_argument("--download-dir", default="./data")
    return parser.parse_args()


def validate_scheme(uri):
    if urlparse(uri).scheme not in ("s3", "gs"):
        print(
            "Use an s3:// or gs:// URI (Google Cloud Storage is gs://, not gcs://)",
            file=sys.stderr,
        )
        sys.exit(2)


def synced_assets(artifact):
    return [
        asset
        for asset in artifact.assets
        if asset.remote and asset.metadata.get("synced") is not False
    ]


if __name__ == "__main__":
    args = parse_args()
    validate_scheme(args.uri)

    # WHY: comet_ml.login() reads COMET_API_KEY from the environment - never hardcode it.
    comet_ml.login(project_name="comet-example-remote-artifacts")

    artifact = comet_ml.Artifact(args.name, artifact_type="model")
    artifact.add_remote(args.uri)

    # WHY: without boto3 / google-cloud-storage and cloud credentials the SDK
    # logs the bare URI as a plain link; this script refuses to pass that off
    # as a synced artifact.
    if not artifact.assets:
        print(
            f"NOT synced: no objects found under {args.uri}. Check the bucket and prefix.",
            file=sys.stderr,
        )
        sys.exit(1)

    assets = synced_assets(artifact)
    if not assets:
        reason = (
            artifact.assets[0].metadata.get("error_message", "sync failed").rstrip(".")
        )
        print(
            f"NOT synced: {reason}. Install boto3 (s3://) or google-cloud-storage "
            "(gs://) and configure cloud credentials.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Synced {len(assets)} objects from {args.uri}")
    for asset in assets:
        print(f"  {asset.logical_path}")

    # 1. Log the synced remote Artifact
    experiment = comet_ml.start()
    experiment.log_artifact(artifact)
    print("Logged synced artifact:", experiment.url)
    experiment.end()

    # 2. Download it back from the cloud storage
    experiment = comet_ml.start()
    logged_artifact = experiment.get_artifact(args.name)
    logged_artifact.download(args.download_dir)
    print("Downloaded to", args.download_dir)
    for root, _dirs, files in os.walk(args.download_dir):
        for file in files:
            print("  " + os.path.join(root, file))
    experiment.end()
