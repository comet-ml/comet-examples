# Manage data with Comet Artifacts

A Comet Artifact is a versioned, aliased snapshot of files or references to files, with
metadata at the artifact and the asset level. Logging an Artifact from one experiment and
fetching it from another gives you data lineage between the two.

## What is in this folder

| Script | Shows | Needs cloud credentials |
|---|---|---|
| `artifacts_intro.py` | Upload local files as a dataset Artifact, fetch it by alias, download it, train a model, log the model as an Artifact, log a new dataset version | no |
| `remote_artifact_links.py` | Generic remote assets: any URL string tracked with per-asset metadata, then reading the `link` back and fetching the file yourself | no |
| `remote_artifact_cloud_storage.py` | `s3://` / `gs://` synced remote assets: one asset per object with checksum and object version, then `download()` back from the bucket | yes |

## Remote artifacts: what the SDK can do

- **Generic links.** `Artifact.add_remote(uri)` accepts any string. Comet stores the link and
  your metadata; `LoggedArtifact.download()` skips these assets, so read `asset.link` from
  `logged_artifact.remote_assets` and fetch the file yourself.
- **Amazon S3.** For a `s3://bucket/prefix/` URI the SDK lists the objects under the prefix and
  logs each one as its own remote asset with its checksum. Needs `boto3` and AWS credentials
  resolved through boto3's standard credential chain. Sync is on by default; pass
  `sync_mode=False` to store only the URI, and `max_synced_objects` (default 10000) caps the
  listing.
- **Google Cloud Storage.** Same behaviour for a `gs://bucket/prefix/` URI. Needs
  `google-cloud-storage` and Application Default Credentials. The scheme is `gs://`.

When object versioning is enabled on the bucket, the SDK also stores each object's version id,
so a later `download()` fetches exactly the bytes that were logged.

Without the client library or credentials, the SDK logs a warning and stores the bare URI as a
plain link with `synced: false` in the asset metadata. `remote_artifact_cloud_storage.py`
detects that and exits non-zero instead of logging a misleading artifact.

## Documentation

- [Register remote artifacts](https://www.comet.com/docs/v2/guides/artifacts/remote-artifacts/)
- [Example: create and use Artifacts](https://www.comet.com/docs/v2/guides/artifacts/example/)
- [Use Artifacts](https://www.comet.com/docs/v2/guides/artifacts/using-artifacts/)

## Setup

Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Set `COMET_API_KEY` in your environment (see the
[quickstart](https://www.comet.com/docs/v2/guides/quickstart/)). For
`remote_artifact_cloud_storage.py` you also need AWS or GCP credentials that can list and read
the bucket.

## Run the examples

```bash
uv run python artifacts_intro.py
uv run python remote_artifact_links.py
uv run python remote_artifact_cloud_storage.py s3://bucket/prefix/ --name my-inference-model --download-dir ./data
uv run python remote_artifact_cloud_storage.py gs://bucket/prefix/
```

`COMET_MODE=offline` is not supported here: every script fetches an Artifact back from Comet.
