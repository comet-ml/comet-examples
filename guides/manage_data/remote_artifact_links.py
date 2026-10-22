# coding: utf-8
import json
import os
import urllib.request

import comet_ml

# WHY: comet_ml.login() reads COMET_API_KEY from the environment - never hardcode it.
comet_ml.login(project_name="comet-example-remote-artifacts")

# DOTA is a public aerial-imagery dataset hosted on a Comet CDN. Each image is
# tracked as a remote asset: Comet stores the link plus metadata, not the bytes.
DOTA_METADATA_URL = "https://cdn.comet.ml/dota_split/DOTA_1.0.json"
BASE_URL = "https://cdn.comet.ml/dota_split"
METADATA_FILE = "./DOTA_1.0.json"
MAX_IMAGES = 200

LABEL_CLASS_NAMES = [
    "plane",
    "baseball-diamond",
    "bridge",
    "ground-track-field",
    "small-vehicle",
    "large-vehicle",
    "ship",
    "tennis-court",
    "basketball-court",
    "storage-tank",
    "soccer-ball-field",
    "roundabout",
    "harbor",
    "swimming-pool",
    "helicopter",
]

if not os.path.exists(METADATA_FILE):
    urllib.request.urlretrieve(DOTA_METADATA_URL, METADATA_FILE)

with open(METADATA_FILE) as f:
    dota_metadata = json.load(f)

annotation_map = {}
for annotation in dota_metadata["annotations"]:
    annotation_map.setdefault(annotation["image_id"], []).append(annotation)

# 1. Track the dataset as an Artifact of remote assets
artifact = comet_ml.Artifact(
    name="DOTA",
    artifact_type="dataset",
    metadata={"class_names": LABEL_CLASS_NAMES},
)
for image in dota_metadata["images"][:MAX_IMAGES]:
    artifact.add_remote(
        f"{BASE_URL}/images/{image['file_name']}",
        metadata={"annotations": annotation_map.get(image["id"], [])},
    )

experiment = comet_ml.start()
experiment.log_artifact(artifact)
print("Logged remote artifact:", experiment.url)
experiment.end()

# 2. Read the links back and fetch one image yourself
experiment = comet_ml.start()
logged_artifact = experiment.get_artifact("DOTA")

remote_assets = logged_artifact.remote_assets
print(f"{len(remote_assets)} remote assets, first 5:")
for asset in remote_assets[:5]:
    print(f"  {asset.logical_path} -> {asset.link}")

# WHY: the SDK only downloads s3:// and gs:// synced assets; a generic link is
# just a string, so fetching the file is up to you.
asset = remote_assets[0]
local_path = "./" + os.path.basename(asset.logical_path)
urllib.request.urlretrieve(asset.link, local_path)
print(f"Fetched {local_path} ({os.path.getsize(local_path)} bytes)")
experiment.end()
