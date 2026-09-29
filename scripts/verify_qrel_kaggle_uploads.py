"""Verify the private Kaggle file set and hashes for QREL_VALIDATION_V2.

Kaggle expands uploaded ZIP archives into their member files. This verifier
compares that expanded remote file set byte-for-byte with the local ZIP before
any judge kernel is launched and writes the corresponding v4 upload manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATH = Path.home() / ".kaggle" / "access_token"
OWNER = "rezabarati2"
FREEZE = ROOT / "versions/sem2act-v4/manifests/qrel_validation_v2_freeze.json"
JOBS = {
    "calibration": {
        "slug": "sem2act-v4-qrel-calibration-inputs",
        "zip": ROOT / "kaggle/inputs_qrel_calibration/qrel_v2_bundle.zip",
        "manifest": ROOT / "versions/sem2act-v4/manifests/qrel_validation_v2_calibration_bundle.json",
    },
    "sem2act": {
        "slug": "sem2act-v4-qrel-sem2act-inputs",
        "zip": ROOT / "kaggle/inputs_qrel_sem2act/qrel_v2_bundle.zip",
        "manifest": ROOT / "versions/sem2act-v4/manifests/qrel_validation_v2_sem2act_bundle.json",
    },
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def remote_archive(client, slug: str, version: int) -> tuple[bytes, dict[str, bytes]]:
    from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest

    request = ApiDownloadDatasetRequest()
    request.owner_slug = OWNER
    request.dataset_slug = slug
    request.dataset_version_number = version
    request.raw = False
    response = client.datasets.dataset_api_client.download_dataset(request)
    if hasattr(response, "content"):
        archive_bytes = response.content
    else:
        url = getattr(response, "url", None) or getattr(response, "redirect_url", None)
        if not url:
            raise RuntimeError("Kaggle returned no dataset download body")
        with urllib.request.urlopen(url, timeout=300) as handle:
            archive_bytes = handle.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    return archive_bytes, files


def verify(mode: str) -> dict:
    config = JOBS[mode]
    if not TOKEN_PATH.exists():
        raise SystemExit(f"missing Kaggle credential {TOKEN_PATH}")
    os.environ.setdefault("KAGGLE_API_TOKEN", TOKEN_PATH.read_text().strip())
    with zipfile.ZipFile(config["zip"]) as archive:
        local = {name: archive.read(name) for name in archive.namelist()}
    if any(name.endswith("data/v3/qrels/test.tsv") or "human_annotation" in name for name in local):
        raise SystemExit("local bundle contains prohibited qrel or human-annotation file")

    from kagglesdk import KaggleClient
    from kagglesdk.datasets.types.dataset_api_service import (
        ApiGetDatasetRequest, ApiGetDatasetStatusRequest, ApiListDatasetFilesRequest,
    )

    with KaggleClient() as client:
        get_request = ApiGetDatasetRequest()
        get_request.owner_slug = OWNER
        get_request.dataset_slug = config["slug"]
        dataset = client.datasets.dataset_api_client.get_dataset(get_request).to_dict()
        status_request = ApiGetDatasetStatusRequest()
        status_request.owner_slug = OWNER
        status_request.dataset_slug = config["slug"]
        status = client.datasets.dataset_api_client.get_dataset_status(status_request).to_dict()
        version = int(dataset["currentVersionNumber"])
        if version != 1 or status.get("status") != "READY":
            raise SystemExit(f"unexpected Kaggle dataset state: version={version}, status={status}")
        files_request = ApiListDatasetFilesRequest()
        files_request.owner_slug = OWNER
        files_request.dataset_slug = config["slug"]
        files_request.dataset_version_number = version
        files_request.page_size = 100
        listing = client.datasets.dataset_api_client.list_dataset_files(files_request).to_dict()
        remote_names = {row["name"] for row in listing.get("datasetFiles", [])}
        if listing.get("nextPageToken"):
            raise SystemExit("Kaggle file listing unexpectedly paginated")
        archive_bytes, remote_files = remote_archive(client, config["slug"], version)
        if remote_names != set(remote_files):
            raise SystemExit({"listed_only": sorted(remote_names - set(remote_files)),
                              "archive_only": sorted(set(remote_files) - remote_names)})
        if remote_names != set(local):
            raise SystemExit({"local_only": sorted(set(local) - remote_names),
                              "remote_only": sorted(remote_names - set(local))})
        records = []
        for name in sorted(remote_names):
            data = remote_files[name]
            expected = local[name]
            if data != expected:
                raise SystemExit(f"remote file differs from local bundle: {name}")
            row = next(item for item in listing["datasetFiles"] if item["name"] == name)
            if int(row["totalBytes"]) != len(data):
                raise SystemExit(f"remote byte count differs for {name}")
            records.append({"path": name, "bytes": len(data), "sha256": sha256_bytes(data)})

    freeze = json.loads(FREEZE.read_text())
    result = {
        "schema_version": 1,
        "experiment_id": "v4-qrel-validation-v2",
        "mode": mode,
        "dataset": f"{OWNER}/{config['slug']}",
        "dataset_url": f"https://www.kaggle.com/datasets/{OWNER}/{config['slug']}",
        "dataset_version_number": version,
        "dataset_status": status["status"],
        "visibility": "private",
        "uploaded_utc": dataset.get("lastUpdated"),
        "verified_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_state_sha256": freeze["source_state"]["sha256"],
        "amendment_sha256": freeze["amendment"]["sha256"],
        "local_bundle": {
            "path": str(config["zip"].relative_to(ROOT)),
            "sha256": sha256(config["zip"]),
            "bytes": config["zip"].stat().st_size,
        },
        "remote_archive_sha256": sha256_bytes(archive_bytes),
        "remote_archive_bytes": len(archive_bytes),
        "qrels_included": False,
        "human_annotations_included": False,
        "credentials_included": False,
        "unrelated_repository_content_included": False,
        "remote_file_hashes_verified": True,
        "file_count": len(records),
        "total_bytes": sum(row["bytes"] for row in records),
        "files": records,
    }
    config["manifest"].write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(JOBS), required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.mode), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
