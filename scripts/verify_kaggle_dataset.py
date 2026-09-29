#!/usr/bin/env python3
"""Verify a private Kaggle dataset against a frozen local input manifest.

The verifier downloads only the expected qrel-free files through Kaggle's
authenticated raw-file endpoint, recomputes SHA-256 locally, and records the
remote file list/version in the supplied manifest.  It fails closed on any
unexpected file, missing file, size mismatch, or hash mismatch.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import sys

import kagglesdk
from kagglesdk.datasets.types import dataset_api_service as api


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def auth_client() -> kagglesdk.KaggleClient:
    if not os.environ.get("KAGGLE_API_TOKEN"):
        token_path = pathlib.Path.home() / ".kaggle" / "access_token"
        if not token_path.exists():
            raise SystemExit(f"missing Kaggle credential {token_path}")
        os.environ["KAGGLE_API_TOKEN"] = token_path.read_text().strip()
    return kagglesdk.KaggleClient(verbose=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--source-dir", type=pathlib.Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--version", type=int, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    expected = dict(manifest.get("files", {}))
    if not expected:
        raise SystemExit("manifest has no expected files")
    local_files = {
        name: args.source_dir.joinpath(name)
        for name in expected
    }
    missing_local = sorted(name for name, path in local_files.items() if not path.is_file())
    if missing_local:
        raise SystemExit(f"missing local files: {missing_local}")
    local_sizes = {name: path.stat().st_size for name, path in local_files.items()}

    client = auth_client()
    status_request = api.ApiGetDatasetStatusRequest()
    status_request.owner_slug = args.owner
    status_request.dataset_slug = args.slug
    status = client.datasets.dataset_api_client.get_dataset_status(status_request)

    list_request = api.ApiListDatasetFilesRequest()
    list_request.owner_slug = args.owner
    list_request.dataset_slug = args.slug
    list_request.dataset_version_number = args.version
    list_request.page_size = 100
    listing = client.datasets.dataset_api_client.list_dataset_files(list_request)
    remote_files = sorted(
        [
            {
                "name": item.name,
                "total_bytes": item.total_bytes,
                "creation_date": str(item.creation_date),
            }
            for item in getattr(listing, "dataset_files", [])
        ],
        key=lambda item: item["name"],
    )
    remote_names = [item["name"] for item in remote_files]
    expected_names = sorted(expected)
    errors: list[str] = []
    if remote_names != expected_names:
        errors.append(f"file-list mismatch: remote={remote_names} expected={expected_names}")

    remote_sha256: dict[str, str] = {}
    remote_headers: dict[str, dict[str, str | None]] = {}
    if not errors:
        for name in expected_names:
            request = api.ApiDownloadDatasetRawRequest()
            request.owner_slug = args.owner
            request.dataset_slug = args.slug
            request.file_name = name
            request.dataset_version_number = args.version
            response = client.datasets.dataset_api_client.download_dataset_raw(request)
            remote_sha256[name] = hashlib.sha256(response.content).hexdigest()
            remote_headers[name] = {
                key: response.headers.get(key)
                for key in ("Content-Length", "ETag", "Last-Modified", "x-goog-hash")
            }
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) != local_sizes[name]:
                errors.append(
                    f"size mismatch for {name}: remote={content_length} local={local_sizes[name]}"
                )
            if remote_sha256[name] != expected[name]:
                errors.append(
                    f"sha256 mismatch for {name}: remote={remote_sha256[name]} expected={expected[name]}"
                )

    now = utc_now()
    manifest.update(
        {
            "status": "remote-verified" if not errors else "remote-verification-failed",
            "remote_verified_utc": now,
            "kaggle_version_number": args.version,
            "remote_api_status": str(getattr(status, "status", None)),
            "remote_file_list": remote_files,
            "remote_sha256": remote_sha256,
            "remote_headers": remote_headers,
            "remote_file_list_match": not errors and remote_names == expected_names,
            "remote_hash_match": not errors and remote_sha256 == expected,
            "qrels_uploaded": False,
        }
    )
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if errors:
        print("remote verification failed", file=sys.stderr)
        for error in errors:
            print(error, file=sys.stderr)
        return 2
    print(json.dumps({
        "dataset": f"{args.owner}/{args.slug}",
        "version": args.version,
        "status": manifest["status"],
        "files": expected_names,
        "sha256_match": True,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
