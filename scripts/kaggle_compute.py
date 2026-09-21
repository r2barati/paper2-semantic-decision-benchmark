"""Paper-2 remote-compute runner (Kaggle GPU backend, ADIA pattern).

Push/run/watch/fetch GPU jobs from the local agent -- no browser clicks.
Authentication is machine-level ONLY (~/.kaggle/access_token, exactly as ADIA);
nothing credential-like lives in this repo (see .gitignore).

Jobs:
    rerank-1c   Phase-1C Qwen3 reranker on the frozen prototype (GPU, ~10 min)

Usage:
    python3 scripts/kaggle_compute.py rerank-1c --dataset        # create/update inputs dataset
    python3 scripts/kaggle_compute.py rerank-1c --push          # launch kernel
    python3 scripts/kaggle_compute.py rerank-1c --status        # poll session status
    python3 scripts/kaggle_compute.py rerank-1c --watch         # block until terminal state
    python3 scripts/kaggle_compute.py rerank-1c --fetch         # download + verify + install outputs
    python3 scripts/kaggle_compute.py rerank-1c --test          # local pytest gate on fetched outputs

The same mechanism serves future jobs (Qwen3-8B-AWQ inference, 200/4000 scale):
add a JOBS entry + kernel script, nothing else changes. GitHub Actions stays CI-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOKEN_PATH = pathlib.Path.home() / ".kaggle" / "access_token"
if not TOKEN_PATH.exists():
    raise SystemExit(f"missing machine credential {TOKEN_PATH} (ADIA pattern); refusing to run")
os.environ.setdefault("KAGGLE_API_TOKEN", TOKEN_PATH.read_text().strip())

from kagglesdk import KaggleClient  # noqa: E402
from kagglesdk.blobs.types.blob_api_service import (  # noqa: E402
    ApiBlobType, ApiStartBlobUploadRequest)
from kagglesdk.datasets.types.dataset_api_service import (  # noqa: E402
    ApiCreateDatasetRequest, ApiCreateDatasetVersionRequest,
    ApiCreateDatasetVersionRequestBody, ApiDatasetNewFile)
from kagglesdk.kernels.types.kernels_api_service import (  # noqa: E402
    ApiDownloadKernelOutputRequest, ApiGetKernelSessionStatusRequest,
    ApiSaveKernelRequest)

OWNER = "rezabarati2"

JOBS = {
    "rerank-1c": {
        "slug": "paper2-v3-rerank-1c",
        "title": "Paper2 v3 rerank 1c",
        "script": "kaggle_kernel/p2_rerank_1c.py",
        "gpu": True,
        "internet": True,
        "dataset_slug": "paper2-v3proto-inputs",
        "dataset_title": "paper2-v3proto-inputs",
        "dataset_dir": "kaggle/inputs",
        "outputs": ["rerank-qwen3-instruct.trec",
                    "rerank-qwen3-instruct_qlevel.json",
                    "run_manifest.json"],
        "dest_dir": "runs/v3proto",
        "verify": "rerank",
    },
    "encode-2b": {
        "slug": "paper2-v3-encode-2b",
        "title": "Paper2 v3 encode 2b",
        "script": "kaggle_kernel/p2_encode_2b.py",
        "gpu": True,
        "internet": True,
        "dataset_slug": "paper2-v3full-inputs",
        "dataset_title": "paper2-v3full-inputs",
        "dataset_dir": "kaggle/inputs_full",
        "outputs": ["qwen3emb_full_vectors.npz",
                    "dense_full.trec",
                    "hybrid_k30_dev.trec",
                    "hybrid_k60_dev.trec",
                    "hybrid_k120_dev.trec",
                    "rerank_top30_dev.trec",
                    "rerank_top50_dev.trec",
                    "rerank_top100_dev.trec",
                    "dev_selection.json",
                    "run_manifest.json"],
        "dest_dir": "runs/v3",
        "verify": "encode2b",
    },
    "rerank-refine": {
        "slug": "paper2-v3-rerank-refine",
        "title": "Paper2 v3 rerank refine",
        "script": "kaggle_kernel/p2_rerank_refine.py",
        "gpu": True,
        "internet": True,
        "dataset_slug": "paper2-v3full-inputs",
        "dataset_title": "paper2-v3full-inputs",
        "dataset_dir": "kaggle/inputs_full",
        "outputs": ["rerank_k120_top30_dev.trec",
                    "rerank_k120_top50_dev.trec",
                    "rerank_k120_top100_dev.trec",
                    "dev_selection_refine.json",
                    "run_manifest.json"],
        "dest_dir": "runs/v3",
        "verify": "refine",
    },
    "runs-main": {
        "slug": "paper2-v3-runs-main",
        "title": "Paper2 v3 runs main",
        "script": "kaggle_kernel/p2_runs_main.py",
        "gpu": True,
        "internet": True,
        "dataset_slug": "paper2-v3main-inputs",
        "dataset_title": "paper2-v3main-inputs",
        "dataset_dir": "kaggle/inputs_main",
        "outputs": ["qwen3emb_main_vectors.npz",
                    "dense_full.trec",
                    "hybrid_k120_full.trec",
                    "rerank_full.trec",
                    "run_manifest.json"],
        "dest_dir": "runs/v3main",
        "verify": "runsmain",
    },
    "awq-probe": {
        "slug": "paper2-v3-awq-probe",
        "title": "Paper2 v3 awq probe",
        "script": "kaggle_kernel/p2_awq_probe.py",
        "gpu": True,
        "internet": True,
        "dataset_slug": "paper2-v3main-inputs",
        "dataset_title": "paper2-v3main-inputs",
        "dataset_dir": "kaggle/inputs_main",
        "outputs": ["probe_report.json"],
        "dest_dir": "runs/v3main",
        "verify": "probe",
    },
    "awq-b0": {
        "slug": "paper2-v3-awq-b0",
        "title": "Paper2 v3 awq b0",
        "script": "kaggle_kernel/p2_awq_beliefs.py",
        "gpu": True,
        "internet": True,
        "dataset_slug": "paper2-v3awq-inputs",
        "dataset_title": "paper2-v3awq-inputs",
        "dataset_dir": "kaggle/inputs_awq",
        "outputs": ["cache_shard_0_50.zip",
                    "shard_manifest.json"],
        "dest_dir": "runs/v3main_awq",
        "verify": "awqbeliefs",
    },
    "awq-b1": {
        "slug": "paper2-v3-awq-b1",
        "title": "Paper2 v3 awq b1",
        "script": "kaggle_kernel/p2_awq_beliefs.py",
        "gpu": True,
        "internet": True,
        "dataset_slug": "paper2-v3awq-inputs",
        "dataset_title": "paper2-v3awq-inputs",
        "dataset_dir": "kaggle/inputs_awq",
        "outputs": ["cache_shard_50_100.zip",
                    "shard_manifest.json"],
        "dest_dir": "runs/v3main_awq",
        "verify": "awqbeliefs",
    },
    "awq-b2": {
        "slug": "paper2-v3-awq-b2",
        "title": "Paper2 v3 awq b2",
        "script": "kaggle_kernel/p2_awq_beliefs.py",
        "gpu": True,
        "internet": True,
        "dataset_slug": "paper2-v3awq-inputs",
        "dataset_title": "paper2-v3awq-inputs",
        "dataset_dir": "kaggle/inputs_awq",
        "outputs": ["cache_shard_100_150.zip",
                    "shard_manifest.json"],
        "dest_dir": "runs/v3main_awq",
        "verify": "awqbeliefs",
    },
    "awq-b3": {
        "slug": "paper2-v3-awq-b3",
        "title": "Paper2 v3 awq b3",
        "script": "kaggle_kernel/p2_awq_beliefs.py",
        "gpu": True,
        "internet": True,
        "dataset_slug": "paper2-v3awq-inputs",
        "dataset_title": "paper2-v3awq-inputs",
        "dataset_dir": "kaggle/inputs_awq",
        "verify": "awqbeliefs",
        "outputs": ["cache_shard_150_200.zip",
                    "shard_manifest.json"],
        "dest_dir": "runs/v3main_awq",
    },
    "phi4-probe": {
        "slug": "paper2-v3-phi4-probe",
        "title": "Paper2 v3 phi4 probe",
        "script": "kaggle_kernel/p2_phi4_probe.py",
        "gpu": True,
        "internet": True,
        "dataset_slug": "paper2-v3main-inputs",
        "dataset_title": "paper2-v3main-inputs",
        "dataset_dir": "kaggle/inputs_main",
        "outputs": ["probe_report.json"],
        "dest_dir": "runs/v3main_phi4",
        "verify": "phi4probe",
    },
    "qwen14-probe": {
        "slug": "paper2-v3-qwen14-probe",
        "title": "Paper2 v3 qwen14 probe",
        "script": "kaggle_kernel/p2_qwen14_probe.py",
        "gpu": True,
        "internet": True,
        "dataset_slug": "paper2-v3main-inputs",
        "dataset_title": "paper2-v3main-inputs",
        "dataset_dir": "kaggle/inputs_main",
        "outputs": ["probe_report.json"],
        "dest_dir": "runs/v3main_qwen14",
        "verify": "qwen14probe",
    },
    "qwen14-b0": {
        "slug": "paper2-v3-qwen14-b0",
        "title": "Paper2 v3 qwen14 b0",
        "script": "kaggle_kernel/p2_qwen14_beliefs.py",
        "gpu": True,
        "internet": True,
        "dataset_slug": "paper2-v3qwen14-inputs",
        "dataset_title": "paper2-v3qwen14-inputs",
        "dataset_dir": "kaggle/inputs_qwen14",
        "outputs": ["cache_shard_0_50.zip",
                    "shard_manifest.json"],
        "dest_dir": "runs/v3main_qwen14",
        "verify": "qwen14beliefs",
    },
    "qwen14-b1": {
        "slug": "paper2-v3-qwen14-b1",
        "title": "Paper2 v3 qwen14 b1",
        "script": "kaggle_kernel/p2_qwen14_beliefs.py",
        "gpu": True,
        "internet": True,
        "dataset_slug": "paper2-v3qwen14-inputs",
        "dataset_title": "paper2-v3qwen14-inputs",
        "dataset_dir": "kaggle/inputs_qwen14",
        "outputs": ["cache_shard_50_100.zip",
                    "shard_manifest.json"],
        "dest_dir": "runs/v3main_qwen14",
        "verify": "qwen14beliefs",
    },
    "qwen14-b2": {
        "slug": "paper2-v3-qwen14-b2",
        "title": "Paper2 v3 qwen14 b2",
        "script": "kaggle_kernel/p2_qwen14_beliefs.py",
        "gpu": True,
        "internet": True,
        "dataset_slug": "paper2-v3qwen14-inputs",
        "dataset_title": "paper2-v3qwen14-inputs",
        "dataset_dir": "kaggle/inputs_qwen14",
        "outputs": ["cache_shard_100_150.zip",
                    "shard_manifest.json"],
        "dest_dir": "runs/v3main_qwen14",
        "verify": "qwen14beliefs",
    },
    "qwen14-b3": {
        "slug": "paper2-v3-qwen14-b3",
        "title": "Paper2 v3 qwen14 b3",
        "script": "kaggle_kernel/p2_qwen14_beliefs.py",
        "gpu": True,
        "internet": True,
        "dataset_slug": "paper2-v3qwen14-inputs",
        "dataset_title": "paper2-v3qwen14-inputs",
        "dataset_dir": "kaggle/inputs_qwen14",
        "outputs": ["cache_shard_150_200.zip",
                    "shard_manifest.json"],
        "dest_dir": "runs/v3main_qwen14",
        "verify": "qwen14beliefs",
    },
    "phi4-b0": {
        "slug": "paper2-v3-phi4-b0",
        "title": "Paper2 v3 phi4 b0",
        "script": "kaggle_kernel/p2_phi4_beliefs.py",
        "gpu": True,
        "internet": True,
        "dataset_slug": "paper2-v3phi4-inputs",
        "dataset_title": "paper2-v3phi4-inputs",
        "dataset_dir": "kaggle/inputs_phi4",
        "outputs": ["cache_shard_0_50.zip",
                    "shard_manifest.json"],
        "dest_dir": "runs/v3main_phi4",
        "verify": "phi4beliefs",
    },
    "phi4-b1": {
        "slug": "paper2-v3-phi4-b1",
        "title": "Paper2 v3 phi4 b1",
        "script": "kaggle_kernel/p2_phi4_beliefs.py",
        "gpu": True,
        "internet": True,
        "dataset_slug": "paper2-v3phi4-inputs",
        "dataset_title": "paper2-v3phi4-inputs",
        "dataset_dir": "kaggle/inputs_phi4",
        "outputs": ["cache_shard_50_100.zip",
                    "shard_manifest.json"],
        "dest_dir": "runs/v3main_phi4",
        "verify": "phi4beliefs",
    },
    "phi4-b2": {
        "slug": "paper2-v3-phi4-b2",
        "title": "Paper2 v3 phi4 b2",
        "script": "kaggle_kernel/p2_phi4_beliefs.py",
        "gpu": True,
        "internet": True,
        "dataset_slug": "paper2-v3phi4-inputs",
        "dataset_title": "paper2-v3phi4-inputs",
        "dataset_dir": "kaggle/inputs_phi4",
        "outputs": ["cache_shard_100_150.zip",
                    "shard_manifest.json"],
        "dest_dir": "runs/v3main_phi4",
        "verify": "phi4beliefs",
    },
    "phi4-b3": {
        "slug": "paper2-v3-phi4-b3",
        "title": "Paper2 v3 phi4 b3",
        "script": "kaggle_kernel/p2_phi4_beliefs.py",
        "gpu": True,
        "internet": True,
        "dataset_slug": "paper2-v3phi4-inputs",
        "dataset_title": "paper2-v3phi4-inputs",
        "dataset_dir": "kaggle/inputs_phi4",
        "outputs": ["cache_shard_150_200.zip",
                    "shard_manifest.json"],
        "dest_dir": "runs/v3main_phi4",
        "verify": "phi4beliefs",
    },
    "loo-pilot-8b": {
        "slug": "paper2-v3-loo-pilot-8b",
        "title": "Paper2 v3 loo pilot 8b",
        "script": "kaggle_kernel/p2_loo_8b.py",
        "gpu": True,
        "internet": True,
        "dataset_slug": "paper2-v3loo-inputs",
        "dataset_title": "paper2-v3loo-inputs",
        "dataset_dir": "kaggle/inputs_loo",
        "outputs": ["cache_loo_dev_8b.zip",
                    "pilot_manifest_8b.json"],
        "dest_dir": "runs/v3main_loo",
        "verify": "loopilot8b",
    },
    "loo-pilot-14b": {
        "slug": "paper2-v3-loo-pilot-14b",
        "title": "Paper2 v3 loo pilot 14b",
        "script": "kaggle_kernel/p2_loo_14b.py",
        "gpu": True,
        "internet": True,
        "dataset_slug": "paper2-v3loo-inputs",
        "dataset_title": "paper2-v3loo-inputs",
        "dataset_dir": "kaggle/inputs_loo",
        "outputs": ["cache_loo_dev_14b.zip",
                    "pilot_manifest_14b.json"],
        "dest_dir": "runs/v3main_loo",
        "verify": "loopilot14b",
    },
    "loo-test-8b": {
        "slug": "paper2-v3-loo-test-8b",
        "title": "Paper2 v3 loo test 8b",
        "script": "kaggle_kernel/p2_loo_test_8b.py",
        "gpu": True,
        "internet": True,
        "dataset_slug": "paper2-v3lootest-inputs",
        "dataset_title": "paper2-v3lootest-inputs",
        "dataset_dir": "kaggle/inputs_loo_test",
        "outputs": ["cache_loo_test_8b.zip",
                    "manifest_test_8b.json"],
        "dest_dir": "runs/v3main_lootest",
        "verify": "lootest8b",
    },
    "loo-test-14b": {
        "slug": "paper2-v3-loo-test-14b",
        "title": "Paper2 v3 loo test 14b",
        "script": "kaggle_kernel/p2_loo_test_14b.py",
        "gpu": True,
        "internet": True,
        "dataset_slug": "paper2-v3lootest-inputs",
        "dataset_title": "paper2-v3lootest-inputs",
        "dataset_dir": "kaggle/inputs_loo_test",
        "outputs": ["cache_loo_test_14b.zip",
                    "manifest_test_14b.json"],
        "dest_dir": "runs/v3main_lootest",
        "verify": "lootest14b",
    },
    "tracka-sim-pilot": {
        "slug": "paper2-tracka-sim-pilot",
        "title": "Paper2 tracka sim pilot",
        "script": "kaggle_kernel/p2_tracka_sim.py",
        "gpu": False,
        "internet": True,
        "dataset_slug": "paper2-tracka-sim-inputs",
        "dataset_title": "paper2-tracka-sim-inputs",
        "dataset_dir": "kaggle/inputs_tracka",
        "outputs": ["episodes_tracka_pilot.parquet",
                    "shard_manifest.json"],
        "dest_dir": "runs/v3main_tracka",
        "verify": "trackasim",
        "shard_spec": "shard_pilot.json",
        "gate": "full-overlap",
    },
    "tracka-sim-b1": {
        "slug": "paper2-tracka-sim-b1",
        "title": "Paper2 tracka sim b1",
        "script": "kaggle_kernel/p2_tracka_b1.py",
        "gpu": False,
        "internet": True,
        "dataset_slug": "paper2-tracka-sim-inputs",
        "dataset_title": "paper2-tracka-sim-inputs",
        "dataset_dir": "kaggle/inputs_tracka",
        "outputs": ["episodes_tracka_b1.parquet",
                    "manifest_tracka_b1.json"],
        "dest_dir": "runs/v3main_tracka",
        "verify": "trackasim",
        "shard_spec": "shard_b1.json",
        "gate": "sample-5pct",
    },
    "tracka-sim-b2": {
        "slug": "paper2-tracka-sim-b2",
        "title": "Paper2 tracka sim b2",
        "script": "kaggle_kernel/p2_tracka_b2.py",
        "gpu": False,
        "internet": True,
        "dataset_slug": "paper2-tracka-sim-inputs",
        "dataset_title": "paper2-tracka-sim-inputs",
        "dataset_dir": "kaggle/inputs_tracka",
        "outputs": ["episodes_tracka_b2.parquet",
                    "manifest_tracka_b2.json"],
        "dest_dir": "runs/v3main_tracka",
        "verify": "trackasim",
        "shard_spec": "shard_b2.json",
        "gate": "sample-5pct",
    },
    "tracka-sim-b3": {
        "slug": "paper2-tracka-sim-b3",
        "title": "Paper2 tracka sim b3",
        "script": "kaggle_kernel/p2_tracka_b3.py",
        "gpu": False,
        "internet": True,
        "dataset_slug": "paper2-tracka-sim-inputs",
        "dataset_title": "paper2-tracka-sim-inputs",
        "dataset_dir": "kaggle/inputs_tracka",
        "outputs": ["episodes_tracka_b3.parquet",
                    "manifest_tracka_b3.json"],
        "dest_dir": "runs/v3main_tracka",
        "verify": "trackasim",
        "shard_spec": "shard_b3.json",
        "gate": "sample-5pct",
    },
    "tracka-sim-b4": {
        "slug": "paper2-tracka-sim-b4",
        "title": "Paper2 tracka sim b4",
        "script": "kaggle_kernel/p2_tracka_b4.py",
        "gpu": False,
        "internet": True,
        "dataset_slug": "paper2-tracka-sim-inputs",
        "dataset_title": "paper2-tracka-sim-inputs",
        "dataset_dir": "kaggle/inputs_tracka",
        "outputs": ["episodes_tracka_b4.parquet",
                    "manifest_tracka_b4.json"],
        "dest_dir": "runs/v3main_tracka",
        "verify": "trackasim",
        "shard_spec": "shard_b4.json",
        "gate": "sample-5pct",
    },
    "tracka-sim-b5": {
        "slug": "paper2-tracka-sim-b5",
        "title": "Paper2 tracka sim b5",
        "script": "kaggle_kernel/p2_tracka_b5.py",
        "gpu": False,
        "internet": True,
        "dataset_slug": "paper2-tracka-sim-inputs",
        "dataset_title": "paper2-tracka-sim-inputs",
        "dataset_dir": "kaggle/inputs_tracka",
        "outputs": ["episodes_tracka_b5.parquet",
                    "manifest_tracka_b5.json"],
        "dest_dir": "runs/v3main_tracka",
        "verify": "trackasim",
        "shard_spec": "shard_b5.json",
        "gate": "sample-5pct",
    },
    "tracka-sim-b6": {
        "slug": "paper2-tracka-sim-b6",
        "title": "Paper2 tracka sim b6",
        "script": "kaggle_kernel/p2_tracka_b6.py",
        "gpu": False,
        "internet": True,
        "dataset_slug": "paper2-tracka-sim-inputs",
        "dataset_title": "paper2-tracka-sim-inputs",
        "dataset_dir": "kaggle/inputs_tracka",
        "outputs": ["episodes_tracka_b6.parquet",
                    "manifest_tracka_b6.json"],
        "dest_dir": "runs/v3main_tracka",
        "verify": "trackasim",
        "shard_spec": "shard_b6.json",
        "gate": "sample-5pct",
    },
    "agentick-probe": {
        "slug": "paper2-agentick-probe",
        "title": "Paper2 agentick probe",
        "script": "kaggle_kernel/p2_agentick_probe.py",
        "gpu": False,
        "internet": True,
        "dataset_slug": "paper2-agentick-inputs",
        "dataset_title": "paper2-agentick-inputs",
        "dataset_dir": "kaggle/inputs_agentick",
        "outputs": ["agentick_probe.json"],
        "dest_dir": "runs/v3main_agentick",
        "verify": "agentickprobe",
    },
}

POLL_SECONDS = 120
WATCH_TIMEOUT_SECONDS = 6 * 3600


def _slugify(title):
    import re
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", title.lower())).strip("-")


def _lint_kernel(path):
    r = subprocess.run([sys.executable, "-m", "pyflakes", str(path)],
                       capture_output=True, text=True)
    bad = [l for l in (r.stdout + r.stderr).splitlines()
           if l.strip() and "imported but unused" not in l
           and "unable to detect undefined names" not in l
           and "redefinition of unused" not in l]
    if bad:
        raise SystemExit("lint failed, not pushing:\n  " + "\n  ".join(bad))


# --- datasets ---------------------------------------------------------------

def _stage_files(job):
    d = ROOT / job["dataset_dir"]
    files = sorted(p for p in d.rglob("*") if p.is_file())
    if not files:
        raise SystemExit(f"nothing staged in {d}")
    return files


def _upload_blob(client, path):
    size = path.stat().st_size
    req = ApiStartBlobUploadRequest()
    req.type = ApiBlobType.DATASET
    req.name = path.name
    req.content_length = size
    req.last_modified_epoch_seconds = int(path.stat().st_mtime)
    resp = client.blobs.blob_api_client.start_blob_upload(req)
    print(f"    {path.name}: {size / 1e6:.2f} MB ...", flush=True)
    with path.open("rb") as fh:
        put = urllib.request.Request(resp.create_url, data=fh, method="PUT",
                                     headers={"Content-Length": str(size)})
        with urllib.request.urlopen(put, timeout=3600) as r:
            if r.status not in (200, 201, 204):
                raise RuntimeError(f"PUT failed for {path.name}: {r.status}")
    return resp.token


def cmd_dataset(job, create):
    files = _stage_files(job)
    print(f"uploading {len(files)} files as PRIVATE dataset "
          f"{OWNER}/{job['dataset_slug']}")
    with KaggleClient() as client:
        tokens = [_upload_blob(client, p) for p in files]
        new_files = []
        for t in tokens:
            f = ApiDatasetNewFile()
            f.token = t
            new_files.append(f)
        if create:
            req = ApiCreateDatasetRequest()
            req.owner_slug = OWNER
            req.slug = job["dataset_slug"]
            req.title = job["dataset_title"]
            req.license_name = "other"
            req.is_private = True
            req.files = new_files
            resp = client.datasets.dataset_api_client.create_dataset(req)
        else:
            req = ApiCreateDatasetVersionRequest()
            req.owner_slug = OWNER
            req.dataset_slug = job["dataset_slug"]
            body = ApiCreateDatasetVersionRequestBody()
            body.version_notes = f"paper2 {job['dataset_slug']} refresh"
            body.files = new_files
            req.body = body
            resp = client.datasets.dataset_api_client.create_dataset_version(req)
    print("response:", resp)


# --- kernels ----------------------------------------------------------------

def cmd_push(job_name, job):
    slug, title = job["slug"], job["title"]
    if _slugify(title) != slug:
        raise SystemExit(f"title {title!r} slugifies to {_slugify(title)!r}, not {slug!r}")
    path = ROOT / job["script"]
    _lint_kernel(path)
    req = ApiSaveKernelRequest()
    req.slug = f"{OWNER}/{slug}"
    req.new_title = title
    req.text = path.read_text()
    req.language = "python"
    req.kernel_type = "script"
    req.is_private = True
    req.enable_gpu = job["gpu"]
    req.enable_internet = job["internet"]
    req.dataset_data_sources = [f"{OWNER}/{job['dataset_slug']}"]
    req.category_ids = []
    with KaggleClient() as c:
        print(job_name, "->", c.kernels.kernels_api_client.save_kernel(req))


def cmd_status(job):
    req = ApiGetKernelSessionStatusRequest()
    req.user_name, req.kernel_slug = OWNER, job["slug"]
    with KaggleClient() as c:
        resp = c.kernels.kernels_api_client.get_kernel_session_status(req)
    print(f"{job['slug']}: {resp}")
    return str(resp)


def cmd_watch(job, timeout=WATCH_TIMEOUT_SECONDS):
    t0 = time.time()
    notfound = 0
    while time.time() - t0 < timeout:
        try:
            out = cmd_status(job)
            notfound = 0
        except Exception as e:
            # Fresh kernels 404 until the first session registers; tolerate briefly.
            notfound += 1
            print(f"  status unavailable ({type(e).__name__} x{notfound}), retrying", flush=True)
            if notfound > 10:
                raise
            time.sleep(POLL_SECONDS)
            continue
        for state in ("COMPLETE", "FAILED", "CANCELLED", "ERROR"):
            if state in out:
                return state
        print(f"  ... {(time.time() - t0) / 60:.0f} min elapsed", flush=True)
        time.sleep(POLL_SECONDS)
    raise SystemExit("watch timed out")


def _sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def cmd_fetch(job):
    req = ApiDownloadKernelOutputRequest()
    req.owner_slug, req.kernel_slug = OWNER, job["slug"]
    dest = ROOT / "artifacts" / "kaggle_out" / job["slug"]
    dest.mkdir(parents=True, exist_ok=True)
    with KaggleClient() as c:
        resp = c.kernels.kernels_api_client.download_kernel_output(req)
        data = resp.content if hasattr(resp, "content") else None
    if not data:
        raise SystemExit(f"empty output response {getattr(resp, 'status_code', '?')}")
    zpath = dest / "output.zip"
    zpath.write_bytes(data)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest)
        print("extracted:", z.namelist())
    # verify: every expected output present + job-specific re-verification
    for name in job["outputs"]:
        if not (dest / name).exists():
            raise SystemExit(f"missing expected output {name} in kernel outputs")
    {"rerank": _verify_rerank, "encode2b": _verify_encode2b,
     "refine": _verify_refine, "runsmain": _verify_runsmain,
     "probe": _verify_probe, "awqbeliefs": _verify_awqbeliefs,
     "phi4probe": _verify_phi4probe, "phi4beliefs": _verify_phi4beliefs,
      "qwen14probe": _verify_qwen14probe,
      "qwen14beliefs": _verify_qwen14beliefs,
      "loopilot8b": _verify_loopilot8b,
      "loopilot14b": _verify_loopilot14b,
      "lootest8b": _verify_lootest8b,
      "lootest14b": _verify_lootest14b,
      "trackasim": _verify_trackasim,
      "agentickprobe": _verify_agentickprobe}[job["verify"]](dest, job)
    # install into runs/ (only after verification passes)
    (ROOT / job["dest_dir"]).mkdir(parents=True, exist_ok=True)
    for name in job["outputs"]:
        if name == "run_manifest.json":
            continue
        target = ROOT / job["dest_dir"] / name
        target.write_bytes((dest / name).read_bytes())
        print(f"installed {target} sha={_sha(target)[:12]}")
    (ROOT / job["dest_dir"] / "kaggle_fetch_manifest.json").write_text(json.dumps({
        "job": job["slug"], "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": {n: _sha(ROOT / job["dest_dir"] / n)
                  for n in job["outputs"] if n != "run_manifest.json"},
    }, indent=2))


def _verify_rerank(dest, job):
    """Recompute ir_measures aggregates locally from the fetched trec and require
    equality with the kernel-reported manifest (catches truncation/corruption)."""
    import ir_measures
    from ir_measures import nDCG, Recall, RR, AP, Qrel, ScoredDoc
    qrels = []
    for fn in ("dev.tsv", "test.tsv"):
        with open(ROOT / job["dataset_dir"] / fn) as f:
            for line in f.read().splitlines()[1:]:
                qid, did, g = line.split("\t")
                qrels.append(Qrel(query_id=qid, doc_id=did, relevance=int(g)))
    run = []
    with open(dest / "rerank-qwen3-instruct.trec") as f:
        for line in f:
            qid, _, did, _, s, _ = line.split()
            run.append(ScoredDoc(query_id=qid, doc_id=did, score=float(s)))
    res = {str(k): round(float(v), 3)
           for k, v in ir_measures.calc_aggregate(
               [nDCG@10, Recall@10, Recall@20, RR, AP], qrels, run).items()}
    claimed = json.loads((dest / "run_manifest.json").read_text())["aggregate"]
    print("kernel-claimed:", claimed)
    print("locally-recomputed:", res)
    if res != claimed:
        raise SystemExit(f"aggregate mismatch: {res} != {claimed}")
    print("verify: aggregates match")


def cmd_logs(job, out_path=None):
    import json as _json
    from kagglesdk.kernels.types.kernels_api_service import (
        ApiListKernelSessionOutputRequest)
    req = ApiListKernelSessionOutputRequest()
    req.user_name, req.kernel_slug = OWNER, job["slug"]
    with KaggleClient() as c:
        resp = c.kernels.kernels_api_client.list_kernel_session_output(req)
    raw = getattr(resp, "log", "") or ""
    try:
        entries = _json.loads(raw) if raw else []
    except _json.JSONDecodeError:
        print(f"no parseable log yet ({len(raw)} chars); session likely still starting")
        return ""
    if not entries:
        print("log empty; session likely still starting")
        return ""
    text = "\n".join(f"[{e['stream_name']} {e['time']:.1f}] {e['data']}" for e in entries)
    if out_path:
        pathlib.Path(out_path).write_text(text)
        print(f"log saved to {out_path} ({len(entries)} entries)")
    else:
        print(text[-4000:])
    return text


def _verify_encode2b(dest, job):
    """Recompute DEV nDCG@10 locally (ir_measures) for every fetched dev run and
    require equality with the kernel's dev_selection.json. Test qrels are local
    only and never enter this check beyond dev (dev.tsv is the attached file)."""
    import ir_measures
    from ir_measures import nDCG, Qrel, ScoredDoc
    qrels = []
    with open(ROOT / "kaggle" / "inputs_full" / "dev.tsv") as f:
        for line in f.read().splitlines()[1:]:
            qid, did, g = line.split("\t")
            qrels.append(Qrel(query_id=qid, doc_id=did, relevance=int(g)))
    claimed = json.loads((dest / "dev_selection.json").read_text())
    for trec, key in [("hybrid_k30_dev.trec", "hybrid_rrf30"),
                      ("hybrid_k60_dev.trec", "hybrid_rrf60"),
                      ("hybrid_k120_dev.trec", "hybrid_rrf120"),
                      ("rerank_top30_dev.trec", "rerank_top30"),
                      ("rerank_top50_dev.trec", "rerank_top50"),
                      ("rerank_top100_dev.trec", "rerank_top100")]:
        run = []
        with open(dest / trec) as f:
            for line in f:
                qid, _, did, _, s, _ = line.split()
                run.append(ScoredDoc(query_id=qid, doc_id=did, score=float(s)))
        got = round(float(ir_measures.calc_aggregate(
            [nDCG@10], qrels, run)[nDCG@10]), 4)
        print(f"  {key}: kernel={claimed[key]} local={got}")
        if abs(got - claimed[key]) > 5e-4:
            raise SystemExit(f"aggregate mismatch on {key}: {got} != {claimed[key]}")
    # dense_full vectors sanity: shape + unit norms
    z = __import__("numpy").load(dest / "qwen3emb_full_vectors.npz")
    assert z["doc_emb"].shape[1] == 1024 and z["q_emb"].shape[1] == 1024
    print("verify: dev selection + vectors ok")


def _verify_refine(dest, job):
    """Refinement dev runs: same local ir_measures re-verification pattern."""
    import ir_measures
    from ir_measures import nDCG, Qrel, ScoredDoc
    qrels = []
    with open(ROOT / "kaggle" / "inputs_full" / "dev.tsv") as f:
        for line in f.read().splitlines()[1:]:
            qid, did, g = line.split("\t")
            qrels.append(Qrel(query_id=qid, doc_id=did, relevance=int(g)))
    claimed = json.loads((dest / "dev_selection_refine.json").read_text())
    for trec, key in [("rerank_k120_top30_dev.trec", "rerank_k120_top30"),
                      ("rerank_k120_top50_dev.trec", "rerank_k120_top50"),
                      ("rerank_k120_top100_dev.trec", "rerank_k120_top100")]:
        run = []
        with open(dest / trec) as f:
            for line in f:
                qid, _, did, _, s, _ = line.split()
                run.append(ScoredDoc(query_id=qid, doc_id=did, score=float(s)))
        got = round(float(ir_measures.calc_aggregate(
            [nDCG@10], qrels, run)[nDCG@10]), 4)
        print(f"  {key}: kernel={claimed[key]} local={got}")
        if abs(got - claimed[key]) > 5e-4:
            raise SystemExit(f"aggregate mismatch on {key}: {got} != {claimed[key]}")
    print("verify: refine selection ok")


def _verify_runsmain(dest, job):
    """Structural verification (no qrels attached by design): 200 queries x full
    depth, no dup doc ids, manifest pins match frozen config. Metrics are computed
    locally at analysis time only."""
    import json as _j
    man = _j.loads((dest / "run_manifest.json").read_text())
    assert man["emb_rev"] == "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3", man
    assert man["rr_rev"] == "e61197ed45024b0ed8a2d74b80b4d909f1255473", man
    assert man["rrf_k"] == 120 and man["topn"] == 30, man
    assert man["n_queries"] == 200 and man["n_docs"] >= 3000, man
    n_docs = man["n_docs"]
    for trec in ("dense_full.trec", "hybrid_k120_full.trec", "rerank_full.trec"):
        seen = {}
        with open(dest / trec) as f:
            for line in f:
                qid, _, did, _, _, _ = line.split()
                seen.setdefault(qid, []).append(did)
        assert len(seen) == 200, f"{trec}: {len(seen)} queries"
        for qid, ids in seen.items():
            assert len(ids) == n_docs, f"{trec}/{qid}: {len(ids)} docs"
            assert len(set(ids)) == n_docs, f"{trec}/{qid}: dup ids"
    z = __import__("numpy").load(dest / "qwen3emb_main_vectors.npz")
    assert z["doc_emb"].shape == (n_docs, 1024) and z["q_emb"].shape == (200, 1024)
    print("verify: main runs structurally ok")


def _verify_probe(dest, job):
    import json as _j
    rep = _j.loads((dest / "probe_report.json").read_text())
    print("probe:", {k: v for k, v in rep.items() if k != "sample"})
    if not (rep.get("identical") and rep.get("parses")):
        raise SystemExit("AWQ probe failed: non-deterministic or unparseable")


def _verify_awqbeliefs(dest, job):
    import json as _j
    import random as _random
    import zipfile as _zip
    man = _j.loads((dest / "shard_manifest.json").read_text())
    print("shard manifest:", {k: v for k, v in man.items() if k != "failures"})
    assert man["model"] == "Qwen/Qwen3-8B-AWQ", man
    assert man["revision"] == "4da05a8edb55c6046cce958586c33b61da07bb79", man
    assert man["c1_sha"] == "66e6890ba9c5464c" and man["c3_sha"] == "5966cadde90e8236", man
    assert man["tau"] == 0.5, man
    assert man["seal"] == "no qrels attached; guard passed", man
    for f in ("consumers_v3.py", "interpreter.py", "events.py"):
        frozen = _sha(ROOT / "src" / f)[:16]
        assert man["src_snap_sha"][f] == frozen, (f, man["src_snap_sha"][f], frozen)
    print("verify: kernel src_snap byte-identical to frozen src")
    s, e = man["shard"]
    zname = f"cache_shard_{s}_{e}.zip"
    assert zname in job["outputs"], f"manifest shard {man['shard']} != job {job['slug']}"
    with _zip.ZipFile(dest / zname) as z:
        names = z.namelist()
        assert len(names) == man["n_cache_files"] and len(names) > 500, len(names)
        assert man["n_cache_files"] <= man["n_llm_calls"], (man["n_cache_files"], man["n_llm_calls"])
        for fn in _random.sample(names, 20):
            d = _j.loads(z.read(fn))
            assert d["model"] == "Qwen/Qwen3-8B-AWQ" and isinstance(d["raw"], str), fn
            assert "prompt_sha" in d and "usage" in d, fn
    if man.get("n_fail"):
        print(f"WARNING: {man['n_fail']} kernel-side failures; "
              f"assembly will retry those keys explicitly")
    print("verify: awq shard cache ok")


PHI4_MODEL = "stelterlab/phi-4-AWQ"
PHI4_REVISION = "075b93fe5ab0d2e86004a5d68c7575ec3bb5a88b"
QWEN14_MODEL = "Qwen/Qwen3-14B-AWQ"
QWEN14_REVISION = "31c69efc29464b6bb0aee1398b5a7b50a99340c3"


def _verify_qwen14probe(dest, job):
    import json as _j
    rep = _j.loads((dest / "probe_report.json").read_text())
    print("qwen14 probe:", {k: v for k, v in rep.items() if k != "sample"})
    assert rep.get("model") == QWEN14_MODEL, rep
    assert rep.get("revision") == QWEN14_REVISION, rep
    if not (rep.get("identical") and rep.get("parses")):
        raise SystemExit("qwen14 probe failed: non-deterministic or unparseable")


def _verify_phi4probe(dest, job):
    import json as _j
    rep = _j.loads((dest / "probe_report.json").read_text())
    print("phi4 probe:", {k: (v if k in ("model", "revision", "identical",
                                         "parses") else v)
                          for k, v in rep.items()})
    assert rep.get("model") == PHI4_MODEL and rep.get("revision") == PHI4_REVISION, rep
    for case in ("c1", "c3"):
        assert rep.get(case, {}).get("parses"), f"phi4 probe {case} unparseable"
    if not rep.get("identical"):
        raise SystemExit("phi4 probe failed: non-deterministic at temperature 0")


def _verify_qwen14beliefs(dest, job):
    import json as _j
    import random as _random
    import zipfile as _zip
    man = _j.loads((dest / "shard_manifest.json").read_text())
    print("shard manifest:", {k: v for k, v in man.items() if k != "failures"})
    assert man["model"] == QWEN14_MODEL, man
    assert man["revision"] == QWEN14_REVISION, man
    assert man["c1_sha"] == "66e6890ba9c5464c" and man["c3_sha"] == "5966cadde90e8236", man
    assert man["tau"] == 0.5, man
    assert man["seal"] == "no qrels attached; guard passed", man
    for f in ("consumers_v3.py", "interpreter.py", "events.py"):
        frozen = _sha(ROOT / "src" / f)[:16]
        assert man["src_snap_sha"][f] == frozen, (f, man["src_snap_sha"][f], frozen)
    print("verify: kernel src_snap byte-identical to frozen src")
    s, e = man["shard"]
    zname = f"cache_shard_{s}_{e}.zip"
    assert zname in job["outputs"], f"manifest shard {man['shard']} != job {job['slug']}"
    with _zip.ZipFile(dest / zname) as z:
        names = z.namelist()
        assert len(names) == man["n_cache_files"] and len(names) > 500, len(names)
        assert man["n_cache_files"] <= man["n_llm_calls"], (man["n_cache_files"], man["n_llm_calls"])
        for fn in _random.sample(names, 20):
            d = _j.loads(z.read(fn))
            assert d["model"] == QWEN14_MODEL and isinstance(d["raw"], str), fn
            assert "prompt_sha" in d and "usage" in d, fn
    if man.get("n_fail"):
        print(f"WARNING: {man['n_fail']} kernel-side failures; "
              f"assembly will retry those keys explicitly")
    print("verify: qwen14 shard cache ok")


def _verify_phi4beliefs(dest, job):
    import json as _j
    import random as _random
    import zipfile as _zip
    man = _j.loads((dest / "shard_manifest.json").read_text())
    print("shard manifest:", {k: v for k, v in man.items() if k != "failures"})
    assert man["model"] == PHI4_MODEL, man
    assert man["revision"] == PHI4_REVISION, man
    assert man["c1_sha"] == "66e6890ba9c5464c" and man["c3_sha"] == "5966cadde90e8236", man
    assert man["tau"] == 0.5, man
    assert man["seal"] == "no qrels attached; guard passed", man
    for f in ("consumers_v3.py", "interpreter.py", "events.py"):
        frozen = _sha(ROOT / "src" / f)[:16]
        assert man["src_snap_sha"][f] == frozen, (f, man["src_snap_sha"][f], frozen)
    print("verify: kernel src_snap byte-identical to frozen src")
    s, e = man["shard"]
    zname = f"cache_shard_{s}_{e}.zip"
    assert zname in job["outputs"], f"manifest shard {man['shard']} != job {job['slug']}"
    with _zip.ZipFile(dest / zname) as z:
        names = z.namelist()
        assert len(names) == man["n_cache_files"] and len(names) > 500, len(names)
        assert man["n_cache_files"] <= man["n_llm_calls"], (man["n_cache_files"], man["n_llm_calls"])
        for fn in _random.sample(names, 20):
            d = _j.loads(z.read(fn))
            assert d["model"] == PHI4_MODEL and isinstance(d["raw"], str), fn
            assert "prompt_sha" in d and "usage" in d, fn
    if man.get("n_fail"):
        print(f"WARNING: {man['n_fail']} kernel-side failures; "
              f"assembly will retry those keys explicitly")
    print("verify: phi4 shard cache ok")


def _verify_loopilot(dest, job, model, revision, scope="c1-loo-dev40-rerank-k3",
                     n_calls=120, inputs_dir="kaggle/inputs_loo",
                     queries_file="queries_dev40.jsonl", sets_file="loo_sets.json",
                     sets_key="dev"):
    import json as _j
    import random as _random
    import zipfile as _zip
    mname = next(n for n in job["outputs"] if n.endswith(".json"))
    man = _j.loads((dest / mname).read_text())
    print("pilot manifest:", {k: v for k, v in man.items() if k != "failures"})
    assert man["model"] == model, man
    assert man["revision"] == revision, man
    assert man["scope"] == scope, man
    assert man["c1_sha"] == "66e6890ba9c5464c", man
    assert man["tau"] == 0.5, man
    assert man["seal"] == "no qrels attached; guard passed", man
    for f in ("consumers_v3.py", "interpreter.py", "events.py"):
        frozen = _sha(ROOT / "src" / f)[:16]
        assert man["src_snap_sha"][f] == frozen, (f, man["src_snap_sha"][f], frozen)
    print("verify: kernel src_snap byte-identical to frozen src")
    assert man["n_llm_calls"] == n_calls, man
    assert man["n_fail"] == 0, man
    # Duplicate-context subtlety (dev-40: 120 invocations -> D=90 keys; test
    # count likewise recomputed): the C1 user string carries no entity_node.
    import hashlib as _hl
    _docs = {}
    for _line in open(ROOT / "data" / "v3" / "corpus.jsonl"):
        _dd = _j.loads(_line)
        _docs[_dd["_id"]] = _dd["text"]
    _spec = _j.loads(open(ROOT / inputs_dir / sets_file).read())
    _qs = {}
    for _line in open(ROOT / inputs_dir / queries_file):
        _qq = _j.loads(_line)
        _qs[_qq["_id"]] = _qq["text"]
    assert set(_qs) == set(_spec[sets_key]) == set(_spec["sets"]), "inputs split mismatch"
    _keys = set()
    for _qid in sorted(_spec["sets"]):
        for _pos in ("drop0", "drop1", "drop2"):
            _user = (f"Operator information need: {_qs[_qid]}\n\nEvidence:\n" +
                     "\n\n".join(f"[DOC {_d}] {_docs[_d]}"
                                 for _d in _spec["sets"][_qid][_pos]))
            _ks = (f"{model}||66e6890ba9c5464c||"
                   f"{_hl.sha256(_user.encode()).hexdigest()[:16]}")
            _keys.add(_hl.sha256(_ks.encode()).hexdigest()[:16] + ".json")
    print(f"distinct-key recompute: D={len(_keys)} ({n_calls} invocations)")
    zname = next(n for n in job["outputs"] if n.endswith(".zip"))
    with _zip.ZipFile(dest / zname) as z:
        names = z.namelist()
        assert len(names) == len(_keys) == man["n_cache_files"], \
            (len(names), len(_keys), man["n_cache_files"])
        for fn in _random.sample(names, 20):
            d = _j.loads(z.read(fn))
            assert d["model"] == model and isinstance(d["raw"], str), fn
            assert d.get("prompt_sha") == "66e6890ba9c5464c", fn
            assert "usage" in d, fn
    print(f"verify: loo pilot cache ok ({model})")


def _verify_loopilot8b(dest, job):
    _verify_loopilot(dest, job, "Qwen/Qwen3-8B-AWQ",
                     "4da05a8edb55c6046cce958586c33b61da07bb79")


def _verify_lootest8b(dest, job):
    _verify_loopilot(dest, job, "Qwen/Qwen3-8B-AWQ",
                     "4da05a8edb55c6046cce958586c33b61da07bb79",
                     scope="c1-loo-test160-rerank-k3", n_calls=480,
                     inputs_dir="kaggle/inputs_loo_test",
                     queries_file="queries_loo.jsonl",
                     sets_file="loo_sets_test.json", sets_key="test")


def _verify_lootest14b(dest, job):
    _verify_loopilot(dest, job, "Qwen/Qwen3-14B-AWQ",
                     "31c69efc29464b6bb0aee1398b5a7b50a99340c3",
                     scope="c1-loo-test160-rerank-k3", n_calls=480,
                     inputs_dir="kaggle/inputs_loo_test",
                     queries_file="queries_loo.jsonl",
                     sets_file="loo_sets_test.json", sets_key="test")


def _verify_loopilot14b(dest, job):
    _verify_loopilot(dest, job, "Qwen/Qwen3-14B-AWQ",
                     "31c69efc29464b6bb0aee1398b5a7b50a99340c3")


def _verify_trackasim(dest, job):
    """Track A CPU-sim shard verification (frozen 2026-09-21).
    Discrete outputs + identifiers: EXACT equality. Floating profits:
    preregistered tolerance |diff| <= 1e-6 AND relative <= 1e-9. Never
    loosened post-hoc. Gate modes: full-overlap (pilot: vs 100-row local
    re-run) or sample-5pct (scale shards: seeded 5% local re-run)."""
    import json as _j
    import pandas as _pd
    ep_name = next(n for n in job["outputs"] if n.endswith(".parquet"))
    mf_name = next(n for n in job["outputs"] if n.endswith(".json"))
    man = _j.loads((dest / mf_name).read_text())
    print("shard manifest:", {k: v for k, v in man.items() if k != "input_hashes"})
    assert man["controller"] == "BeliefBaseStock", man
    assert man["seal"] == "no qrels attached; guard passed", man
    assert man["completed"] == man["total"] > 0, man
    assert man["seeds"] == [60000 + i for i in range(5)], man
    assert man["artifact"] == ep_name, man
    for m, v in (("numpy", "1.26.4"), ("pandas", "2.3.3"), ("pyarrow", "21.0.0"),
                 ("scipy", "1.13.1")):
        assert man["dep_versions"][m] == v, (m, man["dep_versions"])
    spec = _j.loads((ROOT / "kaggle" / "inputs_tracka" / job["shard_spec"]).read_text())
    assert man["config_hash"] == _sha(
        ROOT / "kaggle" / "inputs_tracka" / job["shard_spec"]), man
    assert man["shard_id"] == spec["shard_id"], man
    for f in ("controller_basestock.py", "env.py", "events.py", "metrics.py",
              "interpreter.py", "sim_eval_v3main.py"):
        frozen = _sha(ROOT / "src" / f)[:64]
        assert man["input_hashes"][f] == frozen, (f,)
    df = _pd.read_parquet(dest / ep_name)
    assert len(df) == man["total"], (len(df), man["total"])
    assert set(df["controller"].unique()) == {"BeliefBaseStock"}, df["controller"].unique()
    assert (df["k"] == 3).all() and (df["rung"] == "belief").all()
    # identifier coverage: spec cells x queries x seeds, exactly
    key = ["system", "consumer", "model", "query_id", "seed"]
    assert not df.duplicated(subset=key).any(), "duplicate Kaggle keys"
    want = {(c["system"], c["consumer"], c["model"], q, s)
            for c in spec["cells"] for q in spec["queries"] for s in spec["seeds"]}
    assert set(map(tuple, df[key].astype(str).values.tolist())) == \
        set(map(lambda t: tuple(map(str, t)), want)), "identifier coverage mismatch"
    if job["gate"] == "full-overlap":
        loc = _pd.read_parquet(ROOT / "runs" / "v3main_tracka" / "pilot_local_rerun.parquet")
        assert len(loc) == 100 and not loc.duplicated(subset=key).any(), len(loc)
        ref = loc
    elif job["gate"] == "sample-5pct":
        # Seeded 5% sample (min 50 rows) re-run locally at HEAD; same tolerance.
        import sys as _s
        _s.path.insert(0, str(ROOT))
        _s.path.insert(0, str(ROOT / "tools"))
        from run_controllerB_v3main import _work as _bw
        bel = _pd.concat([_pd.read_parquet(
            ROOT / "results" / "v3main" / f"beliefs_{mm}.parquet")
            for mm in ("Qwen_Qwen3-8B-AWQ", "Qwen_Qwen3-14B-AWQ")],
            ignore_index=True)
        rng = __import__("numpy").random.default_rng(12345)
        idx = rng.choice(len(df), size=max(50, len(df) // 20), replace=False)
        samp = df.iloc[idx]
        rows = []
        for _, r in samp.iterrows():
            b = bel[(bel["system"] == r["system"]) & (bel["consumer"] == r["consumer"]) &
                    (bel["model"] == r["model"]) & (bel["k"] == 3) &
                    (bel["query_id"] == r["query_id"])]
            assert len(b) >= 1, (r["system"], r["consumer"], r["model"], r["query_id"])
            b = b.iloc[0]
            probs = (b["p_normal"], b["p_supplier_delay"], b["p_demand_surge"])
            rows.append(_bw(("belief", r["system"], r["consumer"], r["model"],
                             r["query_id"], r["true_regime"], probs, r["seed"])))
        ref = _pd.DataFrame(rows)
        print(f"sample gate: re-ran {len(ref)} rows locally", flush=True)
    else:
        raise SystemExit(f"unknown gate {job['gate']}")
    mg = df.merge(ref[key + ["profit"]].drop_duplicates(subset=key),
                  on=key, suffixes=("", "_local"))
    assert len(mg) == (len(df) if job["gate"] == "full-overlap" else len(ref)), \
        "identifier mismatch on gate join"
    d = (mg["profit"] - mg["profit_local"]).abs()
    rel = d / mg["profit_local"].abs().clip(lower=1e-12)
    bad = ((d > 1e-6) | (rel > 1e-9)).sum()
    print(f"profit gate ({job['gate']}): n={len(mg)} max_abs={d.max():.2e} "
          f"max_rel={rel.max():.2e} violations={bad}")
    assert bad == 0, f"{bad} tolerance violations (frozen tol: 1e-6 abs + 1e-9 rel)"
    print("verify: tracka sim shard ok")


def _verify_agentickprobe(dest, job):
    import json as _j
    rep = _j.loads((dest / "agentick_probe.json").read_text())
    print("agentick probe:", {k: v for k, v in rep.items() if k != "modes"})
    assert rep.get("commit") == "279fe5f34a35196ba3904550f911d5c8ade3c7c7", rep
    if not rep.get("compatible"):
        raise SystemExit(f"agentick incompatible: {rep.get('reason')}")
    assert set(rep.get("modes", {})) == {"ascii", "language", "state_dict"}, rep
    print("verify: agentick probe compatible")


def cmd_test(job):
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_v3proto.py", "-q"],
                       cwd=str(ROOT))
    raise SystemExit(r.returncode)


def main():
    ap = argparse.ArgumentParser(description="Paper-2 Kaggle remote-compute runner")
    ap.add_argument("job", choices=list(JOBS))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset", choices=["create", "version"])
    g.add_argument("--push", action="store_true")
    g.add_argument("--status", action="store_true")
    g.add_argument("--watch", action="store_true")
    g.add_argument("--fetch", action="store_true")
    g.add_argument("--test", action="store_true")
    g.add_argument("--logs", action="store_true")
    a = ap.parse_args()
    job = JOBS[a.job]
    if a.dataset:
        cmd_dataset(job, create=(a.dataset == "create"))
    elif a.push:
        cmd_push(a.job, job)
    elif a.status:
        cmd_status(job)
    elif a.watch:
        print("terminal state:", cmd_watch(job))
    elif a.fetch:
        cmd_fetch(job)
    elif a.logs:
        cmd_logs(job, out_path=f"/tmp/{a.job}_kernel.log")
    elif a.test:
        cmd_test(job)


if __name__ == "__main__":
    main()
