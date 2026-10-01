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
KaggleClient = None
ApiBlobType = ApiStartBlobUploadRequest = None
ApiCreateDatasetRequest = ApiCreateDatasetVersionRequest = None
ApiCreateDatasetVersionRequestBody = ApiDatasetNewFile = None
ApiDownloadKernelOutputRequest = ApiGetKernelSessionStatusRequest = None
ApiSaveKernelRequest = None


def _require_kaggle_sdk():
    """Load the machine-authenticated SDK only for remote operations.

    Local freeze/quota checks must remain usable without credentials; remote
    operations fail closed at the point where authentication is required.
    """
    global KaggleClient, ApiBlobType, ApiStartBlobUploadRequest
    global ApiCreateDatasetRequest, ApiCreateDatasetVersionRequest
    global ApiCreateDatasetVersionRequestBody, ApiDatasetNewFile
    global ApiDownloadKernelOutputRequest, ApiGetKernelSessionStatusRequest
    global ApiSaveKernelRequest
    if KaggleClient is not None:
        return
    if not TOKEN_PATH.exists():
        raise SystemExit(
            f"missing machine credential {TOKEN_PATH} (ADIA pattern); refusing remote operation"
        )
    token = TOKEN_PATH.read_text().strip()
    if not token:
        raise SystemExit(f"empty machine credential {TOKEN_PATH}; refusing remote operation")
    os.environ.setdefault("KAGGLE_API_TOKEN", token)
    try:
        from kagglesdk import KaggleClient as _KaggleClient
        from kagglesdk.blobs.types.blob_api_service import (
            ApiBlobType as _ApiBlobType, ApiStartBlobUploadRequest as _ApiStartBlobUploadRequest)
        from kagglesdk.datasets.types.dataset_api_service import (
            ApiCreateDatasetRequest as _ApiCreateDatasetRequest,
            ApiCreateDatasetVersionRequest as _ApiCreateDatasetVersionRequest,
            ApiCreateDatasetVersionRequestBody as _ApiCreateDatasetVersionRequestBody,
            ApiDatasetNewFile as _ApiDatasetNewFile)
        from kagglesdk.kernels.types.kernels_api_service import (
            ApiDownloadKernelOutputRequest as _ApiDownloadKernelOutputRequest,
            ApiGetKernelSessionStatusRequest as _ApiGetKernelSessionStatusRequest,
            ApiSaveKernelRequest as _ApiSaveKernelRequest)
    except Exception as exc:
        raise SystemExit(f"Kaggle SDK unavailable: {type(exc).__name__}: {exc}") from exc
    KaggleClient = _KaggleClient
    ApiBlobType = _ApiBlobType
    ApiStartBlobUploadRequest = _ApiStartBlobUploadRequest
    ApiCreateDatasetRequest = _ApiCreateDatasetRequest
    ApiCreateDatasetVersionRequest = _ApiCreateDatasetVersionRequest
    ApiCreateDatasetVersionRequestBody = _ApiCreateDatasetVersionRequestBody
    ApiDatasetNewFile = _ApiDatasetNewFile
    ApiDownloadKernelOutputRequest = _ApiDownloadKernelOutputRequest
    ApiGetKernelSessionStatusRequest = _ApiGetKernelSessionStatusRequest
    ApiSaveKernelRequest = _ApiSaveKernelRequest

OWNER = "rezabarati2"
V5_OWNER = "rezabarati2"
V5_PROTOCOL_HASH = "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022"
V5_KAGGLE_LOCK = ROOT / "versions/sem2act-v5/manifests/kaggle_runtime_lock.json"
V5_KAGGLE_FREEZE = ROOT / "versions/sem2act-v5/manifests/kaggle_runtime_freeze.json"
V5_KAGGLE_PREFLIGHT = ROOT / "versions/sem2act-v5/manifests/kaggle_preflight.json"
V5_PILOT_LOCK = ROOT / "versions/sem2act-v5/manifests/kaggle_reranker_runtime_lock.json"
V5_PILOT_FREEZE = ROOT / "versions/sem2act-v5/manifests/kaggle_reranker_runtime_freeze.json"
V5_PILOT_ACCEPTANCE = ROOT / "versions/sem2act-v5/manifests/kaggle_reranker_pilot_acceptance.json"
V5_PILOT_KERNEL = ROOT / "versions/sem2act-v5/compute/kaggle_kernel/p2_v5_rerank_pilot.py"
V5_PILOT_QUERY_IDS = tuple(f"v5-lb-q{index:04d}" for index in range(1, 21))

JOBS = {
    "v5-canary": {
        "slug": "sem2act-v5-canary",
        "title": "Sem2act v5 canary",
        "script": "versions/sem2act-v5/compute/kaggle_kernel/p2_v5_canary.py",
        "gpu": True,
        "machine_shape": "NvidiaTeslaT4",
        "owner": V5_OWNER,
        "internet": False,
        "dataset_slug": None,
        "dataset_title": None,
        "dataset_dir": None,
        "outputs": ["canary_manifest.json"],
        "dest_dir": "versions/sem2act-v5/runtime/kaggle_canary",
        "verify": "v5canary",
    },
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
    "v4-phi4-probe": {
        "slug": "sem2act-v4-phi4-probe",
        "title": "Sem2act v4 phi4 probe",
        "script": "versions/sem2act-v4/compute/kaggle_kernel/p2_phi4_probe_strict.py",
        "gpu": True,
        "internet": True,
        "dataset_slug": "paper2-v3main-inputs",
        "dataset_title": "paper2-v3main-inputs",
        "dataset_dir": "kaggle/inputs_main",
        "outputs": ["probe_report.json"],
        "dest_dir": "versions/sem2act-v4/runtime/model_family/kaggle_out",
        "verify": "v4phi4probe",
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
    "v4-phi4-b0": {
        "slug": "sem2act-v4-phi4-b0",
        "title": "Sem2act v4 phi4 b0",
        "script": "kaggle_kernel/p2_phi4_beliefs.py",
        "gpu": True,
        "internet": True,
        "dataset_slug": "sem2act-v4-phi4-inputs",
        "dataset_title": "Sem2act v4 Phi-4 inputs",
        "dataset_dir": "kaggle/inputs_phi4",
        "outputs": ["cache_shard_0_50.zip", "shard_manifest.json"],
        "dest_dir": "versions/sem2act-v4/runtime/model_family/shards",
        "verify": "v4phi4beliefs",
        "expected_range": [0, 50],
        "expected_n_queries": 50,
        "expected_n_llm_calls": 2100,
        "install_names": {"shard_manifest.json": "shard_manifest_b0.json"},
        "fetch_manifest_name": "kaggle_fetch_manifest_b0.json",
    },
    "v4-phi4-b1": {
        "slug": "sem2act-v4-phi4-b1",
        "title": "Sem2act v4 phi4 b1",
        "script": "kaggle_kernel/p2_phi4_beliefs.py",
        "gpu": True,
        "internet": True,
        "dataset_slug": "sem2act-v4-phi4-inputs",
        "dataset_title": "Sem2act v4 Phi-4 inputs",
        "dataset_dir": "kaggle/inputs_phi4",
        "outputs": ["cache_shard_50_100.zip", "shard_manifest.json"],
        "dest_dir": "versions/sem2act-v4/runtime/model_family/shards",
        "verify": "v4phi4beliefs",
        "expected_range": [50, 100],
        "expected_n_queries": 50,
        "expected_n_llm_calls": 2100,
        "install_names": {"shard_manifest.json": "shard_manifest_b1.json"},
        "fetch_manifest_name": "kaggle_fetch_manifest_b1.json",
    },
    "v4-phi4-b2": {
        "slug": "sem2act-v4-phi4-b2",
        "title": "Sem2act v4 phi4 b2",
        "script": "kaggle_kernel/p2_phi4_beliefs.py",
        "gpu": True,
        "internet": True,
        "dataset_slug": "sem2act-v4-phi4-inputs",
        "dataset_title": "Sem2act v4 Phi-4 inputs",
        "dataset_dir": "kaggle/inputs_phi4",
        "outputs": ["cache_shard_100_150.zip", "shard_manifest.json"],
        "dest_dir": "versions/sem2act-v4/runtime/model_family/shards",
        "verify": "v4phi4beliefs",
        "expected_range": [100, 150],
        "expected_n_queries": 50,
        "expected_n_llm_calls": 2100,
        "install_names": {"shard_manifest.json": "shard_manifest_b2.json"},
        "fetch_manifest_name": "kaggle_fetch_manifest_b2.json",
    },
    "v4-phi4-b3": {
        "slug": "sem2act-v4-phi4-b3",
        "title": "Sem2act v4 phi4 b3",
        "script": "kaggle_kernel/p2_phi4_beliefs.py",
        "gpu": True,
        "internet": True,
        "dataset_slug": "sem2act-v4-phi4-inputs",
        "dataset_title": "Sem2act v4 Phi-4 inputs",
        "dataset_dir": "kaggle/inputs_phi4",
        "outputs": ["cache_shard_150_200.zip", "shard_manifest.json"],
        "dest_dir": "versions/sem2act-v4/runtime/model_family/shards",
        "verify": "v4phi4beliefs",
        "expected_range": [150, 200],
        "expected_n_queries": 50,
        "expected_n_llm_calls": 2100,
        "install_names": {"shard_manifest.json": "shard_manifest_b3.json"},
        "fetch_manifest_name": "kaggle_fetch_manifest_b3.json",
    },
    "v4-qrel-calibration": {
        "slug": "sem2act-v4-qrel-calibration",
        "title": "Sem2act v4 qrel calibration",
        "script": "versions/sem2act-v4/compute/kaggle_kernel/p2_qrel_judge.py",
        "gpu": True,
        "internet": True,
        "dataset_slug": "sem2act-v4-qrel-calibration-inputs",
        "dataset_title": "Sem2act v4 QREL calibration inputs",
        "dataset_dir": "kaggle/inputs_qrel_calibration",
        "outputs": [
            "raw_outputs.jsonl", "judge_labels.jsonl", "RUN_MANIFEST.json",
            "calibration_result.json", "runtime_versions.json",
            "qrel_kernel_manifest.json",
        ],
        "dest_dir": "versions/sem2act-v4/runtime/qrel_validation_v2/calibration_judge",
        "verify": "qrelcalibration",
    },
    "v4-qrel-sem2act": {
        "slug": "sem2act-v4-qrel-sem2act",
        "title": "Sem2act v4 qrel sem2act judge",
        "script": "versions/sem2act-v4/compute/kaggle_kernel/p2_qrel_judge.py",
        "gpu": True,
        "internet": True,
        "dataset_slug": "sem2act-v4-qrel-sem2act-inputs",
        "dataset_title": "Sem2act v4 QREL Sem2Act inputs",
        "dataset_dir": "kaggle/inputs_qrel_sem2act",
        "outputs": [
            "raw_outputs.jsonl", "judge_labels.jsonl", "RUN_MANIFEST.json",
            "runtime_versions.json", "qrel_kernel_manifest.json",
        ],
        "dest_dir": "versions/sem2act-v4/runtime/qrel_validation_v2/sem2act_judge",
        "verify": "qrelsem2act",
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
    "reasoning-smoke": {
        "slug": "paper2-reasoning-smoke",
        "title": "Paper2 reasoning smoke",
        "script": "kaggle_kernel/p2_reasoning_smoke.py",
        "gpu": True,
        "internet": True,
        "dataset_slug": "paper2-reasoning-inputs",
        "dataset_title": "paper2-reasoning-inputs",
        "dataset_dir": "kaggle/inputs_reasoning",
        "outputs": ["cache_reasoning_smoke.zip",
                    "smoke_manifest.json"],
        "dest_dir": "runs/reasoning_agentic",
        "verify": "reasoningsmoke",
    },
    "reasoning-devpilot": {
        "slug": "paper2-reasoning-devpilot",
        "title": "Paper2 reasoning devpilot",
        "script": "kaggle_kernel/p2_reasoning_smoke.py",
        "gpu": True,
        "internet": True,
        "dataset_slug": "paper2-reasoning-dev2-inputs",
        "dataset_title": "paper2-reasoning-dev2-inputs",
        "dataset_dir": "kaggle/inputs_reasoning_dev2",
        "outputs": ["cache_reasoning_smoke.zip",
                    "smoke_manifest.json"],
        "dest_dir": "runs/reasoning_agentic_dev2",
        "verify": "reasoningsmoke",
    },
    "reasoning-final": {
        "slug": "paper2-reasoning-final",
        "title": "Paper2 reasoning final",
        "script": "kaggle_kernel/p2_reasoning_smoke.py",
        "gpu": True,
        "internet": True,
        "dataset_slug": "paper2-reasoning-final-inputs",
        "dataset_title": "paper2-reasoning-final-inputs",
        "dataset_dir": "kaggle/inputs_reasoning_final",
        "outputs": ["cache_reasoning_smoke.zip",
                    "smoke_manifest.json"],
        "dest_dir": "runs/reasoning_agentic_final",
        "verify": "reasoningfinal",
    },
    "v5-rerank-pilot": {
        "slug": "sem2act-v5-rerank-pilot",
        "title": "Sem2act v5 rerank pilot",
        "script": "versions/sem2act-v5/compute/kaggle_kernel/p2_v5_rerank_pilot.py",
        "gpu": True,
        "machine_shape": "NvidiaTeslaT4",
        "owner": V5_OWNER,
        "internet": True,
        "dataset_slug": "sem2act-v5-rerank-inputs",
        "dataset_title": "Sem2act v5 rerank inputs",
        "dataset_dir": "versions/sem2act-v5/runtime/kaggle_rerank_inputs",
        # The pilot pushes against the already remote-verified input dataset
        # v1, whose exact bytes are pinned in the upload manifest, so it does
        # not require the gitignored local copy to be staged just to push.
        "require_local_dataset_stage": False,
        "upload_manifest": "versions/sem2act-v5/manifests/upload/v5-rerank-inputs.json",
        "outputs": ["rerank.trec", "run_manifest.json", "runtime_versions.json"],
        "dest_dir": "versions/sem2act-v5/runtime/kaggle_rerank_pilot",
        "verify": "v5rerankpilot",
        "install_run_manifest": True,
    },
    "v5-rerank": {
        "slug": "sem2act-v5-rerank",
        "title": "Sem2act v5 rerank",
        "script": "versions/sem2act-v5/compute/kaggle_kernel/p2_v5_rerank.py",
        "gpu": True,
        "machine_shape": "NvidiaTeslaT4",
        "owner": V5_OWNER,
        "internet": True,
        "dataset_slug": "sem2act-v5-rerank-inputs",
        "dataset_title": "Sem2act v5 rerank inputs",
        "dataset_dir": "versions/sem2act-v5/runtime/kaggle_rerank_inputs",
        # Same already remote-verified dataset as the pilot; the local copy is
        # only needed to (re)upload, not to push a kernel against it.
        "require_local_dataset_stage": False,
        "upload_manifest": "versions/sem2act-v5/manifests/upload/v5-rerank-inputs.json",
        "outputs": ["rerank.trec", "run_manifest.json", "runtime_versions.json"],
        "dest_dir": "versions/sem2act-v5/runtime/retrieval_bundle",
        "verify": "v5rerank",
        "install_run_manifest": True,
    },
    "v5-qwen": {
        "slug": "sem2act-v5-qwen",
        "title": "Sem2act v5 Qwen",
        "script": "versions/sem2act-v5/compute/kaggle_kernel/p2_v5_qwen_consumer.py",
        "gpu": True,
        "machine_shape": "NvidiaTeslaT4",
        "owner": V5_OWNER,
        "internet": True,
        "dataset_slug": "sem2act-v5-qwen-inputs",
        "dataset_title": "Sem2act v5 Qwen inputs",
        "dataset_dir": "versions/sem2act-v5/runtime/kaggle_qwen_inputs",
        "outputs": ["raw_outputs.jsonl", "beliefs.jsonl", "run_manifest.json",
                    "failures.json", "runtime_versions.json"],
        "dest_dir": "versions/sem2act-v5/runtime/model_outputs/qwen",
        "verify": "v5qwen",
        "install_run_manifest": True,
    },
    "v5-llama": {
        "slug": "sem2act-v5-llama",
        "title": "Sem2act v5 Llama consumer",
        "script": "versions/sem2act-v5/compute/kaggle_kernel/p2_v5_consumer.py",
        "gpu": True,
        "machine_shape": "NvidiaTeslaT4",
        "owner": V5_OWNER,
        "internet": True,
        "dataset_slug": "sem2act-v5-llama-inputs",
        "dataset_title": "Sem2act v5 Llama inputs",
        "dataset_dir": "versions/sem2act-v5/runtime/kaggle_llama_inputs",
        "outputs": ["raw_outputs.jsonl", "beliefs.jsonl", "run_manifest.json",
                    "failures.json", "runtime_versions.json"],
        "dest_dir": "versions/sem2act-v5/runtime/model_outputs/llama",
        "verify": "v5consumer",
        "install_run_manifest": True,
    },
    "v5-mistral": {
        "slug": "sem2act-v5-mistral",
        "title": "Sem2act v5 Mistral",
        "script": "versions/sem2act-v5/compute/kaggle_kernel/p2_v5_consumer.py",
        "gpu": True,
        "machine_shape": "NvidiaTeslaT4",
        "owner": V5_OWNER,
        "internet": True,
        "dataset_slug": "sem2act-v5-mistral-inputs",
        "dataset_title": "Sem2act v5 Mistral inputs",
        "dataset_dir": "versions/sem2act-v5/runtime/kaggle_mistral_inputs",
        "outputs": ["raw_outputs.jsonl", "beliefs.jsonl", "run_manifest.json",
                    "failures.json", "runtime_versions.json"],
        "dest_dir": "versions/sem2act-v5/runtime/model_outputs/mistral",
        "verify": "v5consumer",
        "install_run_manifest": True,
    },
}

POLL_SECONDS = 120
WATCH_TIMEOUT_SECONDS = 6 * 3600


def _slugify(title):
    import re
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", title.lower())).strip("-")


def _lint_kernel(path):
    import shutil
    pyflakes = shutil.which("pyflakes")
    if pyflakes:
        r = subprocess.run([pyflakes, str(path)], capture_output=True, text=True)
        bad = [l for l in (r.stdout + r.stderr).splitlines()
               if l.strip() and "imported but unused" not in l
               and "unable to detect undefined names" not in l
               and "redefinition of unused" not in l]
        if bad:
            raise SystemExit("lint failed, not pushing:\n  " + "\n  ".join(bad))
        return
    try:
        compile(path.read_text(), str(path), "exec")
    except SyntaxError as exc:
        raise SystemExit(f"syntax lint failed, not pushing: {exc}") from exc


# --- datasets ---------------------------------------------------------------

def _stage_files(job):
    d = ROOT / job["dataset_dir"]
    files = sorted(p for p in d.rglob("*") if p.is_file())
    if not files:
        raise SystemExit(f"nothing staged in {d}")
    return files


def _job_owner(job):
    return job.get("owner", OWNER)


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


def _v5_upload_manifest_path(job):
    # Several jobs share one uploaded dataset, so the verified manifest may be
    # named explicitly instead of being derived from the job slug.
    explicit = job.get("upload_manifest")
    if explicit:
        return ROOT / explicit
    return ROOT / "versions/sem2act-v5/manifests/upload" / f"{job['slug']}-inputs.json"


def _local_upload_record(job, files):
    owner = _job_owner(job)
    return {
        "schema_version": 1,
        "status": "upload-pending",
        "job": job["slug"],
        "dataset": f"{owner}/{job['dataset_slug']}",
        "protocol_hash": V5_PROTOCOL_HASH,
        "qrels_uploaded": False,
        "local_files": {
            path.name: {"sha256": _sha(path), "bytes": path.stat().st_size}
            for path in files
        },
        "files": {path.name: _sha(path) for path in files},
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _require_v5_dataset_verified(job):
    manifest_path = _v5_upload_manifest_path(job)
    if not manifest_path.exists():
        raise SystemExit(f"missing verified Kaggle upload manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "remote-verified":
        raise SystemExit("Kaggle dataset upload is not remote-verified")
    if manifest.get("protocol_hash") != V5_PROTOCOL_HASH:
        raise SystemExit("Kaggle dataset manifest protocol hash drift")
    if manifest.get("qrels_uploaded") is not False:
        raise SystemExit("Kaggle dataset manifest does not prove qrels_uploaded=false")
    expected_dataset = f"{_job_owner(job)}/{job['dataset_slug']}"
    if manifest.get("dataset") != expected_dataset:
        raise SystemExit("Kaggle dataset owner drift")
    # The remote-verified manifest pins the exact dataset bytes, so a job may
    # rely on it alone. Jobs that also (re)stage the dataset locally (upload
    # path, or an operator-managed local copy) keep the stricter check that
    # the local files match the pinned remote hashes.
    if job.get("require_local_dataset_stage", True):
        files = _stage_files(job)
        expected = {path.name: _sha(path) for path in files}
        if manifest.get("files") != expected:
            raise SystemExit("Kaggle dataset local file manifest drift")
    else:
        expected = manifest.get("files")
        if not isinstance(expected, dict) or not expected:
            raise SystemExit("Kaggle dataset manifest does not pin any file hashes")
    if manifest.get("remote_file_list_match") is not True or manifest.get("remote_hash_match") is not True:
        raise SystemExit("Kaggle dataset remote file/hash verification failed")
    if manifest.get("remote_sha256") != expected:
        raise SystemExit("Kaggle dataset remote SHA-256 drift")
    return manifest


def cmd_dataset(job, create):
    _require_kaggle_sdk()
    if not job.get("dataset_slug") or not job.get("dataset_dir"):
        raise SystemExit("this job has no input dataset")
    files = _stage_files(job)
    upload_manifest_path = _v5_upload_manifest_path(job) if job["slug"].startswith("sem2act-v5-") else None
    if upload_manifest_path is not None:
        upload_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        upload_manifest_path.write_text(json.dumps(_local_upload_record(job, files), indent=2, sort_keys=True) + "\n")
    owner = _job_owner(job)
    print(f"uploading {len(files)} files as PRIVATE dataset "
          f"{owner}/{job['dataset_slug']}")
    with KaggleClient() as client:
        tokens = [_upload_blob(client, p) for p in files]
        new_files = []
        for t in tokens:
            f = ApiDatasetNewFile()
            f.token = t
            new_files.append(f)
        if create:
            req = ApiCreateDatasetRequest()
            req.owner_slug = owner
            req.slug = job["dataset_slug"]
            req.title = job["dataset_title"]
            req.license_name = "other"
            req.is_private = True
            req.files = new_files
            resp = client.datasets.dataset_api_client.create_dataset(req)
        else:
            req = ApiCreateDatasetVersionRequest()
            req.owner_slug = owner
            req.dataset_slug = job["dataset_slug"]
            body = ApiCreateDatasetVersionRequestBody()
            body.version_notes = f"paper2 {job['dataset_slug']} refresh"
            body.files = new_files
            req.body = body
            resp = client.datasets.dataset_api_client.create_dataset_version(req)
    record_dir = ROOT / "versions/sem2act-v5/manifests/dataset_versions"
    if job["slug"].startswith("sem2act-v5-"):
        record_dir.mkdir(parents=True, exist_ok=True)
        fields = {}
        for name in ("dataset_version_number", "version_number", "ref", "slug",
                     "status", "message", "error"):
            value = getattr(resp, name, None)
            if value is not None:
                fields[name] = str(value)
        (record_dir / f"{job['slug']}.json").write_text(json.dumps({
            "schema_version": 1,
            "job": job["slug"],
            "dataset": f"{owner}/{job['dataset_slug']}",
            "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "response_fields": fields,
            "response_repr": str(resp),
        }, indent=2, sort_keys=True) + "\n")
    print("response:", resp)
    if upload_manifest_path is not None:
        manifest = json.loads(upload_manifest_path.read_text())
        response_fields = {}
        for name in ("dataset_version_number", "version_number", "ref", "slug", "status", "message", "error"):
            value = getattr(resp, name, None)
            if value is not None:
                response_fields[name] = str(value)
        manifest.update({
            "status": "uploaded-unverified",
            "response_fields": response_fields,
            "uploaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "verification_command": (
                "python3 scripts/verify_kaggle_dataset.py "
                f"--manifest {upload_manifest_path.relative_to(ROOT)} "
                f"--source-dir {job['dataset_dir']} --owner {owner} "
                f"--slug {job['dataset_slug']} --version <DATASET_VERSION>"
            ),
        })
        upload_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


# --- kernels ----------------------------------------------------------------

def cmd_push(job_name, job):
    _require_kaggle_sdk()
    if job_name.startswith("v5-") and job.get("gpu"):
        _require_v5_kaggle_submission_gate(job_name)
    slug, title = job["slug"], job["title"]
    if _slugify(title) != slug:
        raise SystemExit(f"title {title!r} slugifies to {_slugify(title)!r}, not {slug!r}")
    if job_name == "v4-qrel-sem2act":
        calibration_freeze = ROOT / "versions/sem2act-v4/manifests/qrel_validation_v2_calibration_freeze.json"
        calibration_result = ROOT / "versions/sem2act-v4/results/qrel_validation_v2/calibration_result.json"
        submission_marker = ROOT / "versions/sem2act-v4/manifests/qrel_validation_v2_sem2act_submission.json"
        if not calibration_freeze.is_file() or not calibration_result.is_file():
            raise SystemExit("refusing Sem2Act judge: frozen calibration result is missing")
        if submission_marker.exists():
            raise SystemExit("refusing Sem2Act judge: the one permitted submission is already recorded")
        gate = json.loads(calibration_freeze.read_text())
        result = json.loads(calibration_result.read_text())
        if gate.get("status") != "frozen" or gate.get("sem2act_judge_permitted") is not True:
            raise SystemExit("refusing Sem2Act judge: calibration has not produced a frozen passing gate")
        if result.get("status") != "pass":
            raise SystemExit("refusing Sem2Act judge: calibration result is not pass")
        actual = hashlib.sha256(calibration_result.read_bytes()).hexdigest()
        if gate.get("calibration_result_sha256") != actual:
            raise SystemExit("refusing Sem2Act judge: calibration freeze/result hash mismatch")
    path = ROOT / job["script"]
    _lint_kernel(path)
    req = ApiSaveKernelRequest()
    owner = _job_owner(job)
    req.slug = f"{owner}/{slug}"
    req.new_title = title
    req.text = path.read_text()
    req.language = "python"
    req.kernel_type = "script"
    req.is_private = True
    req.enable_gpu = job["gpu"]
    if job.get("machine_shape"):
        req.machine_shape = job["machine_shape"]
    req.enable_internet = job["internet"]
    req.dataset_data_sources = ([f"{owner}/{job['dataset_slug']}"]
                                if job.get("dataset_slug") else [])
    req.category_ids = []
    with KaggleClient() as c:
        resp = c.kernels.kernels_api_client.save_kernel(req)
    print(job_name, "->", resp)
    error = getattr(resp, "error", "") or getattr(resp, "_error", "")
    if error:
        raise RuntimeError(f"Kaggle rejected {job_name}: {error}")
    if job_name == "v4-qrel-sem2act":
        submission_marker = ROOT / "versions/sem2act-v4/manifests/qrel_validation_v2_sem2act_submission.json"
        calibration_freeze = ROOT / "versions/sem2act-v4/manifests/qrel_validation_v2_calibration_freeze.json"
        submission_marker.write_text(json.dumps({
            "schema_version": 1,
            "status": "submitted-once",
            "submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "job": job["slug"],
            "calibration_freeze_sha256": _sha(calibration_freeze),
        }, indent=2, sort_keys=True) + "\n")


def cmd_status(job):
    _require_kaggle_sdk()
    req = ApiGetKernelSessionStatusRequest()
    req.user_name, req.kernel_slug = _job_owner(job), job["slug"]
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
    _require_kaggle_sdk()
    req = ApiDownloadKernelOutputRequest()
    req.owner_slug, req.kernel_slug = _job_owner(job), job["slug"]
    dest = ROOT / "artifacts" / "kaggle_out" / job["slug"]
    dest.mkdir(parents=True, exist_ok=True)
    data = None
    for attempt in range(1, 6):
        try:
            with KaggleClient() as c:
                resp = c.kernels.kernels_api_client.download_kernel_output(req)
                data = resp.content if hasattr(resp, "content") else None
            if data:
                break
            raise RuntimeError(f"empty output response {getattr(resp, 'status_code', '?')}")
        except Exception as exc:
            if attempt == 5:
                raise
            print(f"download attempt {attempt} failed ({type(exc).__name__}); retrying", flush=True)
            time.sleep(15)
    if not data:
        raise SystemExit(f"empty output response {getattr(resp, 'status_code', '?')}")
    zpath = dest / "output.zip"
    zpath.write_bytes(data)
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        z.extractall(dest)
        print(f"extracted {len(names)} files; expected outputs: {job['outputs']}")
    # verify: every expected output present + job-specific re-verification
    for name in job["outputs"]:
        if not (dest / name).exists():
            raise SystemExit(f"missing expected output {name} in kernel outputs")
    verifier = {"v5canary": _verify_v5_canary, "v5rerank": _verify_v5_rerank, "v5qwen": _verify_v5_qwen,
                "v5rerankpilot": _verify_v5_rerank_pilot, "v5consumer": _verify_v5_consumer,
     "rerank": _verify_rerank, "encode2b": _verify_encode2b,
     "refine": _verify_refine, "runsmain": _verify_runsmain,
     "probe": _verify_probe, "awqbeliefs": _verify_awqbeliefs,
     "phi4probe": _verify_phi4probe, "v4phi4probe": _verify_v4phi4probe,
     "phi4beliefs": _verify_phi4beliefs, "v4phi4beliefs": _verify_v4phi4beliefs,
      "qrelcalibration": _verify_qrel_calibration,
      "qrelsem2act": _verify_qrel_sem2act,
      "qwen14probe": _verify_qwen14probe,
      "qwen14beliefs": _verify_qwen14beliefs,
      "loopilot8b": _verify_loopilot8b,
      "loopilot14b": _verify_loopilot14b,
      "lootest8b": _verify_lootest8b,
      "lootest14b": _verify_lootest14b,
      "trackasim": _verify_trackasim,
      "agentickprobe": _verify_agentickprobe,
      "reasoningsmoke": _verify_reasoningsmoke,
      "reasoningfinal": _verify_reasoningfinal}[job["verify"]]
    verification = verifier(dest, job)
    # install into runs/ (only after verification passes)
    (ROOT / job["dest_dir"]).mkdir(parents=True, exist_ok=True)
    install_names = job.get("install_names", {})
    for name in job["outputs"]:
        if name == "run_manifest.json" and not job.get("install_run_manifest"):
            continue
        target_name = install_names.get(name, name)
        target = ROOT / job["dest_dir"] / target_name
        target.write_bytes((dest / name).read_bytes())
        print(f"installed {target} sha={_sha(target)[:12]}")
    fetch_manifest_name = job.get("fetch_manifest_name", "kaggle_fetch_manifest.json")
    (ROOT / job["dest_dir"] / fetch_manifest_name).write_text(json.dumps({
        "job": job["slug"], "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": {n: _sha(ROOT / job["dest_dir"] / install_names.get(n, n))
                  for n in job["outputs"] if n != "run_manifest.json"},
        "verification": verification or {},
    }, indent=2))
    if job.get("verify") == "v5canary":
        record = _read_v5_preflight()
        record.update({
            "status": "canary-passed",
            "canary_job": job["slug"],
            "canary_output_sha256": _sha(dest / "canary_manifest.json"),
            "canary_verification": verification,
            "canary_fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        V5_KAGGLE_PREFLIGHT.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    if job.get("verify") == "v5rerankpilot":
        # A verified pilot is only a candidate for acceptance.  Codex writes the
        # acceptance record; until then the full reranker stays locked.
        pending = ROOT / job["dest_dir"] / "pilot_pending_acceptance.json"
        pending.write_text(json.dumps({
            "schema_version": 1,
            "status": "verified-awaiting-codex-acceptance",
            "job": job["slug"],
            "protocol_hash": V5_PROTOCOL_HASH,
            "runtime_lock_sha256": _sha(V5_PILOT_LOCK),
            "runtime_freeze_sha256": _sha(V5_PILOT_FREEZE),
            "rerank_trec_sha256": _sha(ROOT / job["dest_dir"] / "rerank.trec"),
            "n_queries": verification["n_queries"],
            "n_scores": verification["n_scores"],
            "n_fail": verification["n_fail"],
            "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "full_reranker_authorized": False,
            "next_action": "Codex reviews the pilot evidence and, if acceptable, writes "
                           f"{V5_PILOT_ACCEPTANCE.relative_to(ROOT)}",
        }, indent=2, sort_keys=True) + "\n")


def _verify_v5_canary(dest, job):
    manifest = json.loads((dest / "canary_manifest.json").read_text())
    if manifest.get("status") != "pass":
        raise SystemExit("Kaggle canary did not pass")
    if manifest.get("protocol_hash") != V5_PROTOCOL_HASH:
        raise SystemExit("Kaggle canary protocol hash drift")
    if manifest.get("amendment_id") != "sem2act-v5-kaggle-backend-v3":
        raise SystemExit("Kaggle canary amendment drift")
    if manifest.get("torch") != "2.10.0+cu128" or manifest.get("cuda") != "12.8":
        raise SystemExit("Kaggle canary Torch/CUDA image gate failed")
    if (manifest.get("device_count") != 1
            or manifest.get("physical_device_count") != 2
            or "t4" not in manifest.get("device_name", "").lower()):
        raise SystemExit("Kaggle canary device gate failed")
    if not manifest.get("real_cuda_matmul"):
        raise SystemExit("Kaggle canary did not run a CUDA matmul")
    return {"status": "pass", "device": manifest["device_name"], "torch": manifest["torch"], "cuda": manifest["cuda"]}


def _read_v5_preflight():
    if not V5_KAGGLE_PREFLIGHT.exists():
        raise SystemExit("run the Kaggle quota preflight before v5 submission")
    if not V5_KAGGLE_LOCK.exists() or not V5_KAGGLE_FREEZE.exists():
        raise SystemExit("Kaggle runtime freeze is missing")
    record = json.loads(V5_KAGGLE_PREFLIGHT.read_text())
    if record.get("protocol_hash") != V5_PROTOCOL_HASH:
        raise SystemExit("Kaggle preflight protocol hash drift")
    if record.get("runtime_freeze_sha256") != _sha(V5_KAGGLE_FREEZE):
        raise SystemExit("Kaggle preflight runtime freeze hash mismatch")
    if record.get("runtime_lock_sha256") != _sha(V5_KAGGLE_LOCK):
        raise SystemExit("Kaggle preflight runtime lock hash mismatch")
    if record.get("owner") != V5_OWNER or record.get("owner_verified") is not True:
        raise SystemExit("Kaggle authenticated owner was not verified as the frozen V5 owner")
    return record


def _require_v5_pilot_freeze():
    if not V5_PILOT_LOCK.is_file() or not V5_PILOT_FREEZE.is_file():
        raise SystemExit("Kaggle reranker pilot lock/freeze is missing")
    freeze = json.loads(V5_PILOT_FREEZE.read_text())
    if freeze.get("protocol_hash") != V5_PROTOCOL_HASH:
        raise SystemExit("Kaggle reranker pilot freeze protocol hash drift")
    if freeze.get("lock_sha256") != _sha(V5_PILOT_LOCK):
        raise SystemExit("Kaggle reranker pilot lock hash does not match its freeze")
    if freeze.get("pilot_kernel_sha256") != _sha(V5_PILOT_KERNEL):
        raise SystemExit("Kaggle reranker pilot kernel changed after the freeze")
    if freeze.get("authorized_pilot_queries") != 20 or freeze.get("authorized_pilot_scores") != 1000:
        raise SystemExit("Kaggle reranker pilot freeze does not authorize exactly 20 queries / 1,000 scores")
    if freeze.get("full_reranker_authorized") is not False:
        raise SystemExit("Kaggle reranker pilot freeze must not authorize the full reranker")
    return freeze


def _require_v5_full_reranker_acceptance(job_name):
    """The full 240-query reranker stays locked until Codex accepts the pilot."""
    if job_name != "v5-rerank":
        return
    if not V5_PILOT_ACCEPTANCE.is_file():
        raise SystemExit(
            "v5-rerank is not authorized: Codex acceptance of the 20-query Kaggle pilot is required"
        )
    acceptance = json.loads(V5_PILOT_ACCEPTANCE.read_text())
    if acceptance.get("status") != "accepted":
        raise SystemExit("Kaggle reranker pilot acceptance record is not accepted")
    if acceptance.get("protocol_hash") != V5_PROTOCOL_HASH:
        raise SystemExit("Kaggle reranker pilot acceptance protocol hash drift")
    if acceptance.get("job") != JOBS["v5-rerank-pilot"]["slug"]:
        raise SystemExit("Kaggle reranker pilot acceptance names the wrong job")
    freeze = _require_v5_pilot_freeze()
    if acceptance.get("lock_id") != freeze.get("lock_id"):
        raise SystemExit("Kaggle reranker pilot acceptance names a different lock")
    if acceptance.get("runtime_lock_sha256") != _sha(V5_PILOT_LOCK):
        raise SystemExit("Kaggle reranker pilot acceptance lock hash drift")
    if acceptance.get("runtime_freeze_sha256") != _sha(V5_PILOT_FREEZE):
        raise SystemExit("Kaggle reranker pilot acceptance freeze hash drift")
    if acceptance.get("rerank_trec_sha256") != _sha(
            ROOT / JOBS["v5-rerank-pilot"]["dest_dir"] / "rerank.trec"):
        raise SystemExit("accepted pilot rerank.trec is missing or has drifted since acceptance")
    if acceptance.get("n_queries") != 20 or acceptance.get("n_scores") != 1000:
        raise SystemExit("accepted pilot coverage drift")


def _require_v5_kaggle_submission_gate(job_name):
    if os.environ.get("SEM2ACT_V5_QUOTA_CLI_CONFIRMED") != "1":
        raise SystemExit("set SEM2ACT_V5_QUOTA_CLI_CONFIRMED=1 only after quota preflight passes")
    record = _read_v5_preflight()
    allowed = {"quota-passed"} if job_name == "v5-canary" else {"canary-passed"}
    if record.get("status") not in allowed:
        raise SystemExit(f"Kaggle v5 gate requires {sorted(allowed)}, got {record.get('status')!r}")
    if record.get("gpu_remaining_hours", 0) <= 0:
        raise SystemExit("Kaggle GPU quota is not positive")
    if job_name != "v5-canary":
        _require_v5_dataset_verified(JOBS[job_name])
    if job_name in {"v5-rerank-pilot", "v5-rerank"}:
        _require_v5_pilot_freeze()
        _require_v5_full_reranker_acceptance(job_name)


def _kaggle_cli_path():
    import shutil
    candidates = [
        os.environ.get("SEM2ACT_KAGGLE_CLI"),
        "/private/tmp/kaggle-cli-v2/bin/kaggle",
        str(ROOT / ".venv-kaggle/bin/kaggle"),
        str(pathlib.Path.home() / ".venv-kaggle/bin/kaggle"),
        shutil.which("kaggle"),
    ]
    for candidate in candidates:
        if candidate and pathlib.Path(candidate).is_file():
            return candidate
    raise SystemExit("Kaggle CLI is unavailable")


def cmd_v5_preflight():
    quota_cmd = _kaggle_cli_path()
    owner_proc = subprocess.run(
        [quota_cmd, "datasets", "list", "--mine", "--search", "sem2act-v5-rerank-inputs", "--format", "json"],
        capture_output=True, text=True,
    )
    if owner_proc.returncode != 0:
        raise SystemExit("Kaggle account identity probe failed; refusing V5 preflight")
    try:
        owner_payload = json.loads(owner_proc.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Kaggle account identity output was not JSON: {exc}") from exc
    owner_rows = owner_payload if isinstance(owner_payload, list) else owner_payload.get(
        "datasets", owner_payload.get("items", owner_payload.get("data", []))
    )
    owned_refs = set()
    for row in owner_rows:
        if not isinstance(row, dict):
            continue
        ref = str(row.get("ref", row.get("datasetRef", ""))).strip("/")
        if ref.startswith("datasets/"):
            ref = ref[len("datasets/"):]
        owned_refs.add(ref)
    expected_input = f"{V5_OWNER}/sem2act-v5-rerank-inputs"
    if expected_input not in owned_refs:
        raise SystemExit(
            f"authenticated Kaggle account does not own the frozen V5 input dataset {expected_input}"
        )
    proc = subprocess.run([quota_cmd, "quota", "--format", "json"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        record = {
            "schema_version": 1, "status": "unavailable",
            "protocol_hash": V5_PROTOCOL_HASH,
            "owner": V5_OWNER,
            "owner_verified": True,
            "identity_probe": "kaggle datasets list --mine --search sem2act-v5-rerank-inputs --format json",
            "runtime_lock_sha256": _sha(V5_KAGGLE_LOCK),
            "runtime_freeze_sha256": _sha(V5_KAGGLE_FREEZE),
            "error": "Kaggle quota command failed before a result was returned",
            "exit_code": proc.returncode,
        }
        V5_KAGGLE_PREFLIGHT.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        raise SystemExit(record["error"])
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Kaggle quota output was not JSON: {exc}")
    rows = payload if isinstance(payload, list) else payload.get("quota", [])
    gpu = next((row for row in rows if str(row.get("resource", "")).upper() == "GPU"), None)
    if gpu is None:
        raise SystemExit("Kaggle quota response has no GPU row")
    raw = str(gpu.get("remaining", "0")).lower().replace("h", "").strip()
    try:
        remaining = float(raw)
    except ValueError as exc:
        raise SystemExit(f"unparseable GPU remaining value: {raw!r}") from exc
    record = {
        "schema_version": 1,
        "status": "quota-passed" if remaining > 0 else "blocked-no-gpu-quota",
        "owner": V5_OWNER,
        "owner_verified": True,
        "identity_probe": "kaggle datasets list --mine --search sem2act-v5-rerank-inputs --format json",
        "owned_input_dataset": expected_input,
        "protocol_hash": V5_PROTOCOL_HASH,
        "runtime_lock_sha256": _sha(V5_KAGGLE_LOCK),
        "runtime_freeze_sha256": _sha(V5_KAGGLE_FREEZE),
        "quota_command": "kaggle quota --format json",
        "quota_cli": quota_cmd,
        "gpu_remaining_hours": remaining,
        "gpu": gpu,
        "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    V5_KAGGLE_PREFLIGHT.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    if remaining <= 0:
        raise SystemExit("Kaggle GPU quota is zero")



def _verify_v5_rerank_pilot(dest, job):
    """Accept the 20-query / 1,000-score pilot only against the frozen pilot lock."""
    manifest = json.loads((dest / "run_manifest.json").read_text())
    if manifest.get("experiment_id") != "v5-reranker-pilot":
        raise SystemExit("wrong v5 reranker pilot experiment id")
    if manifest.get("status") != "pass" or manifest.get("qrels_read") is not False:
        raise SystemExit("v5 reranker pilot failure/qrel gate failed")
    if manifest.get("protocol_hash") != V5_PROTOCOL_HASH:
        raise SystemExit("v5 reranker pilot protocol hash drift")
    if manifest.get("amendment_id") != "sem2act-v5-kaggle-reranker-pilot-v1":
        raise SystemExit("v5 reranker pilot amendment drift")
    if manifest.get("lock_id") != "sem2act-v5-kaggle-reranker-pilot-v1":
        raise SystemExit("v5 reranker pilot lock id drift")
    if manifest.get("runtime_lock_sha256") != _sha(V5_PILOT_LOCK):
        raise SystemExit("v5 reranker pilot runtime lock hash drift")
    if manifest.get("parent_runtime_lock_sha256") != _sha(V5_KAGGLE_LOCK):
        raise SystemExit("v5 reranker pilot parent runtime lock hash drift")
    if manifest.get("model") != "Qwen/Qwen3-Reranker-0.6B":
        raise SystemExit("v5 reranker pilot model drift")
    if manifest.get("revision") != "e61197ed45024b0ed8a2d74b80b4d909f1255473":
        raise SystemExit("v5 reranker pilot model revision drift")
    if manifest.get("dtype") != "float16" or manifest.get("attention_backend") != "eager":
        raise SystemExit("v5 reranker pilot dtype/attention drift")
    if manifest.get("candidate_depth") != 50 or manifest.get("batch_size") != 32:
        raise SystemExit("v5 reranker pilot candidate depth or batch size drift")
    if manifest.get("max_context_tokens") != 1024:
        raise SystemExit("v5 reranker pilot max context drift")
    lock = json.loads(V5_PILOT_LOCK.read_text())
    if manifest.get("instruction_sha256") != lock["scoring"]["instruction_sha256"]:
        raise SystemExit("v5 reranker pilot instruction hash drift")
    snapshot = manifest.get("model_snapshot", {})
    if snapshot.get("file_count") != 12:
        raise SystemExit("v5 reranker pilot model snapshot file count drift")
    if snapshot.get("files_sha256") != lock["model"]["snapshot_files_sha256"]:
        raise SystemExit("v5 reranker pilot model snapshot aggregate drift")
    smoke = manifest.get("smoke", {})
    if smoke.get("status") != "pass" or smoke.get("fixture_sha256") != _sha(
            ROOT / "versions/sem2act-v5/fixtures/runtime_smoke_v1.json"):
        raise SystemExit("v5 reranker pilot smoke gate failed")
    upload = json.loads((ROOT / "versions/sem2act-v5/manifests/upload/v5-rerank-inputs.json").read_text())
    inputs = {name: value for name, value in manifest.get("input_sha256", {}).items()
              if name != "runtime_smoke_v1.json"}
    if inputs != upload["remote_sha256"]:
        raise SystemExit("v5 reranker pilot input hash gate failed")
    if manifest.get("n_fail") != 0:
        raise SystemExit("v5 reranker pilot reported kernel-side failures")
    if tuple(manifest.get("query_ids", ())) != V5_PILOT_QUERY_IDS:
        raise SystemExit("v5 reranker pilot query keyspace drift")
    if (manifest.get("n_queries") != 20 or manifest.get("n_scores") != 1000
            or manifest.get("n_ranking_rows") != 1000):
        raise SystemExit("v5 reranker pilot coverage gate failed")
    if manifest.get("rerank_trec_sha256") != _sha(dest / "rerank.trec"):
        raise SystemExit("v5 reranker pilot trec hash drift")
    rows = {}
    with (dest / "rerank.trec").open() as handle:
        for line in handle:
            parts = line.split()
            if len(parts) != 6 or parts[1] != "Q0" or parts[5] != "v5-rerank-pilot":
                raise SystemExit("v5 reranker pilot trec schema drift")
            rows.setdefault(parts[0], []).append((int(parts[3]), parts[2]))
    if set(rows) != set(V5_PILOT_QUERY_IDS):
        raise SystemExit("v5 reranker pilot query coverage failed")
    upload_candidates = {
        row["query_id"]: row["doc_ids"] for row in
        (json.loads(line) for line in (ROOT / job["dataset_dir"] / "rerank_candidates.jsonl")
         .read_text().splitlines() if line.strip())
    } if (ROOT / job["dataset_dir"] / "rerank_candidates.jsonl").is_file() else None
    for qid, ranking in rows.items():
        if len(ranking) != 50 or [rank for rank, _ in ranking] != list(range(1, 51)):
            raise SystemExit("v5 reranker pilot ranking depth or order failed")
        if len({did for _, did in ranking}) != 50:
            raise SystemExit("v5 reranker pilot ranking contains duplicate doc ids")
        if upload_candidates is not None and set(did for _, did in ranking) != set(upload_candidates.get(qid, [])):
            raise SystemExit("v5 reranker pilot ranking is not a permutation of the frozen candidates")
    return {"status": "pass", "n_queries": 20, "n_scores": 1000, "n_fail": 0,
            "n_accepted_queries": 20, "full_reranker_authorized": False}


def _verify_v5_rerank(dest, job):
    manifest = json.loads((dest / "run_manifest.json").read_text())
    if manifest.get("experiment_id") != "v5-lockbox-rerank":
        raise SystemExit("wrong v5 rerank experiment id")
    if manifest.get("qrels_read") is not False:
        raise SystemExit("v5 reranker manifest does not prove qrels_read=false")
    if manifest.get("runtime_lock_sha256") != _sha(V5_KAGGLE_LOCK):
        raise SystemExit("v5 reranker runtime lock hash drift")
    smoke = manifest.get("smoke", {})
    if smoke.get("status") != "pass" or smoke.get("fixture_sha256") != _sha(ROOT / "versions/sem2act-v5/fixtures/runtime_smoke_v1.json"):
        raise SystemExit("v5 reranker smoke gate failed")
    if manifest.get("n_queries") != 240 or manifest.get("n_fail") != 0:
        raise SystemExit("v5 reranker coverage/failure gate failed")
    rows = {}
    with (dest / "rerank.trec").open() as handle:
        for line in handle:
            qid, _, did, rank, *_ = line.split()
            rows.setdefault(qid, []).append((int(rank), did))
    if set(rows) != {f"v5-lb-q{i:04d}" for i in range(1, 241)}:
        raise SystemExit("v5 reranker query coverage failed")
    if any(len(v) != 50 or [r for r, _ in v] != list(range(1, 51))
           or len({d for _, d in v}) != 50 for v in rows.values()):
        raise SystemExit("v5 reranker ranking schema failed")
    return {"status": "pass", "n_queries": 240, "n_fail": 0}


V5_QWEN_MODEL = "Qwen/Qwen3-8B-AWQ"
V5_QWEN_REVISION = "4da05a8edb55c6046cce958586c33b61da07bb79"
V5_QWEN_PROMPT_SHAS = {"C1": "66e6890ba9c5464c", "C3": "5966cadde90e8236"}


def _v5_qwen_cache_fname(model, prompt_sha16, user):
    key_src = f"{model}||{prompt_sha16}||{hashlib.sha256(user.encode()).hexdigest()[:16]}"
    return hashlib.sha256(key_src.encode()).hexdigest()[:16] + ".json"


def _v5_qwen_expected_cache_keys():
    """Recompute the 2880 canonical invocation cache keys from frozen inputs.

    Mirrors the kernel loop order and consumers_v3 user-string construction.
    Deduplication is expected: identical deterministic requests share a key.
    """
    evidence = ROOT / "versions/sem2act-v5/runtime/kaggle_qwen_inputs/evidence_inputs.jsonl"
    if not evidence.is_file():
        raise SystemExit("v5 Qwen coverage recomputation needs the staged evidence bundle")
    rows = [json.loads(line) for line in evidence.read_text().splitlines() if line]
    keys = []
    for row in sorted(rows, key=lambda item: (item["system"], item["query_id"])):
        docs = [{"doc_id": item["doc_id"], "text": item["text"]} for item in row["documents"]]
        c1_user = ("Operator information need: " + row["query_text"] + "\n\nEvidence:\n" +
                   "\n\n".join(f"[DOC {d['doc_id']}] {d['text']}" for d in docs))
        keys.append(_v5_qwen_cache_fname(V5_QWEN_MODEL, V5_QWEN_PROMPT_SHAS["C1"], c1_user))
        for d in docs:
            c3_user = ("Operator's own node: " + row["metadata"]["entity_node"] +
                       "\n\nEvidence document:\n" + d["text"])
            keys.append(_v5_qwen_cache_fname(V5_QWEN_MODEL, V5_QWEN_PROMPT_SHAS["C3"], c3_user))
    return keys


def _v5_qwen_check_raw_coverage(raw_lines, expected_keys):
    """Dedup-aware coverage: 2880 lines in canonical order, every invocation key resolves."""
    if len(raw_lines) != len(expected_keys):
        raise SystemExit("v5 Qwen raw-output coverage failed")
    seen = []
    for index, line in enumerate(raw_lines):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            raise SystemExit("v5 Qwen raw-output parse failed")
        if obj.get("cache_file") != expected_keys[index]:
            raise SystemExit("v5 Qwen raw-output order/key gate failed")
        seen.append(obj["cache_file"])
    if set(seen) != set(expected_keys):
        raise SystemExit("v5 Qwen invocation cache coverage failed")
    return {"n_invocations": len(expected_keys), "n_unique_keys": len(set(expected_keys))}


def _verify_v5_qwen(dest, job):
    manifest = json.loads((dest / "run_manifest.json").read_text())
    if manifest.get("experiment_id") != "v5-primary-qwen-consumer":
        raise SystemExit("wrong v5 Qwen experiment id")
    if manifest.get("model") != V5_QWEN_MODEL or manifest.get("revision") != V5_QWEN_REVISION:
        raise SystemExit("v5 Qwen model pin gate failed")
    if manifest.get("status") != "pass" or manifest.get("qrels_read") is not False:
        raise SystemExit("v5 Qwen failure/qrel gate failed")
    if manifest.get("credential_source") not in ("public_unauthenticated", "hf_token_kaggle_secret"):
        raise SystemExit("v5 Qwen credential source gate failed")
    if manifest.get("runtime_amendment") != "sem2act-v5-qwen-runtime-amendment-2":
        raise SystemExit("v5 Qwen runtime amendment gate failed")
    rt = manifest.get("post_install_runtime", {})
    if (rt.get("torch"), rt.get("cuda_reported")) != ("2.8.0+cu128", "12.8"):
        raise SystemExit("v5 Qwen post-install runtime tuple gate failed")
    if rt.get("vllm") != "0.11.0":
        raise SystemExit("v5 Qwen inference package gate failed")
    for _k, _v in (("transformers", "4.57.6"), ("tokenizers", "0.22.2"),
                   ("triton", "3.4.0"), ("xformers", "0.0.32.post1"),
                   ("torchvision", "0.23.0"), ("torchaudio", "2.8.0"),
                   ("openai", "2.48.0"), ("pydantic", "2.12.5")):
        if rt.get("packages", {}).get(_k) != _v:
            raise SystemExit(f"v5 Qwen resolved-dependency gate failed: {_k}")
    if rt.get("cuda_available") is not True or rt.get("matmul_pass") is not True:
        raise SystemExit("v5 Qwen CUDA/matmul gate failed")
    if "T4" not in str(rt.get("device_name", "")):
        raise SystemExit("v5 Qwen GPU device gate failed")
    if manifest.get("runtime_lock_sha256") != _sha(V5_KAGGLE_LOCK):
        raise SystemExit("v5 Qwen runtime lock hash drift")
    smoke = manifest.get("smoke", {})
    if smoke.get("status") != "pass" or smoke.get("fixture_sha256") != _sha(ROOT / "versions/sem2act-v5/fixtures/runtime_smoke_v1.json"):
        raise SystemExit("v5 Qwen smoke gate failed")
    resp = smoke.get("response_sha256", [])
    if (not isinstance(resp, list) or len(resp) != 2
            or any(not isinstance(h, str) or len(h) != 64 for h in resp)):
        raise SystemExit("v5 Qwen smoke response gate failed")
    if manifest.get("expected_calls") != 2880 or manifest.get("actual_calls") != 2880:
        raise SystemExit("v5 Qwen call-count gate failed")
    expected_keys = _v5_qwen_expected_cache_keys()
    if len(expected_keys) != 2880:
        raise SystemExit("v5 Qwen invocation recomputation failed")
    if manifest.get("n_cache_files") != len(set(expected_keys)):
        raise SystemExit("v5 Qwen cache cardinality gate failed")
    if manifest.get("n_fail") != 0:
        raise SystemExit("v5 Qwen failure gate failed")
    if manifest.get("prompt_shas") != {"C1": "66e6890ba9c5464c", "C3": "5966cadde90e8236"}:
        raise SystemExit("v5 Qwen prompt hash drift")
    raw_lines = (dest / "raw_outputs.jsonl").read_text().splitlines()
    _v5_qwen_check_raw_coverage(raw_lines, expected_keys)
    rows = [json.loads(line) for line in (dest / "beliefs.jsonl").read_text().splitlines()]
    if len(rows) != 1440:
        raise SystemExit("v5 Qwen belief coverage failed")
    keys = {(row["query_id"], row["system"], row["consumer"]) for row in rows}
    expected = {
        (f"v5-lb-q{i:04d}", system, consumer)
        for i in range(1, 241)
        for system in ("bm25", "rerank", "oracle")
        for consumer in ("C1", "C3")
    }
    if keys != expected or len(keys) != len(rows):
        raise SystemExit("v5 Qwen belief keyspace failed")
    return {"status": "pass", "n_calls": 2880, "n_beliefs": 1440}


def _verify_v5_consumer(dest, job):
    manifest = json.loads((dest / "run_manifest.json").read_text())
    if manifest.get("experiment_id") != "v5-cross-family-consumer":
        raise SystemExit("wrong v5 consumer experiment id")
    if manifest.get("status") != "pass" or manifest.get("qrels_read") is not False:
        raise SystemExit("v5 consumer failure/qrel gate failed")
    if manifest.get("credential_source") not in ("public_unauthenticated", "hf_token_kaggle_secret", "rented_runtime_secret"):
        raise SystemExit("v5 consumer credential source gate failed")
    if manifest.get("runtime_lock_sha256") != _sha(V5_KAGGLE_LOCK):
        raise SystemExit("v5 consumer runtime lock hash drift")
    smoke = manifest.get("smoke", {})
    if smoke.get("status") != "pass" or smoke.get("fixture_sha256") != _sha(ROOT / "versions/sem2act-v5/fixtures/runtime_smoke_v1.json"):
        raise SystemExit("v5 consumer smoke gate failed")
    if manifest.get("expected_calls") != 1920 or manifest.get("actual_calls") != 1920:
        raise SystemExit("v5 consumer call-count gate failed")
    if manifest.get("n_fail") != 0:
        raise SystemExit("v5 consumer schema failure gate failed")
    if manifest.get("prompt_shas") != {"C1": "66e6890ba9c5464c", "C3": "5966cadde90e8236"}:
        raise SystemExit("v5 consumer prompt hash drift")
    if len((dest / "raw_outputs.jsonl").read_text().splitlines()) != 1920:
        raise SystemExit("v5 consumer raw-output coverage failed")
    belief_rows = [json.loads(line) for line in (dest / "beliefs.jsonl").read_text().splitlines()]
    if len(belief_rows) != 960:
        raise SystemExit("v5 consumer belief coverage failed")
    keys = {(row["query_id"], row["system"], row["consumer"]) for row in belief_rows}
    expected = {
        (f"v5-lb-q{i:04d}", system, consumer)
        for i in range(1, 241)
        for system in ("rerank", "oracle")
        for consumer in ("C1", "C3")
    }
    if keys != expected or len(keys) != len(belief_rows):
        raise SystemExit("v5 consumer belief keyspace failed")
    for row in belief_rows:
        probs = [row[name] for name in ("p_normal", "p_supplier_delay", "p_demand_surge")]
        if any(not 0 <= float(value) <= 1 for value in probs) or abs(sum(probs) - 1) > 1e-6:
            raise SystemExit("v5 consumer belief simplex failed")
    return {"status": "pass", "n_calls": 1920, "n_beliefs": 960}


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
    _require_kaggle_sdk()
    import json as _json
    from kagglesdk.kernels.types.kernels_api_service import (
        ApiListKernelSessionOutputRequest)
    req = ApiListKernelSessionOutputRequest()
    req.user_name, req.kernel_slug = _job_owner(job), job["slug"]
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


def _verify_v4phi4probe(dest, job):
    import json as _j
    rep = _j.loads((dest / "probe_report.json").read_text())
    print("v4 phi4 probe:", {k: v for k, v in rep.items()
                              if k not in ("cases", "sample")})
    assert rep.get("model") == PHI4_MODEL, rep
    assert rep.get("revision") == PHI4_REVISION, rep
    assert rep.get("prompt_shas") == {
        "C1": "66e6890ba9c5464c",
        "C3": "5966cadde90e8236",
    }, rep
    assert rep.get("pass") is True, rep
    for name in ("c1", "c3"):
        assert rep["cases"][name]["identical"] is True, rep
        assert rep["cases"][name]["both_valid"] is True, rep
    print("verify: v4 Phi-4 exact-schema probe passed")


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


def _verify_v4phi4beliefs(dest, job):
    """Strict v4 acceptance gate for one qrel-free cache shard.

    The older Phi-4 verifier intentionally sampled twenty cache entries.  The
    v4 wave is publication-bound, so this gate checks every returned payload,
    the pre-registered range and invocation count, and the exact output-name
    relation before anything is installed into the v4 runtime tree.
    """
    import json as _j
    import math as _math
    import re as _re
    import zipfile as _zip

    man = _j.loads((dest / "shard_manifest.json").read_text())
    expected_range = list(job["expected_range"])
    assert list(man.get("shard", [])) == expected_range, (man, expected_range)
    assert man.get("n_llm_calls") == job["expected_n_llm_calls"], man
    assert man.get("n_fail") == 0, man
    assert man.get("model") == PHI4_MODEL, man
    assert man.get("revision") == PHI4_REVISION, man
    assert man.get("c1_sha") == "66e6890ba9c5464c", man
    assert man.get("c3_sha") == "5966cadde90e8236", man
    assert man.get("tau") == 0.5, man
    assert man.get("seal") == "no qrels attached; guard passed", man
    assert expected_range[1] - expected_range[0] == job["expected_n_queries"]
    for f in ("consumers_v3.py", "interpreter.py", "events.py"):
        frozen = _sha(ROOT / "src" / f)[:16]
        assert man["src_snap_sha"].get(f) == frozen, (f, man)

    zname = f"cache_shard_{expected_range[0]}_{expected_range[1]}.zip"
    assert zname in job["outputs"], (zname, job["outputs"])
    with _zip.ZipFile(dest / zname) as z:
        names = z.namelist()
        assert len(names) == len(set(names)), "duplicate cache names in shard"
        assert all(n.endswith(".json") and "/" not in n for n in names), names[:3]
        assert len(names) == man["n_cache_files"] and len(names) > 0, man

        def strip_transport(raw):
            raw = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.S).strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return raw

        def finite_number(value):
            return isinstance(value, (int, float)) and not isinstance(value, bool) \
                and _math.isfinite(float(value))

        def validate_raw(prompt_sha, raw, filename):
            try:
                parsed = _j.loads(strip_transport(raw))
            except Exception as exc:
                raise AssertionError(f"invalid JSON in {filename}") from exc
            if not isinstance(parsed, dict):
                raise AssertionError(f"non-object response in {filename}")
            if prompt_sha == "66e6890ba9c5464c":
                required = {
                    "normal", "supplier_delay", "demand_surge",
                    "estimated_lt_increase", "estimated_duration",
                    "estimated_demand_multiplier",
                }
                assert set(parsed) == required, (filename, sorted(parsed))
                for key in required:
                    assert finite_number(parsed[key]), (filename, key, parsed[key])
                assert all(0.0 <= float(parsed[k]) <= 1.0
                           for k in ("normal", "supplier_delay", "demand_surge")), \
                    (filename, parsed)
                # The frozen consumer normalizes these three probabilities.
                # The prompt requests an approximate unit sum, but the
                # pinned model sometimes returns a valid positive vector at a
                # different scale (for example, [1, 1, 0]); rejecting that
                # would change the pre-registered consumer semantics.
                total = sum(float(parsed[k]) for k in
                            ("normal", "supplier_delay", "demand_surge"))
                assert total > 0.0, (filename, total)
            elif prompt_sha == "5966cadde90e8236":
                required = {"entity_match", "event", "fresh", "stance", "confidence"}
                assert set(parsed) == required, (filename, sorted(parsed))
                assert type(parsed["entity_match"]) is bool, (filename, parsed)
                assert type(parsed["fresh"]) is bool, (filename, parsed)
                assert parsed["event"] in {
                    "supplier_delay", "demand_surge", "normal", "none"
                }, (filename, parsed)
                assert parsed["stance"] in {"support", "refute", "na"}, (filename, parsed)
                assert finite_number(parsed["confidence"]), (filename, parsed)
                assert 0.0 <= float(parsed["confidence"]) <= 1.0, (filename, parsed)
            else:
                raise AssertionError((filename, prompt_sha))

        for filename in names:
            payload = _j.loads(z.read(filename))
            assert payload.get("model") == PHI4_MODEL, filename
            assert payload.get("prompt_sha") in {
                "66e6890ba9c5464c", "5966cadde90e8236"
            }, filename
            assert isinstance(payload.get("raw"), str) and payload["raw"].strip(), filename
            assert isinstance(payload.get("usage"), dict), filename
            validate_raw(payload["prompt_sha"], payload["raw"], filename)
    print(json.dumps({
        "verify": "v4 Phi-4 shard",
        "range": expected_range,
        "n_llm_calls": man["n_llm_calls"],
        "n_cache_files": man["n_cache_files"],
        "n_fail": man["n_fail"],
        "schema_records_checked": man["n_cache_files"],
    }, indent=2))
    return {
        "gate": "v4phi4beliefs",
        "strict": True,
        "expected_range": expected_range,
        "expected_n_llm_calls": job["expected_n_llm_calls"],
        "schema_records_checked": man["n_cache_files"],
    }


def _verify_qrel_calibration(dest, job):
    """Accept a complete calibration or a frozen terminal calibration failure.

    The latter is a valid scientific outcome: the external judge is excluded
    downstream and no replacement model or threshold is permitted.
    """
    import hashlib as _hashlib
    import json as _j

    run = _j.loads((dest / "RUN_MANIFEST.json").read_text())
    result = _j.loads((dest / "calibration_result.json").read_text())
    kernel = _j.loads((dest / "qrel_kernel_manifest.json").read_text())
    assert run.get("mode") == "calibration", run
    assert run.get("model") == "Qwen/Qwen3-8B-AWQ", run
    assert run.get("revision") == "4da05a8edb55c6046cce958586c33b61da07bb79", run
    assert run.get("temperature") == 0, run
    assert run.get("top_p") == 1.0, run
    assert run.get("max_tokens") == 256, run
    assert result.get("status") in {"pass", "failed_terminal"}, result
    assert result.get("n_pairs") == 200, result
    assert result.get("failure_is_terminal") is True, result
    assert result.get("no_model_or_threshold_substitution") is True, result
    assert kernel.get("mode") == "calibration", kernel
    assert _sha(dest / "runtime_versions.json") == kernel.get("runtime_versions_sha256"), kernel

    raw_rows = [_j.loads(line) for line in (dest / "raw_outputs.jsonl").read_text().splitlines() if line.strip()]
    if run.get("status") == "complete":
        assert len(raw_rows) == run.get("raw_outputs", {}).get("n_rows"), run
    else:
        print("terminal calibration failure: preserving observed raw-row count", len(raw_rows))
    assert _sha(dest / "raw_outputs.jsonl") == run.get("raw_outputs", {}).get("sha256"), run
    assert len({row.get("annotation_id") for row in raw_rows}) == len(raw_rows), "duplicate raw calibration keys"
    assert all(isinstance(row.get("raw_output"), str) and row.get("raw_output_sha256") ==
               _hashlib.sha256(row["raw_output"].encode()).hexdigest()
               for row in raw_rows), "raw calibration hash mismatch"
    if run.get("status") == "complete":
        assert len(raw_rows) == 200, run
        assert result.get("judge_run_manifest_sha256") == _sha(dest / "RUN_MANIFEST.json"), result
    print(json.dumps({
        "verify": "qrel calibration",
        "status": result["status"],
        "judge_run_status": run.get("status"),
        "n_pairs": result["n_pairs"],
        "raw_outputs_checked": len(raw_rows),
        "external_judge_permitted_for_sem2act": result.get("external_judge_permitted_for_sem2act", False),
    }, indent=2))
    return {
        "gate": "qrelcalibration",
        "calibration_status": result["status"],
        "judge_run_status": run.get("status"),
        "raw_outputs_checked": len(raw_rows),
        "terminal_failure_policy_verified": result.get("failure_is_terminal") is True,
    }


def _verify_qrel_sem2act(dest, job):
    """Strict acceptance gate for the one conditional 160-pair judge run."""
    import hashlib as _hashlib
    import json as _j

    run = _j.loads((dest / "RUN_MANIFEST.json").read_text())
    kernel = _j.loads((dest / "qrel_kernel_manifest.json").read_text())
    assert run.get("mode") == "sem2act", run
    assert run.get("status") == "complete", run
    assert run.get("model") == "Qwen/Qwen3-8B-AWQ", run
    assert run.get("revision") == "4da05a8edb55c6046cce958586c33b61da07bb79", run
    assert run.get("temperature") == 0, run
    assert run.get("top_p") == 1.0, run
    assert run.get("max_tokens") == 256, run
    assert not run.get("errors"), run
    assert run.get("input", {}).get("n_rows") == 160, run
    assert run.get("raw_outputs", {}).get("n_rows") == 160, run
    assert run.get("parsed_labels", {}).get("n_rows") == 160, run
    assert kernel.get("mode") == "sem2act", kernel
    assert _sha(dest / "runtime_versions.json") == kernel.get("runtime_versions_sha256"), kernel
    for filename, expected in (("raw_outputs.jsonl", 160), ("judge_labels.jsonl", 160)):
        rows = [_j.loads(line) for line in (dest / filename).read_text().splitlines() if line.strip()]
        assert len(rows) == expected, (filename, len(rows))
    raw_rows = [_j.loads(line) for line in (dest / "raw_outputs.jsonl").read_text().splitlines() if line.strip()]
    assert _sha(dest / "raw_outputs.jsonl") == run["raw_outputs"]["sha256"], run
    assert all(isinstance(row.get("raw_output"), str) and row.get("raw_output_sha256") ==
               _hashlib.sha256(row["raw_output"].encode()).hexdigest()
               for row in raw_rows), "raw Sem2Act hash mismatch"
    label_rows = [_j.loads(line) for line in (dest / "judge_labels.jsonl").read_text().splitlines() if line.strip()]
    assert _sha(dest / "judge_labels.jsonl") == run["parsed_labels"]["sha256"], run
    assert len({row.get("annotation_id") for row in label_rows}) == 160, "duplicate Sem2Act labels"
    assert all(isinstance(row.get("label"), int) and row["label"] in range(4) for row in label_rows)
    print(json.dumps({
        "verify": "qrel Sem2Act judge",
        "status": run["status"],
        "n_pairs": 160,
        "raw_outputs_checked": len(raw_rows),
        "labels_checked": len(label_rows),
    }, indent=2))
    return {
        "gate": "qrelsem2act",
        "status": run["status"],
        "raw_outputs_checked": len(raw_rows),
        "labels_checked": len(label_rows),
    }


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


def _verify_reasoningsmoke(dest, job):
    """GPU smoke fetch gate: zero failures, budgets exact, prompt SHAs match frozen DEV."""
    import json as _j
    import zipfile as _zf
    man = _j.loads((dest / "smoke_manifest.json").read_text())
    assert man["n_failures"] == 0, man["failures"]
    assert man["gate_zero_parse_failures"] == "PASS", man
    assert man["budgets"] == {"A0": 0, "A1": 0, "A2": 0, "R0": 1, "A3": 2}, man["budgets"]
    assert man["n_warnings"] == 6, man
    sys.path.insert(0, str(ROOT))
    from experiments.reasoning_agentic.prompts import prompt_hashes
    frozen = prompt_hashes()
    got = man["prompt_shas"]
    for k, h in (("A0", "A0_DIRECT_SYS"), ("A1", "A1_DELIB_SYS"), ("A2", "A2_VERIFY_SYS"),
                 ("R0", "R0_SYNTH_SYS"), ("A3H", "A3_HYPO_SYS"), ("A3S", "A3_SYNTH_SYS")):
        assert got[k] == frozen[h], (k, got[k], frozen[h])
    for row in man["call_logs"]:
        cap = {"A0": 0, "A1": 0, "A2": 0, "R0": 1, "A3": 2}[row["arm"]]
        if row["arm"] == "A3":
            assert 1 <= row["retrieval_calls"] <= cap, row
        else:
            assert row["retrieval_calls"] == cap, row
    with _zf.ZipFile(dest / "cache_reasoning_smoke.zip") as z:
        keys = {n[:-5] for n in z.namelist() if n.endswith(".json")}
    assert len(keys) == 30, keys  # 6 warnings x 5 arms
    print(f"verify: reasoning smoke ok ({len(keys)} beliefs, 0 failures, prompts match frozen DEV)")


def _verify_reasoningfinal(dest, job):
    """Final-run fetch gate: 27 warnings x 5 arms = 135 beliefs, zero failures."""
    import json as _j
    import zipfile as _zf
    man = _j.loads((dest / "smoke_manifest.json").read_text())
    assert man["n_failures"] == 0, man["failures"]
    assert man["gate_zero_parse_failures"] == "PASS", man
    assert man["budgets"] == {"A0": 0, "A1": 0, "A2": 0, "R0": 1, "A3": 2}, man["budgets"]
    assert man["n_warnings"] == 27, man
    assert man["model"] == "Qwen/Qwen3-8B-AWQ", man
    assert man["revision"] == "4da05a8edb55c6046cce958586c33b61da07bb79", man
    sys.path.insert(0, str(ROOT))
    from experiments.reasoning_agentic.prompts import prompt_hashes
    frozen = prompt_hashes()
    got = man["prompt_shas"]
    for k, h in (("A0", "A0_DIRECT_SYS"), ("A1", "A1_DELIB_SYS"), ("A2", "A2_VERIFY_SYS"),
                 ("R0", "R0_SYNTH_SYS"), ("A3H", "A3_HYPO_SYS"), ("A3S", "A3_SYNTH_SYS")):
        assert got[k] == frozen[h], (k, got[k], frozen[h])
    for row in man["call_logs"]:
        if row["arm"] == "A3":
            assert 1 <= row["retrieval_calls"] <= 2, row
        else:
            assert row["retrieval_calls"] == {"A0": 0, "A1": 0, "A2": 0, "R0": 1}[row["arm"]], row
    with _zf.ZipFile(dest / "cache_reasoning_smoke.zip") as z:
        keys = {n[:-5] for n in z.namelist() if n.endswith(".json")}
    assert len(keys) == 135, len(keys)
    print(f"verify: reasoning final ok ({len(keys)} beliefs, 0 failures, prompts+model match v1.1)")


def cmd_test(job):
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_v3proto.py", "-q"],
                       cwd=str(ROOT))
    raise SystemExit(r.returncode)


def main():
    ap = argparse.ArgumentParser(description="Paper-2 Kaggle remote-compute runner")
    ap.add_argument("job", nargs="?", choices=list(JOBS))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset", choices=["create", "version"])
    g.add_argument("--push", action="store_true")
    g.add_argument("--status", action="store_true")
    g.add_argument("--watch", action="store_true")
    g.add_argument("--fetch", action="store_true")
    g.add_argument("--test", action="store_true")
    g.add_argument("--logs", action="store_true")
    g.add_argument("--preflight", action="store_true")
    a = ap.parse_args()
    if a.preflight:
        cmd_v5_preflight()
        return
    if not a.job:
        ap.error("a job is required unless --preflight is used")
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
