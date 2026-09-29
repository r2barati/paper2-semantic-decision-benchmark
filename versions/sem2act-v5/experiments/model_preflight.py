"""Local preflight for the v5 official consumer jobs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODELS = (
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
)
PROMPT_SHAS = {"C1": "66e6890ba9c5464c", "C3": "5966cadde90e8236"}
QUANTIZATION = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_use_double_quant": True,
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    lockbox = json.loads(
        (ROOT / "versions/sem2act-v5/manifests/lockbox_freeze.json").read_text()
    )
    protocol = json.loads(
        (ROOT / "versions/sem2act-v5/manifests/protocol_freeze.json").read_text()
    )
    staged = ROOT / "versions/sem2act-v5/runtime/kaggle_rerank_inputs"
    if any("qrel" in p.name.lower() for p in staged.iterdir()):
        raise SystemExit("qrel-bearing staged input detected")
    report = {
        "schema_version": 1,
        "experiment_id": "v5-model-preflight",
        "status": "gpu-probe-pending",
        "qrels_read": False,
        "lockbox_manifest_sha256": sha(
            ROOT / "versions/sem2act-v5/manifests/lockbox_freeze.json"
        ),
        "protocol_hash": protocol["protocol_hash"],
        "cross_family_models": [
            {"id": model, "revision": "resolved_at_gpu_preflight"}
            for model in MODELS
        ],
        "prompt_shas": PROMPT_SHAS,
        "quantization": QUANTIZATION,
        "reranker_inputs": {
            p.name: sha(p) for p in sorted(staged.iterdir()) if p.is_file()
        },
        "lockbox_status": lockbox["status"],
        "failure_policy": "A failed exact-schema preflight excludes that official model without substitution.",
    }
    out = ROOT / "versions/sem2act-v5/manifests/model_preflight.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
