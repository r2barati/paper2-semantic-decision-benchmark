"""Validate local v5 model-input bundles before any remote submission."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "versions/sem2act-v5/manifests/protocol_freeze.json"
KERNELS = {
    "qwen": ROOT / "versions/sem2act-v5/compute/kaggle_kernel/p2_v5_qwen_consumer.py",
    "llama": ROOT / "versions/sem2act-v5/compute/kaggle_kernel/p2_v5_consumer.py",
    "mistral": ROOT / "versions/sem2act-v5/compute/kaggle_kernel/p2_v5_consumer.py",
}
MODELS = {
    "qwen": ("Qwen/Qwen3-8B-AWQ", "4da05a8edb55c6046cce958586c33b61da07bb79"),
    "llama": ("meta-llama/Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659"),
    "mistral": ("mistralai/Mistral-7B-Instruct-v0.3", "d79d1742f78eb0cc788c11e5b41a7539d7cb56ef"),
}
PROMPTS = {"C1": "66e6890ba9c5464c", "C3": "5966cadde90e8236"}
RUNTIME_LOCK = ROOT / "versions/sem2act-v5/manifests/kaggle_runtime_lock.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text())
    bundles = {}
    for family, (model_id, revision) in MODELS.items():
        directory = ROOT / f"versions/sem2act-v5/runtime/kaggle_{family}_inputs"
        if not directory.is_dir():
            raise SystemExit(f"missing staged {family} bundle: {directory}")
        forbidden = [
            path for path in directory.rglob("*")
            if path.is_file() and "qrel" in path.name.lower()
        ]
        if forbidden:
            raise SystemExit(f"qrel-bearing staged file: {forbidden[0]}")
        spec_path = directory / "model_spec.json"
        evidence_path = directory / "evidence_inputs.jsonl"
        fixture_path = directory / "runtime_smoke_v1.json"
        if not spec_path.exists() or not evidence_path.exists() or not fixture_path.exists():
            raise SystemExit(f"{family}: required staged files missing")
        spec = json.loads(spec_path.read_text())
        if spec.get("model_id") != model_id or spec.get("revision") != revision:
            raise SystemExit(f"{family}: model pin drift")
        if spec.get("protocol_hash") != protocol["protocol_hash"]:
            raise SystemExit(f"{family}: protocol hash drift")
        if spec.get("runtime_lock_sha256") != sha(RUNTIME_LOCK):
            raise SystemExit(f"{family}: Kaggle runtime lock hash drift")
        if spec.get("prompt_shas") != PROMPTS:
            raise SystemExit(f"{family}: prompt hash drift")
        if sha(fixture_path) != sha(ROOT / "versions/sem2act-v5/fixtures/runtime_smoke_v1.json"):
            raise SystemExit(f"{family}: smoke fixture hash drift")
        if "qrel" in evidence_path.read_text(errors="ignore").lower():
            raise SystemExit(f"{family}: qrel token in evidence")
        bundles[family] = {
            "model_id": model_id,
            "revision": revision,
            "files": {
                path.name: {"sha256": sha(path), "bytes": path.stat().st_size}
                for path in sorted(directory.iterdir()) if path.is_file()
            },
        }
    report = {
        "schema_version": 1,
        "experiment_id": "v5-model-preflight",
        "status": "input-preflight-passed-gpu-probe-pending",
        "protocol_hash": protocol["protocol_hash"],
        "prompt_shas": PROMPTS,
        "runtime_lock_sha256": sha(RUNTIME_LOCK),
        "models": bundles,
        "kernels": {family: sha(path) for family, path in KERNELS.items()},
        "gpu_probe_required": True,
        "failure_policy": "Any failed official access or exact-schema probe excludes that model; no substitution.",
    }
    out = ROOT / "versions/sem2act-v5/manifests/model_preflight.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
