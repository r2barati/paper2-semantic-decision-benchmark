"""Hash a public Kaggle mirror of pinned Mistral files; never infer."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path

EMBEDDED_MIRROR_CONFIG = None


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(path: Path) -> dict:
    files = []
    for item in sorted(path.rglob("*")):
        if not item.is_file() or ".cache" in item.parts:
            continue
        files.append({
            "path": str(item.relative_to(path)),
            "sha256": sha(item),
            "bytes": item.stat().st_size,
        })
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"file_count": len(files), "files_sha256": digest, "files": files}


def find_model(root: Path) -> Path:
    candidates = []
    for config_path in root.rglob("config.json"):
        try:
            config = json.loads(config_path.read_text())
        except Exception:
            continue
        if config.get("model_type") != "mistral":
            continue
        if "MistralForCausalLM" not in (config.get("architectures") or []):
            continue
        model_dir = config_path.parent
        if any(model_dir.glob("*.safetensors")) and (model_dir / "model.safetensors.index.json").exists():
            candidates.append(model_dir)
    if len(candidates) != 1:
        raise RuntimeError(f"expected one candidate Mistral mirror, found {len(candidates)}")
    return candidates[0]


def main() -> int:
    started = time.monotonic()
    result = {
        "schema_version": 1,
        "manifest_id": "sem2act-v5-kaggle-mistral-mirror-probe-v1",
        "status": "failed",
        "source_only": True,
        "result_bearing_execution_started": False,
        "qrels_read": False,
        "inference_calls": 0,
    }
    try:
        config = EMBEDDED_MIRROR_CONFIG
        if config is None:
            config = json.loads(Path("mirror_probe_config.json").read_text())
        input_root = Path(os.environ.get("SEM2ACT_MODEL_INPUT_ROOT", "/kaggle/input"))
        model_dir = find_model(input_root)
        actual = snapshot(model_dir)
        expected = config["expected_files"]
        actual_by_path = {item["path"]: item for item in actual["files"]}
        comparisons = {}
        for path, expected_item in expected.items():
            observed = actual_by_path.get(path)
            comparisons[path] = {
                "expected": expected_item,
                "observed": observed,
                "match": observed == {"path": path, **expected_item},
            }
        required_match = all(item["match"] for item in comparisons.values())
        exact_file_set = set(actual_by_path) == set(expected)
        result.update({
            "family": "mistral",
            "model_id": config["model_id"],
            "revision": config["revision"],
            "candidate_dataset": config["candidate_dataset"],
            "resolved_model_path": str(model_dir),
            "source_model_snapshot": actual,
            "expected_files": expected,
            "file_comparisons": comparisons,
            "required_files_match": required_match,
            "exact_file_set": exact_file_set,
            "platform": {
                "python": platform.python_version(),
                "machine": platform.machine(),
                "cpu_count": os.cpu_count(),
            },
        })
        if not required_match or not exact_file_set:
            raise RuntimeError("candidate mirror does not match the pinned HF file set")
        result["status"] = "pass"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["wall_seconds"] = round(time.monotonic() - started, 3)
    Path("mistral_mirror_probe_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
