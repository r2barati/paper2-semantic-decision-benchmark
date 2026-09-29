"""Verify an exact Mistral source assembly from a public mirror plus pinned patch."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path

EMBEDDED_ASSEMBLY_CONFIG = None


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        raise RuntimeError(f"expected one Mistral weight directory, found {len(candidates)}")
    return candidates[0]


def find_patch(root: Path, slug: str) -> Path:
    candidates = [
        item for item in root.rglob("tokenizer_config.json")
        if slug in str(item)
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one pinned tokenizer patch, found {len(candidates)}")
    return candidates[0]


def main() -> int:
    started = time.monotonic()
    result = {
        "schema_version": 1,
        "manifest_id": "sem2act-v5-kaggle-mistral-source-assembly-probe-v2",
        "status": "failed",
        "source_only": True,
        "result_bearing_execution_started": False,
        "qrels_read": False,
        "inference_calls": 0,
    }
    try:
        config = EMBEDDED_ASSEMBLY_CONFIG
        if config is None:
            config = json.loads(Path("assembly_probe_config.json").read_text())
        input_root = Path(os.environ.get("SEM2ACT_MODEL_INPUT_ROOT", "/kaggle/input"))
        model_dir = find_model(input_root)
        patch_path = find_patch(input_root, config["patch_dataset_slug"])
        actual_files = []
        for item in sorted(model_dir.rglob("*")):
            if item.is_file() and ".cache" not in item.parts:
                source_path = patch_path if item.name == "tokenizer_config.json" else item
                actual_files.append({
                    "path": item.relative_to(model_dir).as_posix(),
                    "sha256": sha(source_path),
                    "bytes": source_path.stat().st_size,
                    "source_path": str(source_path),
                    "patched": item.name == "tokenizer_config.json",
                })
        expected = config["expected_files"]
        observed = {item["path"]: item for item in actual_files}
        comparisons = {}
        for path, expected_item in expected.items():
            found = observed.get(path)
            comparable = None if found is None else {
                "path": path,
                "sha256": found["sha256"],
                "bytes": found["bytes"],
            }
            comparisons[path] = {
                "expected": {"path": path, **expected_item},
                "observed": comparable,
                "match": comparable == {"path": path, **expected_item},
            }
        effective_files = [
            {key: item[key] for key in ("path", "sha256", "bytes")}
            for item in actual_files
        ]
        snapshot_digest = hashlib.sha256(
            json.dumps(effective_files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        required_match = all(item["match"] for item in comparisons.values())
        exact_file_set = {item["path"] for item in actual_files} == set(expected)
        result.update({
            "family": "mistral",
            "model_id": config["model_id"],
            "revision": config["revision"],
            "candidate_dataset": config["candidate_dataset"],
            "patch_dataset": config["patch_dataset"],
            "resolved_model_path": str(model_dir),
            "resolved_patch_path": str(patch_path),
            "source_assembly": "candidate_weights_plus_pinned_tokenizer_config",
            "source_model_snapshot": {
                "file_count": len(effective_files),
                "files_sha256": snapshot_digest,
                "files": effective_files,
            },
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
            raise RuntimeError("assembled Mistral source does not match the pinned HF file set")
        result["status"] = "pass"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["wall_seconds"] = round(time.monotonic() - started, 3)
    Path("mistral_source_assembly_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
