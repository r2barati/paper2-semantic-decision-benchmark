"""Hash and validate one pinned official Kaggle model source; never infer."""
from __future__ import annotations
import hashlib
import json
import os
import platform
import time
from pathlib import Path

EMBEDDED_SOURCE_CONFIG = None


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
        files.append({"path": str(item.relative_to(path)), "sha256": sha(item), "bytes": item.stat().st_size})
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"file_count": len(files), "files_sha256": digest, "files": files}


def input_inventory(root: Path) -> dict:
    if not root.exists():
        return {"root": str(root), "exists": False, "entries": []}
    entries = []
    for item in sorted(root.rglob("*")):
        entries.append(str(item.relative_to(root)))
        if len(entries) >= 200:
            break
    return {"root": str(root), "exists": True, "entry_count_sampled": len(entries), "entries": entries}


def find_model(config: dict) -> tuple[Path, dict]:
    source = config["model_source"]
    root = Path(os.environ.get("SEM2ACT_MODEL_INPUT_ROOT", "/kaggle/input"))
    if not root.exists():
        raise RuntimeError("Kaggle input root is missing")
    candidates = []
    for config_path in root.rglob("config.json"):
        try:
            model_config = json.loads(config_path.read_text())
        except Exception:
            continue
        if model_config.get("model_type") != source["model_type"]:
            continue
        if source["architecture"] not in (model_config.get("architectures") or []):
            continue
        model_dir = config_path.parent
        has_weights = any(model_dir.glob("*.safetensors")) or any(model_dir.glob("*.bin")) or (model_dir / "model.safetensors.index.json").exists()
        if has_weights:
            candidates.append((model_dir, model_config))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one mounted official source, found {len(candidates)}")
    return candidates[0]


def main() -> int:
    started = time.monotonic()
    result = {
        "schema_version": 1,
        "manifest_id": "sem2act-v5-kaggle-model-source-probe-v1",
        "status": "failed",
        "source_only": True,
        "result_bearing_execution_started": False,
        "qrels_read": False,
    }
    try:
        config = EMBEDDED_SOURCE_CONFIG
        if config is None:
            config = json.loads(Path("source_probe_config.json").read_text())
        input_root = Path(os.environ.get("SEM2ACT_MODEL_INPUT_ROOT", "/kaggle/input"))
        result["input_inventory"] = input_inventory(input_root)
        model_dir, model_config = find_model(config)
        result.update({
            "family": config["family"],
            "model_id": config["model_id"],
            "revision": config["revision"],
            "source": dict(config["model_source"], resolved_model_path=str(model_dir)),
            "model_config": {
                "model_type": model_config.get("model_type"),
                "architectures": model_config.get("architectures"),
                "config_sha256": sha(model_dir / "config.json"),
            },
            "source_model_snapshot": snapshot(model_dir),
            "platform": {"python": platform.python_version(), "machine": platform.machine(), "cpu_count": os.cpu_count()},
            "inference_calls": 0,
        })
        result["status"] = "pass"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["wall_seconds"] = round(time.monotonic() - started, 3)
    Path("source_probe_manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
