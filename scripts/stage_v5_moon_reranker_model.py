"""Stage exactly the model files authorized by the frozen reranker snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "versions/sem2act-v5/manifests/cpu_reranker_canary.json"


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


def authorized_snapshot(reference_path: Path = REFERENCE) -> tuple[str, str, dict, list[str]]:
    reference = json.loads(reference_path.read_text())
    if (reference.get("status") != "pass"
            or reference.get("model_id") != "Qwen/Qwen3-Reranker-0.6B"
            or reference.get("revision") != "e61197ed45024b0ed8a2d74b80b4d909f1255473"):
        raise RuntimeError("committed reranker model snapshot reference is not accepted")
    accepted = reference.get("model_snapshot")
    if not isinstance(accepted, dict) or accepted.get("file_count") != 12:
        raise RuntimeError("accepted reranker snapshot must contain exactly 12 files")
    entries = accepted.get("files")
    if not isinstance(entries, list) or len(entries) != 12:
        raise RuntimeError("accepted reranker snapshot file list is malformed")
    paths = []
    for entry in entries:
        relative = Path(entry.get("path", ""))
        if (relative.is_absolute() or not relative.parts
                or ".." in relative.parts or relative.as_posix() != entry.get("path")
                or len(entry.get("sha256", "")) != 64
                or not isinstance(entry.get("bytes"), int)):
            raise RuntimeError("accepted reranker snapshot has an invalid file entry")
        paths.append(relative.as_posix())
    if paths != sorted(paths) or len(set(paths)) != 12:
        raise RuntimeError("accepted reranker snapshot paths are not unique and sorted")
    aggregate = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if aggregate != accepted.get("files_sha256"):
        raise RuntimeError("accepted reranker snapshot aggregate hash is inconsistent")
    actual_shape = {"file_count": len(entries), "files_sha256": aggregate, "files": entries}
    if actual_shape != accepted:
        raise RuntimeError("accepted reranker snapshot manifest shape is inconsistent")
    return reference["model_id"], reference["revision"], accepted, paths


def stage(output: Path, manifest_output: Path) -> dict:
    model_id, revision, accepted, allow_patterns = authorized_snapshot()
    output = output.resolve()
    manifest_output = manifest_output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to stage into a non-empty model directory: {output}")
    if manifest_output.exists():
        raise RuntimeError(f"refusing to overwrite existing staging manifest: {manifest_output}")
    if manifest_output == output or manifest_output.is_relative_to(output):
        raise RuntimeError("staging manifest must be written outside the model directory")

    cache_root = Path("/tmp/sem2act-v5/cache/huggingface")
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_root / "hub")
    os.environ["TORCH_HOME"] = "/tmp/sem2act-v5/cache/torch"

    from huggingface_hub import snapshot_download

    output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=str(output),
        allow_patterns=allow_patterns,
    )
    actual = snapshot(output)
    if actual != accepted:
        raise RuntimeError("staged files differ from the exact accepted 12-file snapshot")
    result = {
        "schema_version": 1,
        "manifest_id": "sem2act-v5-moon-reranker-model-staging-v1",
        "status": "pass",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "revision": revision,
        "model_path": str(output),
        "reference_manifest": str(REFERENCE.relative_to(ROOT)),
        "reference_manifest_sha256": sha(REFERENCE),
        "allow_patterns": allow_patterns,
        "model_snapshot": actual,
        "model_snapshot_sha256": actual["files_sha256"],
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = stage(args.output, args.manifest_output)
    except Exception as exc:
        raise SystemExit(f"model staging stopped: {type(exc).__name__}: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
