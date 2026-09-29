"""Moon's single-shard reranker pilot with a host and 4-vs-16 parity gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_HASH = "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022"
LOCK_REL = Path("versions/sem2act-v5/manifests/moon_reranker_runtime_lock.json")
FREEZE_REL = Path("versions/sem2act-v5/manifests/moon_reranker_runtime_freeze.json")
INPUT_MANIFEST_REL = Path("versions/sem2act-v5/manifests/upload/v5-rerank-inputs.json")
MODEL_REFERENCE_REL = Path("versions/sem2act-v5/manifests/cpu_reranker_canary.json")
FIXTURE_REL = Path("versions/sem2act-v5/fixtures/runtime_smoke_v1.json")
MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
MODEL_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
INSTRUCTION = (
    "Given an operational supply-chain information need, retrieve documents "
    "relevant to the described entity, event, and time context."
)


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


def require_checkout(expected_sha: str) -> str:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "-uall"], cwd=ROOT, text=True
    )
    if head != expected_sha:
        raise SystemExit(f"checkout SHA mismatch: expected {expected_sha}, got {head}")
    if status:
        raise SystemExit("checkout is dirty; refusing Moon pilot")
    return subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()


def nfs_type(path: Path) -> str:
    result = subprocess.run(
        ["findmnt", "-n", "-o", "FSTYPE", "--target", str(path)],
        check=False, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("findmnt could not identify the persistent output filesystem")
    return result.stdout.strip().lower()


def memory_total_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/meminfo has no MemTotal")


def cpu_flags() -> list[set[str]]:
    per_cpu = []
    for block in Path("/proc/cpuinfo").read_text().split("\n\n"):
        for line in block.splitlines():
            if line.startswith("flags") or line.startswith("Features"):
                per_cpu.append(set(line.split(":", 1)[1].split()))
                break
    if not per_cpu:
        raise RuntimeError("could not read the CPU instruction flags")
    return per_cpu


def probe_nfs_capacity(directory: Path, required_bytes: int) -> int:
    """Prove this account can write the compact checkpoint budget on NFS."""
    fd, filename = tempfile.mkstemp(prefix=".sem2act-v5-quota-probe-", dir=directory)
    probe = Path(filename)
    written = 0
    try:
        with os.fdopen(fd, "wb") as handle:
            block = b"\0" * (1024 * 1024)
            while written < required_bytes:
                chunk = block[:min(len(block), required_bytes - written)]
                handle.write(chunk)
                written += len(chunk)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        probe.unlink(missing_ok=True)
    return written


def validate_inputs(input_root: Path) -> dict:
    manifest = json.loads((ROOT / INPUT_MANIFEST_REL).read_text())
    if (manifest.get("status") != "remote-verified"
            or manifest.get("protocol_hash") != PROTOCOL_HASH
            or manifest.get("qrels_uploaded") is not False):
        raise RuntimeError("the frozen Kaggle reranker input manifest is not accepted and qrel-free")
    expected = manifest.get("files", {})
    names = {"queries.jsonl", "corpus.jsonl", "rerank_candidates.jsonl"}
    if set(expected) != names:
        raise RuntimeError("frozen reranker input manifest has unexpected files")
    if {path.name for path in input_root.iterdir()} != names:
        raise RuntimeError("Moon input directory must contain only the three frozen qrel-free files")
    actual = {name: sha(input_root / name) for name in sorted(names)}
    if actual != expected:
        raise RuntimeError("Moon input files differ from the version-1 Kaggle SHA-256 manifest")
    if any("qrel" in path.name.lower() for path in input_root.iterdir()):
        raise RuntimeError("qrel-bearing input filename refused")
    for name in names:
        if "qrel" in (input_root / name).read_text(errors="ignore").lower():
            raise RuntimeError(f"qrel token detected in {name}")
    queries = [json.loads(line) for line in (input_root / "queries.jsonl").read_text().splitlines() if line]
    candidates = [json.loads(line) for line in (input_root / "rerank_candidates.jsonl").read_text().splitlines() if line]
    if len(queries) != 240 or len(candidates) != 240:
        raise RuntimeError("reranker input keyspace must contain exactly 240 queries")
    qids = {row.get("_id") for row in queries}
    if qids != {f"v5-lb-q{i:04d}" for i in range(1, 241)}:
        raise RuntimeError("reranker query keyspace differs from the frozen lockbox")
    candidate_map = {row.get("query_id"): row.get("doc_ids", []) for row in candidates}
    if set(candidate_map) != qids or any(len(ids) != 50 for ids in candidate_map.values()):
        raise RuntimeError("reranker candidate coverage/depth differs from the frozen protocol")
    return {"manifest_sha256": sha(ROOT / INPUT_MANIFEST_REL), "files": actual,
            "query_count": len(queries), "candidate_depth": 50}


def validate_model(model_path: Path) -> dict:
    expected_manifest = json.loads((ROOT / MODEL_REFERENCE_REL).read_text())
    if (expected_manifest.get("status") != "pass"
            or expected_manifest.get("model_id") != MODEL_ID
            or expected_manifest.get("revision") != MODEL_REVISION):
        raise RuntimeError("committed reranker model snapshot reference is not accepted")
    expected_snapshot = expected_manifest["model_snapshot"]
    actual_snapshot = snapshot(model_path)
    if actual_snapshot != expected_snapshot:
        raise RuntimeError("local reranker files do not match the pinned accepted model snapshot")
    return actual_snapshot


def host_preflight(args: argparse.Namespace, branch: str, inputs: dict, model_snapshot: dict) -> dict:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if platform.system() != "Linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        raise RuntimeError(f"Moon host must be Linux x86_64, got {platform.platform()}")
    if sys.version.split()[0] != "3.9.19":
        raise RuntimeError(f"Python profile drift: expected 3.9.19, got {sys.version.split()[0]}")
    if torch.__version__ != "2.8.0+cpu":
        raise RuntimeError(f"Torch profile drift: expected 2.8.0+cpu, got {torch.__version__}")
    if transformers.__version__ != "4.57.6":
        raise RuntimeError(f"Transformers profile drift: expected 4.57.6, got {transformers.__version__}")
    if torch.cuda.is_available() or (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise RuntimeError("Moon reranker route requires CPU-only execution")
    if os.cpu_count() != 16:
        raise RuntimeError(f"Moon profile requires exactly 16 visible vCPUs, got {os.cpu_count()}")
    total_memory = memory_total_bytes()
    if total_memory < 60_000_000_000:
        raise RuntimeError(f"Moon profile requires at least 60 GB RAM, got {total_memory} bytes")
    per_cpu_flags = cpu_flags()
    if any("avx2" in flags or "avx512f" in flags for flags in per_cpu_flags):
        raise RuntimeError("CPU ISA profile changed: AVX2/AVX512 flags must remain masked")

    scratch = Path("/tmp/sem2act-v5").resolve()
    input_root = args.input_root.resolve()
    model_root = args.model_path.resolve()
    for label, path in (("input", input_root), ("model", model_root)):
        if not path.is_relative_to(scratch):
            raise RuntimeError(f"{label} must be under node-local {scratch}: {path}")
    scratch.mkdir(parents=True, exist_ok=True)
    scratch_free = shutil.disk_usage(scratch).free
    if scratch_free < 5_000_000_000:
        raise RuntimeError(f"node-local scratch needs at least 5 GB free, got {scratch_free}")
    output_root = args.output_root.resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    filesystem = nfs_type(output_root.parent)
    if not filesystem.startswith("nfs"):
        raise RuntimeError(f"persistent output must be on NFS, got {filesystem!r}")
    quota_probe_bytes = probe_nfs_capacity(output_root.parent, 10_000_000)
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty pilot output directory: {output_root}")

    # Load exactly the accepted reranker snapshot, then compare all three
    # non-lockbox fixture scores and ranks across the 4- and 16-thread profiles.
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(
        model_root, revision=MODEL_REVISION, padding_side="left", trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_root, revision=MODEL_REVISION, trust_remote_code=True,
        torch_dtype=torch.float32, device_map={"": "cpu"}
    ).eval()
    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")
    fixture_path = ROOT / FIXTURE_REL
    fixture = json.loads(fixture_path.read_text())
    prefix = tokenizer.encode(
        '<|im_start|>system\nJudge whether the Document meets the requirements based '
        'on the Query and the Instruct provided. Note that the answer can only be '
        '"yes" or "no".<|im_end|>\n<|im_start|>user\n', add_special_tokens=False
    )
    suffix = tokenizer.encode(
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False
    )

    def score(doc: dict) -> float:
        text = f"<Instruct>: {INSTRUCTION}\n<Query>: {fixture['query_text']}\n<Document>: {doc['text']}"
        encoded = tokenizer(
            text, truncation=True, max_length=1024 - len(prefix) - len(suffix),
            return_attention_mask=False,
        )
        batch = tokenizer.pad({"input_ids": [prefix + encoded["input_ids"] + suffix]},
                              padding=True, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**batch).logits[:, -1, :]
        return float(torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1)
                     .log_softmax(dim=1)[:, 1].exp()[0])

    profiles = {}
    for thread_count in (4, 16):
        torch.set_num_threads(thread_count)
        os.environ["OMP_NUM_THREADS"] = str(thread_count)
        os.environ["MKL_NUM_THREADS"] = str(thread_count)
        os.environ["OPENBLAS_NUM_THREADS"] = str(thread_count)
        first = [score(doc) for doc in fixture["documents"]]
        second = [score(doc) for doc in fixture["documents"]]
        if first != second:
            raise RuntimeError(f"{thread_count}-thread fixture repeat was not exactly equal")
        doc_ids = [doc["doc_id"] for doc in fixture["documents"]]
        order = sorted(range(len(first)), key=lambda i: (-first[i], doc_ids[i]))
        profiles[str(thread_count)] = {
            "scores": first,
            "ranking": [doc_ids[i] for i in order],
            "repeat_equal": first == second,
        }
    reference = profiles["4"]
    candidate = profiles["16"]
    deltas = [abs(a - b) for a, b in zip(reference["scores"], candidate["scores"])]
    max_delta = max(deltas, default=0.0)
    if reference["ranking"] != candidate["ranking"]:
        raise RuntimeError("4-vs-16 Moon canary changed fixture ranking")
    if max_delta > 1e-6:
        raise RuntimeError(f"4-vs-16 Moon canary score delta exceeded 1e-6: {max_delta}")

    return {
        "schema_version": 1, "manifest_id": "sem2act-v5-moon-host-preflight-v1",
        "status": "pass", "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution_sha": args.expected_sha, "execution_branch": branch,
        "protocol_hash": PROTOCOL_HASH, "runtime_lock_sha256": sha(ROOT / LOCK_REL),
        "runtime_freeze_sha256": sha(ROOT / FREEZE_REL),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "machine": platform.machine(), "logical_vcpus": os.cpu_count(),
        "memory_bytes": total_memory, "cpu_flags_records": len(per_cpu_flags),
        "cpu_flags_sha256": hashlib.sha256(json.dumps(
            [sorted(flags) for flags in per_cpu_flags], separators=(",", ":")
        ).encode()).hexdigest(),
        "avx2_masked": all("avx2" not in flags for flags in per_cpu_flags),
        "avx512f_masked": all("avx512f" not in flags for flags in per_cpu_flags),
        "torch": torch.__version__, "torch_cuda_available": bool(torch.cuda.is_available()),
        "transformers": transformers.__version__, "scratch_root": str(scratch),
        "scratch_free_bytes": scratch_free, "output_filesystem": filesystem,
        "output_filesystem_free_bytes": shutil.disk_usage(output_root.parent).free,
        "nfs_quota_probe_bytes_written_and_removed": quota_probe_bytes,
        "input_manifest_sha256": inputs["manifest_sha256"], "input_files": inputs["files"],
        "model_snapshot": model_snapshot, "model_snapshot_sha256": hashlib.sha256(
            json.dumps(model_snapshot, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "qrels_read": False,
    }, {
        "schema_version": 1, "manifest_id": "sem2act-v5-moon-reranker-thread-parity-v1",
        "status": "pass", "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution_sha": args.expected_sha, "protocol_hash": PROTOCOL_HASH,
        "fixture_sha256": sha(fixture_path), "model_revision": MODEL_REVISION,
        "runtime_lock_sha256": sha(ROOT / LOCK_REL), "profiles": profiles,
        "exact_repeat_equality": True, "ranking_equal": True,
        "max_absolute_score_delta": max_delta, "score_tolerance": 1e-6,
        "authorized_threads": 16, "device": "cpu", "dtype": "float32",
        "qrels_read": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if len(args.expected_sha) != 40 or any(c not in "0123456789abcdef" for c in args.expected_sha):
        parser.error("--expected-sha must be a lowercase full Git SHA")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["HF_HOME"] = "/tmp/sem2act-v5/cache/huggingface"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/tmp/sem2act-v5/cache/huggingface/hub"
    os.environ["TORCH_HOME"] = "/tmp/sem2act-v5/cache/torch"

    branch = require_checkout(args.expected_sha)
    freeze = subprocess.run(
        [sys.executable, str(ROOT / "scripts/freeze_v5_moon_runtime.py"), "--check"],
        cwd=ROOT, check=False,
    )
    if freeze.returncode != 0:
        raise SystemExit("Moon static runtime freeze check failed")
    input_root = args.input_root.resolve()
    model_path = args.model_path.resolve()
    inputs = validate_inputs(input_root)
    model_snapshot = validate_model(model_path)
    output_root = args.output_root.resolve()
    try:
        host, canary = host_preflight(args, branch, inputs, model_snapshot)
    except Exception as exc:
        if not output_root.exists() or not any(output_root.iterdir()):
            try:
                output_root.parent.mkdir(parents=True, exist_ok=True)
                if nfs_type(output_root.parent).startswith("nfs"):
                    output_root.mkdir(parents=True, exist_ok=True)
                    blocked = {
                        "schema_version": 1,
                        "manifest_id": "sem2act-v5-moon-reranker-pilot-blocked-v1",
                        "status": "blocked-before-lockbox-scoring",
                        "created_utc": datetime.now(timezone.utc).isoformat(),
                        "execution_sha": args.expected_sha,
                        "protocol_hash": PROTOCOL_HASH,
                        "runtime_lock_sha256": sha(ROOT / LOCK_REL),
                        "runtime_freeze_sha256": sha(ROOT / FREEZE_REL),
                        "error": f"{type(exc).__name__}: {exc}",
                        "result_bearing_execution_started": False,
                        "qrels_read": False,
                    }
                    (output_root / "pilot_blocked.json").write_text(
                        json.dumps(blocked, indent=2, sort_keys=True) + "\n"
                    )
            except Exception:
                pass
        raise SystemExit(f"Moon pilot stopped before lockbox scoring: {type(exc).__name__}: {exc}") from exc
    output_root.mkdir(parents=True, exist_ok=True)
    host_path = output_root / "moon_preflight.json"
    canary_path = output_root / "moon_reranker_canary.json"
    host_path.write_text(json.dumps(host, indent=2, sort_keys=True) + "\n")
    canary_path.write_text(json.dumps(canary, indent=2, sort_keys=True) + "\n")

    scratch = Path("/tmp/sem2act-v5").resolve()
    shard_root = scratch / "pilot-reranker-000-shards"
    if shard_root.exists():
        raise SystemExit(f"refusing to overwrite existing shard plan: {shard_root}")
    plan = subprocess.run([
        sys.executable, str(ROOT / "scripts/plan_v5_cpu_rerank_shards.py"),
        "--queries", str(input_root / "queries.jsonl"),
        "--corpus", str(input_root / "corpus.jsonl"),
        "--candidates", str(input_root / "rerank_candidates.jsonl"),
        "--queries-per-shard", "20", "--output-root", str(shard_root),
    ], cwd=ROOT, check=False)
    if plan.returncode != 0:
        raise SystemExit("could not plan the frozen 20-query reranker shards")
    shard = json.loads((shard_root / "shard-000.json").read_text())
    if shard.get("query_ids") != [f"v5-lb-q{i:04d}" for i in range(1, 21)]:
        raise SystemExit("shard 000 does not match the frozen first 20 query IDs")

    authorization = {
        "schema_version": 1, "manifest_id": "sem2act-v5-moon-reranker-pilot-authorization-v1",
        "status": "authorized-after-gates", "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution_sha": args.expected_sha, "execution_branch": branch,
        "protocol_hash": PROTOCOL_HASH, "runtime_lock_sha256": sha(ROOT / LOCK_REL),
        "runtime_freeze_sha256": sha(ROOT / FREEZE_REL), "thread_count": 16,
        "parity_status": "pass",
        "host_preflight": {"path": str(host_path), "sha256": sha(host_path)},
        "thread_canary": {"path": str(canary_path), "sha256": sha(canary_path)},
        "input_manifest_sha256": inputs["manifest_sha256"],
        "input_files": inputs["files"],
        "model_snapshot_sha256": host["model_snapshot_sha256"],
        "pilot_shard_sha256": sha(shard_root / "shard-000.json"),
        "result_bearing_execution_started": False,
        "qrels_read": False,
    }
    authorization_path = output_root / "pilot_authorization.json"
    authorization_path.write_text(json.dumps(authorization, indent=2, sort_keys=True) + "\n")

    run = subprocess.run([
        sys.executable, str(ROOT / "scripts/run_v5_cpu_rerank_shard.py"),
        "--moon-pilot-manifest", str(authorization_path),
        "--shard", str(shard_root / "shard-000.json"),
        "--queries", str(input_root / "queries.jsonl"),
        "--corpus", str(input_root / "corpus.jsonl"),
        "--candidates", str(input_root / "rerank_candidates.jsonl"),
        "--model-path", str(model_path), "--output-dir", str(output_root),
    ], cwd=ROOT, check=False)
    run_manifest = output_root / "run_manifest.json"
    rankings = output_root / "rankings.jsonl"
    pass_status = run.returncode == 0 and run_manifest.is_file() and rankings.is_file()
    final = {
        "schema_version": 1, "manifest_id": "sem2act-v5-moon-reranker-pilot-v1",
        "status": "pass" if pass_status else "failed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution_sha": args.expected_sha, "execution_branch": branch,
        "protocol_hash": PROTOCOL_HASH, "runtime_lock_sha256": sha(ROOT / LOCK_REL),
        "runtime_freeze_sha256": sha(ROOT / FREEZE_REL),
        "host_preflight_sha256": sha(host_path), "thread_canary_sha256": sha(canary_path),
        "pilot_authorization_sha256": sha(authorization_path),
        "input_manifest_sha256": inputs["manifest_sha256"], "input_files": inputs["files"],
        "model_snapshot_sha256": host["model_snapshot_sha256"],
        "shard_id": shard["manifest_id"], "expected_queries": 20,
        "expected_candidate_scores": 1000, "qrels_read": False,
        "result_bearing_execution_started": (output_root / "result_bearing_started.json").is_file(),
        "outputs": {
            path.name: sha(path) for path in (
                host_path, canary_path, authorization_path, rankings,
                output_root / "failures.json", run_manifest,
                output_root / "result_bearing_started.json",
            ) if path.is_file()
        },
        "stop_after_this_pilot": True,
        "next_action": "return evidence to Codex; do not run another shard",
        "launcher_exit_code": run.returncode,
    }
    (output_root / "pilot_manifest.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    print(json.dumps(final, indent=2, sort_keys=True), flush=True)
    return 0 if pass_status else (run.returncode or 1)


if __name__ == "__main__":
    raise SystemExit(main())
