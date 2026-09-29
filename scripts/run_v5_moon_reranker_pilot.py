"""Moon's single-shard reranker pilot with a host and 4-vs-16 parity gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
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
QUERY_CHUNK_SIZE = 1
CHILD_CPU_SOFT_SECONDS = 540
CHILD_CPU_HARD_SECONDS = 600
ESTIMATED_CPU_SECONDS_PER_PAIR = 8.99
PILOT_CANDIDATE_DEPTH = 50
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
        raise RuntimeError(f"checkout SHA mismatch: expected {expected_sha}, got {head}")
    if status:
        raise RuntimeError("checkout is dirty; refusing Moon pilot")
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
    if (set(candidate_map) != qids
            or any(len(ids) != PILOT_CANDIDATE_DEPTH
                   or len(set(ids)) != PILOT_CANDIDATE_DEPTH
                   for ids in candidate_map.values())):
        raise RuntimeError("reranker candidate coverage/depth differs from the frozen protocol")
    corpus_ids = {
        json.loads(line)["_id"]
        for line in (input_root / "corpus.jsonl").read_text().splitlines() if line
    }
    if any(doc_id not in corpus_ids for ids in candidate_map.values() for doc_id in ids):
        raise RuntimeError("reranker candidate references a document missing from the corpus")
    return {"manifest_sha256": sha(ROOT / INPUT_MANIFEST_REL), "files": actual,
            "query_count": len(queries), "candidate_depth": PILOT_CANDIDATE_DEPTH}


def validate_model(model_path: Path, staging_manifest_path: Path) -> tuple[dict, str]:
    expected_manifest = json.loads((ROOT / MODEL_REFERENCE_REL).read_text())
    if (expected_manifest.get("status") != "pass"
            or expected_manifest.get("model_id") != MODEL_ID
            or expected_manifest.get("revision") != MODEL_REVISION):
        raise RuntimeError("committed reranker model snapshot reference is not accepted")
    expected_snapshot = expected_manifest["model_snapshot"]
    actual_snapshot = snapshot(model_path)
    if actual_snapshot != expected_snapshot:
        raise RuntimeError("local reranker files do not match the pinned accepted model snapshot")
    staging = json.loads(staging_manifest_path.read_text())
    if (staging.get("status") != "pass"
            or staging.get("manifest_id") != "sem2act-v5-moon-reranker-model-staging-v1"
            or staging.get("model_id") != MODEL_ID
            or staging.get("revision") != MODEL_REVISION
            or staging.get("model_path") != str(model_path)
            or staging.get("reference_manifest_sha256") != sha(ROOT / MODEL_REFERENCE_REL)
            or staging.get("allow_patterns") != [entry["path"] for entry in expected_snapshot["files"]]
            or staging.get("model_snapshot") != expected_snapshot
            or staging.get("model_snapshot_sha256") != expected_snapshot["files_sha256"]):
        raise RuntimeError("model staging manifest does not attest the accepted 12-file snapshot")
    return actual_snapshot, sha(staging_manifest_path)


def cpu_child_limits(parent_soft: int, parent_hard: int) -> tuple[int, int]:
    """Return the fixed safe child limits, refusing a host with tighter limits."""
    infinity = resource.RLIM_INFINITY
    effective_soft = CHILD_CPU_SOFT_SECONDS if parent_soft == infinity else int(parent_soft)
    effective_hard = CHILD_CPU_HARD_SECONDS if parent_hard == infinity else int(parent_hard)
    if effective_soft < CHILD_CPU_SOFT_SECONDS or effective_hard < CHILD_CPU_HARD_SECONDS:
        raise RuntimeError(
            "Moon CPU-time limit is tighter than the authorized per-query subprocess policy: "
            f"soft={effective_soft}, hard={effective_hard}"
        )
    return CHILD_CPU_SOFT_SECONDS, CHILD_CPU_HARD_SECONDS


def install_child_cpu_limits() -> None:
    current_soft, current_hard = resource.getrlimit(resource.RLIMIT_CPU)
    soft, hard = cpu_child_limits(current_soft, current_hard)
    resource.setrlimit(resource.RLIMIT_CPU, (soft, hard))


def query_chunks(query_ids: list[str]) -> list[tuple[str, list[str]]]:
    if QUERY_CHUNK_SIZE != 1:
        raise RuntimeError("Moon CPU safety policy requires one complete query per subprocess")
    return [
        (f"query-{index:04d}", query_ids[index:index + QUERY_CHUNK_SIZE])
        for index in range(0, len(query_ids), QUERY_CHUNK_SIZE)
    ]


def deterministic_rankings(query_ids: list[str], candidate_map: dict[str, list[str]],
                           records: list[dict]) -> bytes:
    actual_ids = [record.get("query_id") for record in records]
    if actual_ids != query_ids:
        raise RuntimeError("chunk ranking query order has a duplicate, omission, or permutation")
    expected_pairs = Counter()
    observed_pairs = Counter()
    for qid in query_ids:
        if qid not in candidate_map:
            raise RuntimeError(f"authorized query has no frozen candidates: {qid}")
        candidates = candidate_map[qid]
        if (len(candidates) != PILOT_CANDIDATE_DEPTH
                or len(set(candidates)) != PILOT_CANDIDATE_DEPTH):
            raise RuntimeError(f"frozen candidate set is not 50 unique documents: {qid}")
        expected_pairs.update((qid, doc_id) for doc_id in candidates)
    for qid, record in zip(query_ids, records):
        if record.get("key") != f"reranker|{qid}":
            raise RuntimeError(f"chunk ranking key mismatch for {qid}")
        ranking = record.get("ranking")
        if (not isinstance(ranking, list) or len(ranking) != PILOT_CANDIDATE_DEPTH
                or len(set(ranking)) != PILOT_CANDIDATE_DEPTH
                or set(ranking) != set(candidate_map[qid])):
            raise RuntimeError(f"expected candidate pair coverage failed for {qid}")
        observed_pairs.update((qid, doc_id) for doc_id in ranking)
    if observed_pairs != expected_pairs:
        raise RuntimeError("pilot pair coverage has a duplicate or omission")
    return b"".join(
        (json.dumps(record, sort_keys=True) + "\n").encode() for record in records
    )


def write_blocked(output_root: Path, expected_sha: str, stage: str,
                  exc: Exception) -> str | None:
    marker = output_root / "result_bearing_started.json"
    if marker.exists():
        return None
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        blocked_path = output_root / "pilot_blocked.json"
        if blocked_path.exists():
            return sha(blocked_path)
        blocked = {
            "schema_version": 1,
            "manifest_id": "sem2act-v5-moon-reranker-pilot-blocked-v1",
            "status": "blocked-before-lockbox-scoring",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "execution_sha": expected_sha,
            "protocol_hash": PROTOCOL_HASH,
            "runtime_lock_sha256": sha(ROOT / LOCK_REL),
            "runtime_freeze_sha256": sha(ROOT / FREEZE_REL),
            "blocked_stage": stage,
            "error": f"{type(exc).__name__}: {exc}",
            "result_bearing_execution_started": False,
            "qrels_read": False,
        }
        blocked_path.write_text(json.dumps(blocked, indent=2, sort_keys=True) + "\n")
        return sha(blocked_path)
    except OSError as write_error:
        print(f"could not persist pilot_blocked.json: {write_error}", file=sys.stderr)
        return None


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
    cpu_limit_soft, cpu_limit_hard = resource.getrlimit(resource.RLIMIT_CPU)
    cpu_child_limits(cpu_limit_soft, cpu_limit_hard)
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
        "process_cpu_limit_soft_seconds": (
            None if cpu_limit_soft == resource.RLIM_INFINITY else cpu_limit_soft
        ),
        "process_cpu_limit_hard_seconds": (
            None if cpu_limit_hard == resource.RLIM_INFINITY else cpu_limit_hard
        ),
        "moon_query_chunk_policy": {
            "queries_per_subprocess": QUERY_CHUNK_SIZE,
            "subprocesses_expected": 20,
            "candidate_depth": PILOT_CANDIDATE_DEPTH,
            "estimated_scoring_cpu_seconds_per_pair": ESTIMATED_CPU_SECONDS_PER_PAIR,
            "estimated_scoring_cpu_seconds_per_query": (
                ESTIMATED_CPU_SECONDS_PER_PAIR * PILOT_CANDIDATE_DEPTH
            ),
            "subprocess_soft_limit_seconds": CHILD_CPU_SOFT_SECONDS,
            "subprocess_hard_limit_seconds": CHILD_CPU_HARD_SECONDS,
        },
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
    parser.add_argument("--model-staging-manifest", type=Path, required=True)
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
    input_root = args.input_root.resolve()
    model_path = args.model_path.resolve()
    output_root = args.output_root.resolve()
    safe_to_write_blocker = True
    if output_root.exists():
        safe_to_write_blocker = output_root.is_dir() and not any(output_root.iterdir())
    stage = "checkout"
    try:
        if not safe_to_write_blocker:
            raise RuntimeError(f"refusing to overwrite non-empty pilot output directory: {output_root}")
        branch = require_checkout(args.expected_sha)
        stage = "runtime-freeze"
        freeze = subprocess.run(
            [sys.executable, str(ROOT / "scripts/freeze_v5_moon_runtime.py"), "--check"],
            cwd=ROOT, check=False,
        )
        if freeze.returncode != 0:
            raise RuntimeError("Moon static runtime freeze check failed")
        stage = "input-validation"
        inputs = validate_inputs(input_root)
        stage = "model-snapshot-validation"
        model_snapshot, staging_manifest_sha256 = validate_model(
            model_path, args.model_staging_manifest.resolve()
        )
        stage = "Moon-host-and-parity-preflight"
        host, canary = host_preflight(args, branch, inputs, model_snapshot)
        output_root.mkdir(parents=True, exist_ok=True)
        host_path = output_root / "moon_preflight.json"
        canary_path = output_root / "moon_reranker_canary.json"
        host_path.write_text(json.dumps(host, indent=2, sort_keys=True) + "\n")
        canary_path.write_text(json.dumps(canary, indent=2, sort_keys=True) + "\n")

        stage = "pilot-shard-planning"
        scratch = Path("/tmp/sem2act-v5").resolve()
        shard_root = scratch / f"{output_root.name}-shards"
        if shard_root.exists():
            raise RuntimeError(f"refusing to overwrite existing shard plan: {shard_root}")
        plan = subprocess.run([
            sys.executable, str(ROOT / "scripts/plan_v5_cpu_rerank_shards.py"),
            "--queries", str(input_root / "queries.jsonl"),
            "--corpus", str(input_root / "corpus.jsonl"),
            "--candidates", str(input_root / "rerank_candidates.jsonl"),
            "--queries-per-shard", "20",
            "--resume-policy", "one-shot-empty-output-only",
            "--output-root", str(shard_root),
        ], cwd=ROOT, check=False)
        if plan.returncode != 0:
            raise RuntimeError("could not plan the frozen 20-query reranker shard")
        shard_path = shard_root / "shard-000.json"
        shard = json.loads(shard_path.read_text())
        authorized_qids = [f"v5-lb-q{i:04d}" for i in range(1, 21)]
        if (shard.get("query_ids") != authorized_qids
                or shard.get("expected_query_keys") != [f"reranker|{qid}" for qid in authorized_qids]
                or shard.get("resume_policy") != "one-shot-empty-output-only"):
            raise RuntimeError("shard 000 does not match the frozen first 20 query IDs and keys")
        chunks = query_chunks(authorized_qids)
        if len(chunks) != 20 or any(len(qids) != 1 for _, qids in chunks):
            raise RuntimeError("Moon subprocess plan is not 20 deterministic one-query chunks")

        authorization = {
            "schema_version": 1,
            "manifest_id": "sem2act-v5-moon-reranker-pilot-authorization-v1",
            "status": "authorized-after-gates",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "execution_sha": args.expected_sha,
            "execution_branch": branch,
            "protocol_hash": PROTOCOL_HASH,
            "runtime_lock_sha256": sha(ROOT / LOCK_REL),
            "runtime_freeze_sha256": sha(ROOT / FREEZE_REL),
            "thread_count": 16,
            "parity_status": "pass",
            "host_preflight": {"path": str(host_path), "sha256": sha(host_path)},
            "thread_canary": {"path": str(canary_path), "sha256": sha(canary_path)},
            "input_manifest_sha256": inputs["manifest_sha256"],
            "input_files": inputs["files"],
            "model_snapshot_sha256": host["model_snapshot_sha256"],
            "model_staging_manifest_sha256": staging_manifest_sha256,
            "pilot_shard_sha256": sha(shard_path),
            "authorized_query_ids": authorized_qids,
            "candidate_depth": PILOT_CANDIDATE_DEPTH,
            "subprocess_query_chunk_size": QUERY_CHUNK_SIZE,
            "resume_policy": "one-shot-empty-output-only",
            "result_bearing_execution_started": False,
            "qrels_read": False,
        }
        authorization_path = output_root / "pilot_authorization.json"
        authorization_path.write_text(json.dumps(authorization, indent=2, sort_keys=True) + "\n")
        started_path = output_root / "result_bearing_started.json"
    except Exception as exc:
        blocker_sha = write_blocked(output_root, args.expected_sha, stage, exc) if safe_to_write_blocker else None
        print(
            f"Moon pilot stopped before lockbox scoring at {stage}: "
            f"{type(exc).__name__}: {exc}"
            + (f"; pilot_blocked.json sha256={blocker_sha}" if blocker_sha else ""),
            file=sys.stderr,
        )
        return 2

    candidate_rows = [
        json.loads(line) for line in (input_root / "rerank_candidates.jsonl").read_text().splitlines()
        if line
    ]
    candidate_map = {row["query_id"]: row["doc_ids"] for row in candidate_rows}
    expected_qids = [qid for _, qids in chunks for qid in qids]
    chunk_records = []
    ranking_records = []
    failures = []
    for chunk_index, (chunk_id, chunk_qids) in enumerate(chunks):
        qid = chunk_qids[0]
        chunk_dir = output_root / "chunks" / chunk_id
        record = {
            "chunk_id": chunk_id,
            "query_ids": chunk_qids,
            "expected_pairs": len(candidate_map[qid]),
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            chunk_dir.mkdir(parents=True, exist_ok=False)
            stdout_path = chunk_dir / "stdout.log"
            stderr_path = chunk_dir / "stderr.log"
            command = [
                sys.executable, str(ROOT / "scripts/run_v5_cpu_rerank_shard.py"),
                "--moon-pilot-manifest", str(authorization_path),
                "--shard", str(shard_path),
                "--queries", str(input_root / "queries.jsonl"),
                "--corpus", str(input_root / "corpus.jsonl"),
                "--candidates", str(input_root / "rerank_candidates.jsonl"),
                "--model-path", str(model_path),
                "--output-dir", str(chunk_dir),
                "--moon-query-id", qid,
            ]
            record["command"] = command
            record["cpu_limit_soft_seconds"] = CHILD_CPU_SOFT_SECONDS
            record["cpu_limit_hard_seconds"] = CHILD_CPU_HARD_SECONDS
            record["started_utc"] = datetime.now(timezone.utc).isoformat()
            with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
                child = subprocess.run(
                    command, cwd=ROOT, check=False,
                    stdout=stdout_handle, stderr=stderr_handle,
                    preexec_fn=install_child_cpu_limits,
                )
            record["return_code"] = child.returncode
            record["completed_utc"] = datetime.now(timezone.utc).isoformat()
            child_started_path = chunk_dir / "result_bearing_started.json"
            if child_started_path.is_file() and not started_path.exists():
                child_started = json.loads(child_started_path.read_text())
                started_path.write_text(json.dumps({
                    "schema_version": 1,
                    "manifest_id": "sem2act-v5-moon-reranker-pilot-started-v1",
                    "status": "started",
                    "created_utc": child_started["created_utc"],
                    "execution_sha": args.expected_sha,
                    "protocol_hash": PROTOCOL_HASH,
                    "pilot_shard_sha256": sha(shard_path),
                    "first_chunk_id": chunk_id,
                    "first_query_id": qid,
                    "expected_queries": len(expected_qids),
                    "expected_pairs": len(expected_qids) * PILOT_CANDIDATE_DEPTH,
                    "subprocess_query_chunk_size": QUERY_CHUNK_SIZE,
                    "subprocesses_expected": len(chunks),
                    "first_child_marker_sha256": sha(child_started_path),
                    "qrels_read": False,
                }, indent=2, sort_keys=True) + "\n")
            record["stdout_sha256"] = sha(stdout_path)
            record["stderr_sha256"] = sha(stderr_path)
            record["run_manifest_sha256"] = (
                sha(chunk_dir / "run_manifest.json")
                if (chunk_dir / "run_manifest.json").is_file() else None
            )
            record["failures_sha256"] = (
                sha(chunk_dir / "failures.json")
                if (chunk_dir / "failures.json").is_file() else None
            )
            record["result_bearing_started_sha256"] = (
                sha(chunk_dir / "result_bearing_started.json")
                if (chunk_dir / "result_bearing_started.json").is_file() else None
            )
            if child.returncode != 0:
                raise RuntimeError(f"query subprocess exited with code {child.returncode}")
            run_manifest_path = chunk_dir / "run_manifest.json"
            failures_path = chunk_dir / "failures.json"
            rankings_path = chunk_dir / "rankings.jsonl"
            if not all(path.is_file() for path in (run_manifest_path, failures_path, rankings_path)):
                raise RuntimeError("query subprocess omitted a required output")
            child_manifest = json.loads(run_manifest_path.read_text())
            child_failures = json.loads(failures_path.read_text())
            if (child_manifest.get("status") != "pass"
                    or child_manifest.get("execution_sha") != args.expected_sha
                    or child_manifest.get("runtime_profile") != "moon-reranker-only-v1"
                    or child_manifest.get("runtime_lock_sha256") != authorization["runtime_lock_sha256"]
                    or child_manifest.get("runtime_freeze_sha256") != authorization["runtime_freeze_sha256"]
                    or child_manifest.get("host_preflight_sha256") != authorization["host_preflight"]["sha256"]
                    or child_manifest.get("thread_canary_sha256") != authorization["thread_canary"]["sha256"]
                    or child_manifest.get("model_id") != MODEL_ID
                    or child_manifest.get("revision") != MODEL_REVISION
                    or child_manifest.get("dtype") != "float32"
                    or child_manifest.get("device") != "cpu"
                    or child_manifest.get("thread_count") != 16
                    or child_manifest.get("inter_op_threads") != 1
                    or child_manifest.get("chunk_query_id") != qid
                    or child_manifest.get("expected_query_ids") != [qid]
                    or child_manifest.get("expected_pairs") != PILOT_CANDIDATE_DEPTH
                    or child_manifest.get("accepted_queries") != 1
                    or child_manifest.get("accepted_pairs") != PILOT_CANDIDATE_DEPTH
                    or child_manifest.get("n_fail") != 0
                    or child_manifest.get("model_staging_manifest_sha256") != staging_manifest_sha256
                    or child_manifest.get("process_cpu_limit_soft_seconds") != CHILD_CPU_SOFT_SECONDS
                    or child_manifest.get("process_cpu_limit_hard_seconds") != CHILD_CPU_HARD_SECONDS
                    or child_manifest.get("process_cpu_seconds", CHILD_CPU_SOFT_SECONDS + 1)
                    > CHILD_CPU_SOFT_SECONDS
                    or child_failures):
                raise RuntimeError("query subprocess manifest/status/CPU budget did not pass")
            lines = [line for line in rankings_path.read_text().splitlines() if line]
            if len(lines) != 1:
                raise RuntimeError("query subprocess must produce exactly one ranking record")
            ranking_record = json.loads(lines[0])
            deterministic_rankings([qid], candidate_map, [ranking_record])
            record["process_cpu_seconds"] = child_manifest["process_cpu_seconds"]
            record["ranking_sha256"] = sha(rankings_path)
            record["chunk_outputs"] = {
                str(path.relative_to(chunk_dir)): sha(path)
                for path in sorted(chunk_dir.rglob("*")) if path.is_file()
            }
            ranking_records.append(ranking_record)
            record["status"] = "pass"
        except Exception as exc:
            record["completed_utc"] = datetime.now(timezone.utc).isoformat()
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            failures.append({"chunk_id": chunk_id, "query_id": qid, "error": record["error"]})
        chunk_records.append(record)
        if record["status"] != "pass":
            break

    expected_pairs = len(expected_qids) * PILOT_CANDIDATE_DEPTH
    rankings_path = output_root / "rankings.jsonl"
    concatenation_sha256 = None
    if not failures and len(ranking_records) == len(expected_qids):
        try:
            combined = deterministic_rankings(expected_qids, candidate_map, ranking_records)
            repeated = deterministic_rankings(expected_qids, candidate_map, ranking_records)
            if combined != repeated:
                raise RuntimeError("chunk output concatenation was not deterministic")
            rankings_path.write_bytes(combined)
            if rankings_path.read_bytes() != combined:
                raise RuntimeError("concatenated ranking bytes failed verification")
            concatenation_sha256 = sha(rankings_path)
        except Exception as exc:
            failures.append({"stage": "deterministic-concatenation", "error": f"{type(exc).__name__}: {exc}"})
    elif not failures:
        failures.append({"stage": "query-coverage", "error": "one or more authorized query subprocesses are missing"})

    if not started_path.exists() and failures:
        write_blocked(
            output_root, args.expected_sha, "query-subprocess-before-scoring",
            RuntimeError(failures[0].get("error", "query subprocess failed before scoring")),
        )

    pass_status = not failures and len(chunk_records) == 20 and len(ranking_records) == 20
    if pass_status:
        try:
            merged = [json.loads(line) for line in rankings_path.read_text().splitlines() if line]
            deterministic_rankings(expected_qids, candidate_map, merged)
            if len(merged) != 20 or len({record["query_id"] for record in merged}) != 20:
                raise RuntimeError("final output does not contain 20 unique authorized queries")
        except Exception as exc:
            pass_status = False
            failures.append({"stage": "final-coverage-verification", "error": f"{type(exc).__name__}: {exc}"})

    failure_path = output_root / "failures.json"
    failure_path.write_text(json.dumps(failures, indent=2, sort_keys=True) + "\n")
    run_manifest = {
        "schema_version": 1,
        "manifest_id": "sem2act-v5-moon-reranker-pilot-run-v1",
        "status": "pass" if pass_status else "failed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution_sha": args.expected_sha,
        "execution_branch": branch,
        "protocol_hash": PROTOCOL_HASH,
        "runtime_lock_sha256": sha(ROOT / LOCK_REL),
        "runtime_freeze_sha256": sha(ROOT / FREEZE_REL),
        "host_preflight_sha256": sha(host_path),
        "thread_canary_sha256": sha(canary_path),
        "pilot_authorization_sha256": sha(authorization_path),
        "input_manifest_sha256": inputs["manifest_sha256"],
        "input_files": inputs["files"],
        "model_snapshot_sha256": host["model_snapshot_sha256"],
        "model_staging_manifest_sha256": staging_manifest_sha256,
        "shard_id": shard["manifest_id"],
        "expected_query_ids": expected_qids,
        "expected_queries": len(expected_qids),
        "expected_pairs": expected_pairs,
        "accepted_queries": len(ranking_records),
        "accepted_pairs": len(ranking_records) * PILOT_CANDIDATE_DEPTH,
        "query_chunk_size": QUERY_CHUNK_SIZE,
        "pair_coverage_verified": pass_status,
        "deterministic_concatenation": {
            "verified_twice": bool(concatenation_sha256),
            "output_sha256": concatenation_sha256,
            "query_order": expected_qids,
        },
        "subprocesses": chunk_records,
        "outputs": {
            str(path.relative_to(output_root)): sha(path)
            for path in sorted(output_root.rglob("*"))
            if path.is_file() and path.name not in ("run_manifest.json", "pilot_manifest.json")
        },
        "failure_count": len(failures),
        "qrels_read": False,
        "result_bearing_execution_started": started_path.is_file(),
        "stop_after_this_pilot": True,
        "next_action": "return evidence to Codex; do not run another shard",
    }
    run_manifest_path = output_root / "run_manifest.json"
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")
    final = {
        "schema_version": 1,
        "manifest_id": "sem2act-v5-moon-reranker-pilot-v1",
        "status": "pass" if pass_status else "failed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution_sha": args.expected_sha,
        "execution_branch": branch,
        "protocol_hash": PROTOCOL_HASH,
        "runtime_lock_sha256": sha(ROOT / LOCK_REL),
        "runtime_freeze_sha256": sha(ROOT / FREEZE_REL),
        "host_preflight_sha256": sha(host_path),
        "thread_canary_sha256": sha(canary_path),
        "pilot_authorization_sha256": sha(authorization_path),
        "input_manifest_sha256": inputs["manifest_sha256"],
        "input_files": inputs["files"],
        "model_snapshot_sha256": host["model_snapshot_sha256"],
        "model_staging_manifest_sha256": staging_manifest_sha256,
        "shard_id": shard["manifest_id"],
        "expected_queries": len(expected_qids),
        "expected_candidate_scores": expected_pairs,
        "accepted_queries": len(ranking_records),
        "accepted_pairs": len(ranking_records) * PILOT_CANDIDATE_DEPTH,
        "pair_coverage_verified": pass_status,
        "deterministic_concatenation_sha256": concatenation_sha256,
        "subprocesses": chunk_records,
        "qrels_read": False,
        "result_bearing_execution_started": started_path.is_file(),
        "outputs": {
            "run_manifest.json": sha(run_manifest_path),
            "failures.json": sha(failure_path),
            "rankings.jsonl": sha(rankings_path) if rankings_path.is_file() else None,
            "result_bearing_started.json": sha(started_path) if started_path.is_file() else None,
            "pilot_blocked.json": sha(output_root / "pilot_blocked.json")
            if (output_root / "pilot_blocked.json").is_file() else None,
            "moon_preflight.json": sha(host_path),
            "moon_reranker_canary.json": sha(canary_path),
            "pilot_authorization.json": sha(authorization_path),
        },
        "stop_after_this_pilot": True,
        "next_action": "return evidence to Codex; do not run another shard",
    }
    (output_root / "pilot_manifest.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    print(json.dumps(final, indent=2, sort_keys=True), flush=True)
    return 0 if pass_status else 1


if __name__ == "__main__":
    raise SystemExit(main())
