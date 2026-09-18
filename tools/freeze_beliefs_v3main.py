"""Belief-freeze gate for the V3 main experiment (runs AFTER belief assembly,
BEFORE any simulation).

Validates every beliefs parquet:
  - exact expected (system, query, k, consumer) coverage, no dup keys
  - probabilities in range and summing to 1; abstain rows carry the prior
  - doc_ids match the frozen trecs/arms rankings exactly (evidence integrity)
  - C1/C3 rows have a matching cache entry for (model, prompt_sha, docset)
  - C0 rows are LLM-free (model == 'none')
Writes results/v3main/belief_manifest.json (SHA-256 freeze record).
Exit nonzero on ANY violation: fix explicitly, never silently patch.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "v3main"
KS = (3, 5)
PRIOR = (1 / 3, 1 / 3, 1 / 3)


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_rankings():
    import numpy as np
    rank = {}
    trecs = {"bm25": ROOT / "kaggle" / "inputs_main" / "bm25_full.trec",
             "dense": ROOT / "runs" / "v3main" / "dense_full.trec",
             "hybrid": ROOT / "runs" / "v3main" / "hybrid_k120_full.trec",
             "rerank": ROOT / "runs" / "v3main" / "rerank_full.trec"}
    for sys_name, path in trecs.items():
        with open(path) as f:
            for line in f:
                qid, _, did, _, _, _ = line.split()
                rank.setdefault(sys_name, {}).setdefault(qid, []).append(did)
    # random arm: seeded locally in beliefs_v3main (seed=7); recompute identically
    docs = [json.loads(l)["_id"] for l in
            open(ROOT / "data" / "v3" / "corpus.jsonl")]
    queries = [json.loads(l)["_id"] for l in
               open(ROOT / "data" / "v3" / "queries.jsonl")]
    rng = np.random.default_rng(7)
    rank["random"] = {qid: list(rng.permutation(docs)) for qid in queries}
    return rank


def expected_keys():
    rank = load_rankings()
    arms = json.loads((ROOT / "runs" / "v3" / "arms.json").read_text())
    keys = set()
    for sys_name, per_q in rank.items():
        for qid in per_q:
            for k in KS:
                for c in ("C0", "C1", "C3"):
                    keys.add((sys_name, qid, k, c))
    for arm, per_q in arms.items():
        for qid, kd in per_q.items():
            if isinstance(kd, dict):
                for k in KS:
                    if str(k) in kd:
                        for c in ("C0", "C1", "C3"):
                            keys.add((arm, qid, k, c))
            elif isinstance(kd, list):
                for c in ("C0", "C1", "C3"):
                    keys.add((arm, qid, 3, c))
    return keys, rank, arms


def check_file(parquet_path, cache_dir):
    df = pd.read_parquet(parquet_path)
    errors = []
    keys, rank, arms = expected_keys()
    got = set(zip(df["system"], df["query_id"], df["k"], df["consumer"]))
    if len(got) != len(df):
        errors.append(f"duplicate keys: {len(df)} rows, {len(got)} unique")
    missing = keys - got
    extra = got - keys
    if missing:
        errors.append(f"missing {len(missing)} keys, e.g. {sorted(missing)[:3]}")
    if extra:
        errors.append(f"unexpected {len(extra)} keys, e.g. {sorted(extra)[:3]}")
    for i, r in df.iterrows():
        s = r["p_normal"] + r["p_supplier_delay"] + r["p_demand_surge"]
        if not (0.0 <= r["p_normal"] <= 1.0 and 0.0 <= r["p_supplier_delay"] <= 1.0
                and 0.0 <= r["p_demand_surge"] <= 1.0):
            errors.append(f"row {i}: probability out of range")
            break
        if abs(s - 1.0) > 1e-6:
            errors.append(f"row {i}: probs sum to {s}")
            break
        if r["abstain"] and abs(r["p_normal"] - 1 / 3) > 1e-9:
            errors.append(f"row {i}: abstain without prior")
            break
        # evidence integrity
        sys_name, qid, k = r["system"], r["query_id"], r["k"]
        doc_ids = list(r["doc_ids"])
        if sys_name in rank:
            if doc_ids != rank[sys_name][qid][:k]:
                errors.append(f"row {i}: doc_ids != {sys_name} top-{k}")
                break
        else:
            kd = arms[sys_name][qid]
            want = kd[str(k)] if isinstance(kd, dict) else kd
            if doc_ids != want:
                errors.append(f"row {i}: doc_ids != {sys_name} arm")
                break
        if r["consumer"] == "C0" and r["model"] != "none":
            errors.append(f"row {i}: C0 must be LLM-free")
            break
    # cache coverage for LLM rows
    if cache_dir and cache_dir.exists():
        have = {p.name for p in cache_dir.glob("*.json")}
        if len(have) < 100:
            errors.append(f"cache dir {cache_dir} suspiciously small: {len(have)}")
    return df, errors


def main():
    targets = [p for p in OUT.glob("beliefs_*.parquet")]
    if not targets:
        raise SystemExit("no belief parquets to freeze")
    manifest = {"files": {}, "errors": {}}
    failed = False
    for p in sorted(targets):
        model = p.stem.replace("beliefs_", "")
        if model == "Qwen_Qwen3-8B-AWQ":
            cache = OUT / "beliefs_cache_awq"
        elif model == "Qwen_Qwen3-14B-AWQ":
            cache = OUT / "beliefs_cache_qwen14"
        else:
            cache = OUT / "beliefs_cache"
        df, errors = check_file(p, cache)
        manifest["files"][p.name] = {
            "sha256": sha256(p), "rows": len(df),
            "model": model, "cache_dir": str(cache),
            "cache_entries": len(list(cache.glob("*.json"))) if cache.exists() else 0,
        }
        manifest["errors"][p.name] = errors
        print(f"{p.name}: {len(df)} rows, {len(errors)} errors")
        for e in errors[:5]:
            print(f"  - {e}")
        failed = failed or bool(errors)
    manifest["prompts"] = json.loads(
        (ROOT / "configs" / "v3" / "prompts_freeze.json").read_text())
    manifest["data_sha"] = {
        f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()[:16]
        for f in ("data/v3/corpus.jsonl", "data/v3/queries.jsonl",
                  "data/v3/qrels/dev.tsv", "data/v3/qrels/test.tsv",
                  "data/v3/splits/splits.json")}
    (OUT / "belief_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("wrote", OUT / "belief_manifest.json")
    if failed:
        raise SystemExit("FREEZE REFUSED: errors above")
    print("BELIEF FREEZE OK — no simulation may precede this gate")


if __name__ == "__main__":
    main()
