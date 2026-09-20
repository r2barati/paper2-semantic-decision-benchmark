"""G2A-1 assembly: parse 180 pilot payloads -> C1 LOO belief rows (dev only).

Steps: extract zips from runs/v3main_loo -> parse every payload under the
FROZEN C1 normalization (probs/s; confidence rule byte-identical to
consume_c1) -> recompute cache key per (model,query,subset) and assert it
matches the payload filename -> assert dev-only query IDs -> assert ZERO
filename collisions with the live beliefs caches -> additive install ->
write extension_g2a/c1_loo_beliefs.parquet (240 rows: 40q x 3pos x 2models).

No test access: asserts query_id in dev_ids throughout.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.consumers_v3 import assert_frozen_prompts  # noqa: E402

assert_frozen_prompts()

OUT = ROOT / "results" / "v3main" / "extension_g2a"
RUNS = ROOT / "runs" / "v3main_loo"
CACHE = {"Qwen/Qwen3-8B-AWQ": ROOT / "results" / "v3main" / "beliefs_cache_awq",
         "Qwen/Qwen3-14B-AWQ": ROOT / "results" / "v3main" / "beliefs_cache_qwen14"}
MODELS = list(CACHE)

sp = json.load(open(ROOT / "data" / "v3" / "splits" / "splits.json"))
dev_ids = set(sp["dev"])
assert len(dev_ids) == 40


def _parse_json(raw: str):
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(raw)


def _c1_key(model: str, user: str) -> str:
    src = f"{model}||66e6890ba9c5464c||{hashlib.sha256(user.encode()).hexdigest()[:16]}"
    return hashlib.sha256(src.encode()).hexdigest()[:16] + ".json"


def main():
    docs = {}
    for line in open(ROOT / "data" / "v3" / "corpus.jsonl"):
        d = json.loads(line)
        docs[d["_id"]] = d["text"]
    qs = {}
    for line in open(ROOT / "kaggle" / "inputs_loo" / "queries_dev40.jsonl"):
        q = json.loads(line)
        assert q["metadata"].get("split") == "dev"
        qs[q["_id"]] = q["text"]
    spec = json.load(open(ROOT / "kaggle" / "inputs_loo" / "loo_sets.json"))
    assert set(spec["dev"]) == dev_ids and set(spec["sets"]) == dev_ids

    payloads = {}  # (model, filename) -> parsed
    for model in MODELS:
        tag = "8b" if model.endswith("8B-AWQ") else "14b"
        with zipfile.ZipFile(RUNS / f"cache_loo_dev_{tag}.zip") as z:
            for fn in z.namelist():
                p = json.loads(z.read(fn))
                assert p["model"] == model, fn
                assert p.get("prompt_sha") == "66e6890ba9c5464c", fn
                payloads[(model, fn)] = _parse_json(p["raw"])
    print(f"payloads parsed: {len(payloads)} (expect 180)")

    rows = []
    for model in MODELS:
        for qid in sorted(spec["sets"]):
            assert qid in dev_ids, f"non-dev query {qid} -- refusing"
            for pos in ("drop0", "drop1", "drop2"):
                dids = spec["sets"][qid][pos]
                user = (f"Operator information need: {qs[qid]}\n\nEvidence:\n" +
                        "\n\n".join(f"[DOC {d}] {docs[d]}" for d in dids))
                fn = _c1_key(model, user)
                probs = payloads[(model, fn)]
                # Frozen consume_c1 reads exactly these 3 keys and ignores the
                # rest (all 180 pilot payloads carry extra estimates, as usual).
                assert {"normal", "supplier_delay", "demand_surge"} <= set(probs), fn
                assert all(v >= 0 for v in probs.values()), fn
                s = probs["normal"] + probs["supplier_delay"] + probs["demand_surge"]
                assert s > 0, fn
                conf = min(1.0, s / 1.0) if s <= 1.5 else 0.5
                rows.append({"query_id": qid, "system": "rerank", "consumer": "C1",
                             "model": model, "k": 3, "n_docs": 2,
                             "ablation": pos, "doc_ids": list(dids),
                             "p_normal": probs["normal"]/s,
                             "p_supplier_delay": probs["supplier_delay"]/s,
                             "p_demand_surge": probs["demand_surge"]/s,
                             "abstain": False, "confidence": conf,
                             "split": "dev",
                             "cache_file": fn})
    assert len(rows) == 240, len(rows)
    # collision-free install (additive only; frozen entries never overwritten)
    staged = {}
    for model in MODELS:
        tag = "8b" if model.endswith("8B-AWQ") else "14b"
        with zipfile.ZipFile(RUNS / f"cache_loo_dev_{tag}.zip") as z:
            for fn in z.namelist():
                staged.setdefault(model, {})[fn] = z.read(fn)
    for model, files in staged.items():
        overlap = [fn for fn in files if (CACHE[model] / fn).exists()]
        assert not overlap, f"{len(overlap)} collisions in {model} cache!"
        for fn, data in files.items():
            (CACHE[model] / fn).write_bytes(data)
        print(f"installed {len(files)} new files into {CACHE[model]} (0 overwrites)")
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "c1_loo_beliefs.parquet", index=False)
    print(f"wrote {OUT/'c1_loo_beliefs.parquet'} rows={len(df)} (dev only)")


if __name__ == "__main__":
    main()
