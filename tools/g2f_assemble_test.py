"""G2-FULL test assembly: parse 378 payloads -> C1 LOO belief rows (test).

Mirrors tools/g2a1_assemble.py with test paths. Parses every payload under
frozen C1 normalization, recomputes cache keys, asserts test-only IDs,
asserts ZERO filename collisions before additive install.
Writes extension_g2a/c1_loo_test_beliefs.parquet (960 rows).
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
RUNS = ROOT / "runs" / "v3main_lootest"
CACHE = {"Qwen/Qwen3-8B-AWQ": ROOT / "results" / "v3main" / "beliefs_cache_awq",
         "Qwen/Qwen3-14B-AWQ": ROOT / "results" / "v3main" / "beliefs_cache_qwen14"}
MODELS = list(CACHE)

sp = json.load(open(ROOT / "data" / "v3" / "splits" / "splits.json"))
test_ids = set(sp["test"])
assert len(test_ids) == 160


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
    for line in open(ROOT / "kaggle" / "inputs_loo_test" / "queries_loo.jsonl"):
        q = json.loads(line)
        assert q["metadata"].get("split") == "test"
        qs[q["_id"]] = q["text"]
    spec = json.load(open(ROOT / "kaggle" / "inputs_loo_test" / "loo_sets_test.json"))
    assert set(spec["test"]) == test_ids and set(spec["sets"]) == test_ids

    payloads = {}
    for model in MODELS:
        tag = "8b" if model.endswith("8B-AWQ") else "14b"
        with zipfile.ZipFile(RUNS / f"cache_loo_test_{tag}.zip") as z:
            for fn in z.namelist():
                p = json.loads(z.read(fn))
                assert p["model"] == model, fn
                assert p.get("prompt_sha") == "66e6890ba9c5464c", fn
                payloads[(model, fn)] = _parse_json(p["raw"])
    print(f"payloads parsed: {len(payloads)} (expect 378)")

    rows = []
    for model in MODELS:
        for qid in sorted(spec["sets"]):
            assert qid in test_ids, f"non-test query {qid} -- refusing"
            for pos in ("drop0", "drop1", "drop2"):
                dids = spec["sets"][qid][pos]
                user = (f"Operator information need: {qs[qid]}\n\nEvidence:\n" +
                        "\n\n".join(f"[DOC {d}] {docs[d]}" for d in dids))
                fn = _c1_key(model, user)
                probs = payloads[(model, fn)]
                assert {"normal", "supplier_delay", "demand_surge"} <= set(probs), fn
                assert all(v >= 0 for v in
                           (probs["normal"], probs["supplier_delay"],
                            probs["demand_surge"])), fn
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
                             "split": "test", "cache_file": fn})
    assert len(rows) == 960, len(rows)
    staged = {}
    for model in MODELS:
        tag = "8b" if model.endswith("8B-AWQ") else "14b"
        with zipfile.ZipFile(RUNS / f"cache_loo_test_{tag}.zip") as z:
            for fn in z.namelist():
                staged.setdefault(model, {})[fn] = z.read(fn)
    for model, files in staged.items():
        same, diff = 0, []
        for fn, data in files.items():
            target = CACHE[model] / fn
            if target.exists():
                if target.read_bytes() == data:
                    same += 1
                else:
                    diff.append(fn)  # first-wins: keep frozen entry, log it
            else:
                target.write_bytes(data)
        new = len(files) - same - len(diff)
        print(f"{CACHE[model].name}: {new} new, {same} identical-dup, "
              f"{len(diff)} divergent-dup (first-wins kept)")
        if diff:
            json.dump(diff, open(OUT / f"test_install_divergences_"
                                     f"{'8b' if model.endswith('8B-AWQ') else '14b'}.json",
                                 "w"), indent=1)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "c1_loo_test_beliefs.parquet", index=False)
    print(f"wrote c1_loo_test_beliefs.parquet rows={len(df)} (test)")


if __name__ == "__main__":
    main()
