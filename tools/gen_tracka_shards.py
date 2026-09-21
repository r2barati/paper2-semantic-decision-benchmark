"""Generate Track A scale-up shards (deterministic scaffolding, committed).

Reads kaggle_kernel/p2_tracka_sim.py (pilot, verified pattern) and emits,
per shard: kaggle/inputs_tracka/shard_bN.json + kaggle_kernel/p2_tracka_bN.py
with ONLY three constants substituted (SHARD_FILE, OUT_EPISODES,
OUT_MANIFEST) plus docstring shard id. Then refreshes src_shas.json.
Shard map (frozen): b1 dense, b2 hybrid, b3 random, b4 inject-contra,
b5 stale-swap, b6 drop-decisive; each 1 system x 5 strata
(C0/none, C1/C3 x 8B/14B) x k=3 x 160 test queries x 5 seeds = 4000 eps.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INP = ROOT / "kaggle" / "inputs_tracka"
KER = ROOT / "kaggle_kernel"

SYSTEMS = ["dense", "hybrid", "random", "intervention-inject-contra",
           "intervention-stale-swap", "intervention-drop-decisive"]
STRATA = [("C0", "none"),
          ("C1", "Qwen/Qwen3-8B-AWQ"), ("C1", "Qwen/Qwen3-14B-AWQ"),
          ("C3", "Qwen/Qwen3-8B-AWQ"), ("C3", "Qwen/Qwen3-14B-AWQ")]

sp = json.load(open(ROOT / "data" / "v3" / "splits" / "splits.json"))
test = sorted(sp["test"])
assert len(test) == 160
seeds = [60000 + i for i in range(5)]

template = (KER / "p2_tracka_sim.py").read_text()
for const in ('SHARD_FILE = "shard_pilot.json"',
              'OUT_EPISODES = "episodes_tracka_pilot.parquet"',
              'OUT_MANIFEST = "shard_manifest.json"'):
    assert const in template, const

for n, sys_name in enumerate(SYSTEMS, start=1):
    sid = f"b{n}"
    spec = {"shard_id": f"bstock-{sys_name}", "controller": "BeliefBaseStock",
            "cells": [{"system": sys_name, "consumer": c, "model": m, "k": 3}
                      for c, m in STRATA],
            "queries": test, "seeds": seeds}
    (INP / f"shard_{sid}.json").write_text(json.dumps(spec, indent=1))
    out_ep = f"episodes_tracka_{sid}.parquet"
    out_mf = f"manifest_tracka_{sid}.json"
    src = template.replace('SHARD_FILE = "shard_pilot.json"',
                           f'SHARD_FILE = "shard_{sid}.json"')
    src = src.replace('OUT_EPISODES = "episodes_tracka_pilot.parquet"',
                      f'OUT_EPISODES = "{out_ep}"')
    src = src.replace('OUT_MANIFEST = "shard_manifest.json"',
                      f'OUT_MANIFEST = "{out_mf}"')
    src = src.replace("Driven by shard_pilot.json.",
                      f"Scale-up shard {sid} ({sys_name}); template-generated.")
    (KER / f"p2_tracka_{sid}.py").write_text(src)
    print(f"{sid}: {sys_name} spec + kernel ({len(spec['cells'])} cells, "
          f"{len(spec['cells']) * len(test) * len(seeds)} eps)")

shas = {}
for p in sorted(INP.rglob("*")):
    if p.is_file() and p.name != "src_shas.json":  # self-describing record
        shas[str(p.relative_to(INP))] = hashlib.sha256(p.read_bytes()).hexdigest()
(INP / "src_shas.json").write_text(json.dumps(shas, indent=1))
print(f"src_shas.json refreshed: {len(shas)} files")
