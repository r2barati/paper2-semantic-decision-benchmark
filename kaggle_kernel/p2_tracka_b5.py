"""Paper2 Track A sim shard (CPU-only): BeliefBaseStock episodes for a frozen cell set.

Mirrors tools/run_controllerB_v3main.py _work exactly (same beliefs, paired
seeds, warning protocol; only the fixed policy class differs from the main
sim, per the labeled controller-swap extension). Scale-up shard b5 (intervention-stale-swap); template-generated.
NO qrels attached (beliefs carry no labels; hard abort if seen).
Outputs: episodes_tracka_pilot.parquet + shard_manifest.json
"""

import hashlib
import json
import os
import subprocess
import sys
import time


PINS = [("numpy", "1.26.4"), ("pandas", "2.3.3"), ("pyarrow", "21.0.0"),
        ("scipy", "1.13.1"), ("scikit-learn", "1.6.1")]


def _need(mod, ver):
    """Metadata-only check (never imports: importing numpy pre-install would
    pin the stale module in sys.modules past the reinstall)."""
    import importlib.metadata as _md
    try:
        return _md.version(mod) != ver
    except _md.PackageNotFoundError:
        return True


if any(_need(mod, ver) for mod, ver in PINS):
    # Single combined install so binary wheels stay mutually consistent
    # (downgrading numpy alone breaks preinstalled pandas/scipy binaries).
    # --force-reinstall --no-deps: the stock resolver otherwise keeps
    # numpy>=2 to satisfy unrelated preinstalled packages we never use.
    reqs = [f"{mod.replace('_', '-')}=={ver}" for mod, ver in PINS]
    print("installing pinned set:", reqs, flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "--force-reinstall", "--no-deps"] + reqs, check=True)
import importlib as _il
_il.invalidate_caches()
IMPORT_OF = {"scikit-learn": "sklearn"}
for mod, ver in PINS:
    got = _il.import_module(IMPORT_OF.get(mod, mod)).__version__
    assert got == ver, (mod, got, ver)
    print(f"{mod} {got} pinned ok", flush=True)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


SHARD_FILE = "shard_b5.json"
OUT_EPISODES = "episodes_tracka_b5.parquet"
OUT_MANIFEST = "manifest_tracka_b5.json"


def find_input(name):
    for root in ("/kaggle/input", ".", "inputs", "/kaggle/working/inputs"):
        for dirpath, _, files in os.walk(root):
            if name in files:
                return os.path.join(dirpath, name)
    raise FileNotFoundError(name)


def find_root(name):
    for root in ("/kaggle/input",):
        for dirpath, _, files in os.walk(root):
            if name in files:
                return dirpath
    raise FileNotFoundError(name)


def sha16_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _work(item):
    """Top-level for multiprocessing pickling (spawn). Self-contained imports."""
    import sys as _s
    _s.path.insert(0, "/kaggle/working/pkgsrc")
    from src.controller_basestock import BeliefBaseStock
    from src.env import InventoryEnv
    from src.events import Regime, P5_HORIZON, P5_WARNING_TIME
    from src.metrics import compute_episode_metrics
    from src.interpreter import no_info_regime_belief
    from src.sim_eval_v3main import belief_row_to_interpretation
    import time as _t
    sys_name, cons, model, qid, true_regime, probs, seed = item
    t0 = _t.time()
    env = InventoryEnv(seed=seed, regime=Regime(true_regime))
    env.reset()
    interp = belief_row_to_interpretation({
        "p_normal": probs[0], "p_supplier_delay": probs[1],
        "p_demand_surge": probs[2], "abstain": False,
        "confidence": 1.0, "doc_ids": []})
    prior = no_info_regime_belief().normalized().regime_probabilities
    belief = interp.normalized().regime_probabilities
    ctl = BeliefBaseStock(prior)
    history = []
    for t in range(P5_HORIZON):
        if t == P5_WARNING_TIME:
            ctl = BeliefBaseStock(belief)
        history.append(env.step(ctl.decide(env._state)))
    m = compute_episode_metrics(history, seed=seed,
                                condition=f"B+{sys_name}+{cons}")
    return {"system": sys_name, "consumer": cons, "model": model, "k": 3,
            "query_id": qid, "true_regime": true_regime, "seed": seed,
            "rung": "belief", "controller": "BeliefBaseStock",
            "profit": float(m.total_profit),
            "wall_s": round(_t.time() - t0, 3)}


def main():
    t_all = time.time()
    for root in ("/kaggle/input", "/kaggle/working"):
        for dirpath, _, files in os.walk(root):
            assert "dev.tsv" not in files and "test.tsv" not in files, \
                f"QRELS VISIBLE at {dirpath} -- refusing to run (test seal)"
    print("seal guard ok: no qrels attached", flush=True)

    inp = find_root(SHARD_FILE)
    # input-hash gate. NOTE: Kaggle dataset attach FLATTENS paths (blob name
    # = basename), so compare by basename: assert recorded basenames are
    # unique, then match each staged file by basename.
    recorded = json.loads(open(os.path.join(inp, "src_shas.json")).read())
    rec_base = {}
    for k, v in recorded.items():
        b = os.path.basename(k)
        assert b not in rec_base, f"basename collision in record: {b}"
        rec_base[b] = v
    actual = {}
    for dirpath, _, files in os.walk(inp):
        for fn in files:
            if fn == "src_shas.json":
                continue  # self-describing record: verified by content, not by hash
            p = os.path.join(dirpath, fn)
            if fn in actual:
                raise SystemExit(f"duplicate basename staged: {fn}")
            actual[fn] = sha16_file(p)
    print(f"DEBUG inp={inp} walk={sorted(actual)}", flush=True)
    assert set(actual) == set(rec_base), \
        f"staged file set drift: {set(actual) ^ set(rec_base)}"
    for k, v in actual.items():
        assert v == rec_base[k], f"hash mismatch on {k}"
    print(f"input-hash gate ok: {len(actual)} files (flat attach)", flush=True)
    # reconstruct the src/ package from flat files for frozen imports
    pkgdir = "/kaggle/working/pkgsrc"
    os.makedirs(os.path.join(pkgdir, "src"), exist_ok=True)
    for dirpath, _, files in os.walk(inp):
        for fn in files:
            if fn.endswith(".py"):
                with open(os.path.join(dirpath, fn), "rb") as fh:
                    data = fh.read()
                with open(os.path.join(pkgdir, "src", fn), "wb") as fh:
                    fh.write(data)
    open(os.path.join(pkgdir, "src", "__init__.py"), "a").close()

    sys.path.insert(0, pkgdir)
    import src.controller_basestock  # noqa: E402
    import src.env  # noqa: E402
    import src.sim_eval_v3main as SE  # noqa: E402
    print("snapshot modules:",
          sha16_file(os.path.join(pkgdir, "src", "controller_basestock.py"))[:12],
          sha16_file(os.path.join(pkgdir, "src", "env.py"))[:12], flush=True)

    spec = json.loads(open(os.path.join(inp, SHARD_FILE)).read())
    assert spec["controller"] == "BeliefBaseStock", spec["controller"]
    from src.controller_basestock import BeliefBaseStock
    from src.env import InventoryEnv
    from src.events import Regime, P5_HORIZON, P5_WARNING_TIME
    from src.metrics import compute_episode_metrics
    from src.interpreter import no_info_regime_belief
    bel = pd.concat([pd.read_parquet(os.path.join(inp, f))
                     for f in ("beliefs_Qwen_Qwen3-8B-AWQ.parquet",
                               "beliefs_Qwen_Qwen3-14B-AWQ.parquet")],
                    ignore_index=True)
    # C0 twin dedupe (mirrors run_sim_v3main.load_beliefs_dedup): C0 rows
    # appear in both model files; assert identical, keep first.
    c0 = bel[bel["consumer"] == "C0"]
    _grp = c0.groupby(["system", "query_id", "k"])
    assert (_grp.size() == 2).all(), "every C0 key must appear exactly twice"
    _scalar = [c for c in c0.columns if c not in ("doc_ids", "latency_s", "model")]
    _nun = _grp[_scalar].nunique()
    assert bool((_nun == 1).all().all()), "C0 twin rows differ!"
    bel = pd.concat([c0.sort_values(["system", "query_id", "k"]).drop_duplicates(
        ["system", "query_id", "k"], keep="first"),
        bel[bel["consumer"] != "C0"]], ignore_index=True)
    bel["model"] = bel["model"].fillna("none")
    items = []
    for cell in spec["cells"]:
        sub = bel[(bel["system"] == cell["system"]) &
                  (bel["consumer"] == cell["consumer"]) &
                  (bel["model"] == cell["model"]) &
                  (bel["k"] == cell["k"]) &
                  (bel["query_id"].isin(spec["queries"]))]
        assert len(sub) == len(spec["queries"]), \
            (cell, len(sub), len(spec["queries"]))
        for _, r in sub.iterrows():
            probs = (r["p_normal"], r["p_supplier_delay"], r["p_demand_surge"])
            for seed in spec["seeds"]:
                assert cell["k"] == 3, cell
                items.append((cell["system"], cell["consumer"], cell["model"],
                              r["query_id"], r["true_regime"], probs, seed))
    total = len(items)
    print(f"shard {spec['shard_id']}: {total} episodes", flush=True)

    from multiprocessing import Pool
    import multiprocessing as mp
    workers = min(4, mp.cpu_count())
    done, rows = 0, []
    with Pool(workers, maxtasksperchild=10) as pool:
        for row in pool.imap_unordered(_work, items, chunksize=4):
            rows.append(row)
            done += 1
            if done % 10 == 0 or done == total:
                print(f"  progress {done}/{total} "
                      f"elapsed={(time.time() - t_all) / 60:.1f}min", flush=True)
    assert done == total == len(rows), (done, total, len(rows))
    df = pd.DataFrame(rows)
    df.to_parquet("/kaggle/working/" + OUT_EPISODES, index=False)
    manifest = {
        "shard_id": spec["shard_id"], "controller": spec["controller"],
        "config_hash": sha16_file(os.path.join(inp, SHARD_FILE)),
        "spec_file": SHARD_FILE,
        "input_hashes": {k: v for k, v in actual.items()},
        "seeds": spec["seeds"], "completed": done, "total": total,
        "runtime_min": round((time.time() - t_all) / 60, 2),
        "artifact": OUT_EPISODES,
        "dep_versions": {m: __import__("importlib").import_module(
            m).__version__ for m in ("numpy", "pandas", "pyarrow", "scipy")},
        "seal": "no qrels attached; guard passed",
    }
    with open("/kaggle/working/" + OUT_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print("SHARD MANIFEST:", json.dumps(
        {k: v for k, v in manifest.items() if k != "input_hashes"}),
        flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
