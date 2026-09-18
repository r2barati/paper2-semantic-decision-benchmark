"""Paper2 V3 Phi-4 belief shard: serve stelterlab/phi-4-AWQ via vLLM, warm the
frozen C1/C3 belief cache over a query-index shard. NO qrels attached (hard
abort if any test/dev qrels are visible). Prompts/tau/model pinned; src_snap
files are verified byte-identical to the frozen repo copies before use.

Plain instruct model: NO thinking-trace transport patch (none needed; prompt
TEXT is untouched and SHAs asserted below).

Env: shard comes from shard.json inside the attached dataset (one dataset
version per shard push). SHARD_START/SHARD_END env override is for local smoke
tests only.
Outputs: cache_shard_{S}_{E}.zip + shard_manifest.json
"""

import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
import zipfile

MODEL = "stelterlab/phi-4-AWQ"
REVISION = "075b93fe5ab0d2e86004a5d68c7575ec3bb5a88b"
KS = (3, 5)


def _shard():
    for root in ("/kaggle/input", "."):
        for dirpath, _, files in os.walk(root):
            if "shard.json" in files:
                s = json.load(open(os.path.join(dirpath, "shard.json")))
                return int(s["start"]), int(s["end"])
    if "SHARD_START" in os.environ:  # local smoke tests only
        return int(os.environ["SHARD_START"]), int(os.environ["SHARD_END"])
    raise SystemExit("shard.json not found in attached dataset -- refusing to run")


SHARD_START, SHARD_END = _shard()

os.environ["PAPER2_V3_CACHE_DIR"] = "/kaggle/working/cache_out"
os.environ["LLM_BASE_URL"] = "http://localhost:8000/v1"
os.environ["LLM_API_KEY"] = "local"


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


def sha16(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def main():
    print(f"shard [{SHARD_START},{SHARD_END}) model={MODEL}", flush=True)
    # Seal guard: no evaluation labels may be visible to this kernel.
    for root in ("/kaggle/input", "/kaggle/working"):
        for dirpath, _, files in os.walk(root):
            assert "test.tsv" not in files and "dev.tsv" not in files, \
                f"QRELS VISIBLE at {dirpath} -- refusing to run (test seal)"
    print("seal guard ok: no qrels attached", flush=True)

    snap = find_root("consumers_v3.py")  # .../src (package) or flat dir
    if os.path.basename(snap) == "src" and os.path.exists(os.path.join(snap, "__init__.py")):
        sys.path.insert(0, os.path.dirname(snap))
    else:
        # Fallback: materialise a src/ package from flat files (dataset race).
        import shutil
        pkg = "/tmp/pkgsrc/src"
        os.makedirs(pkg, exist_ok=True)
        open(os.path.join(pkg, "__init__.py"), "w").write("")
        for f in ("consumers_v3.py", "interpreter.py", "events.py"):
            shutil.copy(os.path.join(snap, f), os.path.join(pkg, f))
        sys.path.insert(0, "/tmp/pkgsrc")
        snap = pkg
    import src.consumers_v3 as C
    assert C.prompt_sha(C.PROMPT_C1) == "66e6890ba9c5464c", "C1 prompt drift!"
    assert C.prompt_sha(C.PROMPT_C3_DOC) == "5966cadde90e8236", "C3 prompt drift!"
    assert C.ABSTAIN_TAU == 0.5, "tau drift!"
    print("frozen prompts+tau ok", flush=True)

    print("cuda:", __import__("torch").cuda.is_available(), flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm", "openai",
                    "pydantic", "numpy"], check=True)
    sirven = subprocess.Popen(
        [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
         "--model", MODEL, "--revision", REVISION,
         "--quantization", "awq", "--max-model-len", "4096",
         "--gpu-memory-utilization", "0.85", "--port", "8000"],
        stdout=open("/tmp/vllm.log", "w"), stderr=subprocess.STDOUT)
    deadline = time.time() + 1800
    while time.time() < deadline:
        try:
            urllib.request.urlopen("http://localhost:8000/health", timeout=5)
            break
        except Exception:
            time.sleep(15)
    else:
        raise SystemExit("vLLM server never became healthy")
    print("server healthy", flush=True)

    inp = find_root("corpus.jsonl")
    docs, queries = {}, []
    for line in open(os.path.join(inp, "corpus.jsonl")):
        d = json.loads(line)
        docs[d["_id"]] = d
    for line in open(os.path.join(inp, "queries.jsonl")):
        queries.append(json.loads(line))
    rank = {}
    for sys_name, trec in (("bm25", "bm25_full.trec"), ("dense", "dense_full.trec"),
                           ("hybrid", "hybrid_k120_full.trec"),
                           ("rerank", "rerank_full.trec")):
        with open(os.path.join(inp, trec)) as f:
            for line in f:
                qid, _, did, _, _, _ = line.split()
                rank.setdefault(sys_name, {}).setdefault(qid, []).append(did)
    import numpy as np
    rng = np.random.default_rng(7)
    ids = list(docs)
    rank["random"] = {q["_id"]: list(rng.permutation(ids)) for q in queries}
    arms = json.loads(open(os.path.join(inp, "arms.json")).read())
    sets = {}
    for sys_name, per_q in rank.items():
        for qid, ranking in per_q.items():
            for k in KS:
                sets[(sys_name, qid, k)] = ranking[:k]
    for arm, per_q in arms.items():
        for qid, kd in per_q.items():
            if isinstance(kd, dict):
                for k in KS:
                    if str(k) in kd:
                        sets[(arm, qid, k)] = kd[str(k)]
            elif isinstance(kd, list):
                sets[(arm, qid, 3)] = kd
    systems = sorted({s for s, _, _ in sets} - {"none"})
    print(f"systems={systems}", flush=True)

    queries = queries[SHARD_START:SHARD_END]
    qmeta = {q["_id"]: q["metadata"] for q in queries}
    qtext = {q["_id"]: q["text"] for q in queries}
    n_calls = n_fail = 0
    fails = []
    t0 = time.time()
    for i, q in enumerate(queries):
        qid = q["_id"]
        for sys_name in systems:
            for k in KS:
                key = (sys_name, qid, k)
                if key not in sets:
                    continue
                top = [{"doc_id": did, "text": docs[did]["text"]}
                       for did in sets[key]]
                for consumer in ("C1", "C3"):
                    for retry in range(6):
                        try:
                            if consumer == "C1":
                                C.consume_c1(top, qtext[qid], MODEL)
                            else:
                                C.consume_c3(top, qtext[qid], MODEL,
                                             qmeta[qid]["entity_node"])
                            n_calls += 1
                            break
                        except Exception as e:
                            if retry < 5:
                                time.sleep(10)
                                continue
                            n_fail += 1
                            fails.append([qid, sys_name, k, consumer, str(e)[:150]])
                            time.sleep(5)
                            break
        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{len(queries)} qid={qid} calls={n_calls} "
                  f"fails={n_fail} elapsed={(time.time() - t0) / 60:.1f}min",
                  flush=True)
    cache_files = sorted(os.listdir("/kaggle/working/cache_out"))
    zpath = (f"/kaggle/working/cache_shard_{SHARD_START}_{SHARD_END}.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in cache_files:
            z.write(os.path.join("/kaggle/working/cache_out", fn), fn)
    manifest = {
        "model": MODEL, "revision": REVISION,
        "shard": [SHARD_START, SHARD_END],
        "c1_sha": C.prompt_sha(C.PROMPT_C1),
        "c3_sha": C.prompt_sha(C.PROMPT_C3_DOC), "tau": C.ABSTAIN_TAU,
        "src_snap_sha": {f: sha16(os.path.join(snap, f)) for f in
                         ("consumers_v3.py", "interpreter.py", "events.py")},
        "n_llm_calls": n_calls, "n_fail": n_fail, "failures": fails,
        "n_cache_files": len(cache_files),
        "elapsed_min": round((time.time() - t0) / 60, 1),
        "seal": "no qrels attached; guard passed",
    }
    with open("/kaggle/working/shard_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("SHARD:", json.dumps({k: v for k, v in manifest.items()
                                if k != "failures"}), flush=True)
    sirven.terminate()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
