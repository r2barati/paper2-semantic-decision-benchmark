"""Paper2 G2-FULL test ablations (14B): C1 leave-one-out on test-160 rerank k=3.

Serves Qwen3-8B-AWQ via vLLM (same flags/patch as the frozen belief runs),
runs consume_c1 over the 3 two-doc subsets per test query (480 calls).
TEST QUERIES ONLY (sealed-test single-touch authorization); hard abort on
any qrels or non-test query. Prompts/tau/model pinned; src snap byte-verified.
Outputs: cache_loo_test_14b.zip + manifest_test_14b.json
"""

import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
import zipfile

MODEL = "Qwen/Qwen3-14B-AWQ"
REVISION = "31c69efc29464b6bb0aee1398b5a7b50a99340c3"
GPU_MEM = "0.85"
EXPECTED_CALLS = 480

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
    print(f"loo pilot model={MODEL}", flush=True)
    for root in ("/kaggle/input", "/kaggle/working"):
        for dirpath, _, files in os.walk(root):
            assert "test.tsv" not in files and "dev.tsv" not in files, \
                f"QRELS VISIBLE at {dirpath} -- refusing to run (seal)"
    print("seal guard ok: no qrels attached", flush=True)

    snap = find_root("consumers_v3.py")
    if os.path.basename(snap) == "src" and os.path.exists(os.path.join(snap, "__init__.py")):
        sys.path.insert(0, os.path.dirname(snap))
    else:
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
    assert C.ABSTAIN_TAU == 0.5, "tau drift!"
    print("frozen C1 prompt+tau ok", flush=True)

    print("cuda:", __import__("torch").cuda.is_available(), flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm", "openai",
                    "pydantic", "numpy"], check=True)
    sirven = subprocess.Popen(
        [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
         "--model", MODEL, "--revision", REVISION,
         "--quantization", "awq", "--max-model-len", "4096",
         "--gpu-memory-utilization", GPU_MEM, "--port", "8000"],
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

    _orig_client = C._client

    def _patched_client():
        client = _orig_client()
        orig_create = client.chat.completions.create

        def create(*a, **k):
            k.setdefault("extra_body",
                         {"chat_template_kwargs": {"enable_thinking": False}})
            return orig_create(*a, **k)

        client.chat.completions.create = create
        return client

    C._client = _patched_client

    inp = find_root("corpus.jsonl")
    docs = {}
    for line in open(os.path.join(inp, "corpus.jsonl")):
        d = json.loads(line)
        docs[d["_id"]] = d
    queries = {}
    for line in open(os.path.join(inp, "queries_loo.jsonl")):
        q = json.loads(line)
        assert q["metadata"].get("split") == "test", f"non-test query {q['_id']}"
        queries[q["_id"]] = q
    spec = json.loads(open(os.path.join(inp, "loo_sets_test.json")).read())
    assert set(queries) == set(spec["test"]) == set(spec["sets"]), \
        "query/loo_sets dev mismatch"
    assert len(queries) == 160, f"expected 160 test queries, got {len(queries)}"
    print("test scope ok: 160 queries", flush=True)

    n_calls = n_fail = 0
    fails = []
    t0 = time.time()
    for i, qid in enumerate(sorted(queries)):
        q = queries[qid]
        for pos in ("drop0", "drop1", "drop2"):
            dids = spec["sets"][qid][pos]
            assert len(dids) == 2
            top = [{"doc_id": did, "text": docs[did]["text"]} for did in dids]
            try:
                C.consume_c1(top, q["text"], MODEL)
                n_calls += 1
            except Exception as e:
                n_fail += 1
                fails.append([qid, pos, str(e)[:150]])
                time.sleep(5)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/40 qid={qid} calls={n_calls} fails={n_fail} "
                  f"elapsed={(time.time() - t0) / 60:.1f}min", flush=True)
    assert n_calls == EXPECTED_CALLS, f"expected {EXPECTED_CALLS} calls, got {n_calls}"
    cache_files = sorted(os.listdir("/kaggle/working/cache_out"))
    zpath = "/kaggle/working/cache_loo_test_14b.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in cache_files:
            z.write(os.path.join("/kaggle/working/cache_out", fn), fn)
    manifest = {
        "model": MODEL, "revision": REVISION,
        "scope": "c1-loo-test160-rerank-k3",
        "c1_sha": C.prompt_sha(C.PROMPT_C1), "tau": C.ABSTAIN_TAU,
        "src_snap_sha": {f: sha16(os.path.join(snap, f)) for f in
                         ("consumers_v3.py", "interpreter.py", "events.py")},
        "n_llm_calls": n_calls, "n_fail": n_fail, "failures": fails,
        "n_cache_files": len(cache_files),
        "elapsed_min": round((time.time() - t0) / 60, 1),
        "seal": "no qrels attached; guard passed",
    }
    with open("/kaggle/working/manifest_test_14b.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("PILOT:", json.dumps({k: v for k, v in manifest.items()
                                if k != "failures"}), flush=True)
    sirven.terminate()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
