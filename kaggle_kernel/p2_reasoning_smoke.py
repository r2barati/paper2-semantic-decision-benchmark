"""Paper2 EXP-P2-REASONING-AGENTIC-v1 smoke kernel (GPU: Qwen3-8B-AWQ via vLLM; retrieval: CPU BM25).

Ladder: A0 Direct -> A1 Structured deliberation -> A2 Verification -> R0 Static RAG -> A3 Agentic RAG.
Correction 2: live frozen BM25 over corpus.jsonl text-only (no TREC mapping, no switching).
Correction 4: ONE belief per (warning x arm); sim replay is a separate CPU stage.
Correction 5: smoke requires ZERO parse/schema failures; enable_thinking=False.

NETWORK POLICY (infrastructure exception I-1, not a design change):
allowlisted infrastructure only — PyPI (pinned pip installs) + Hugging Face Hub
(model weight retrieval). The A0/A1/A2/R0/A3 arms perform ZERO network calls:
all LLM traffic goes to the local vLLM endpoint http://localhost:8000 only.

Seal: hard-aborts if test.tsv / dev.tsv / evidence_labels.parquet / queries.jsonl
is present under /kaggle/input. Loader reads d["text"] only.

Inputs (/kaggle/input): corpus.jsonl, shard.json {"warnings":[{warning_id,text}]}.
Outputs (/kaggle/working): cache_reasoning_smoke.zip, smoke_manifest.json
"""

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import zipfile
from collections import Counter

# ---------------------------------------------------------------- pins / config
MODEL = "Qwen/Qwen3-8B-AWQ"
REVISION = "4da05a8edb55c6046cce958586c33b61da07bb79"
VLLM_PIN = "vllm==0.11.0"  # v2: 0.10.2 crashed on Qwen2Tokenizer/all_special_tokens_extended
OPENAI_PIN = "openai==2.48.0"  # matches repo requirements.txt
# v3: transformers 5.0.0 (pulled by image/vLLM) removed all_special_tokens_extended,
# crashing vLLM's CachedTokenizer under both 0.10.2 and 0.11.0. Pin the in-repo proven
# 4.x line already used for Qwen3 encode/rerank kernels (p2_encode_2b.py).
TRANSFORMERS_PIN = "transformers==4.57.6"
MAX_MODEL_LEN = 4096
GPU_MEM_UTIL = 0.90
K1, B, TOP_K = 1.5, 0.75, 3
BUDGET = {"A0": 0, "A1": 0, "A2": 0, "R0": 1, "A3": 2}

NETWORK_POLICY = ("allowlisted infrastructure only: PyPI (pinned: vllm==0.11.0, "
                  "openai==2.48.0, transformers==4.57.6) + Hugging Face Hub (model weight retrieval). "
                  "openai==2.48.0) + Hugging Face Hub (model weight retrieval). "
                  "Experiment arms A0/A1/A2/R0/A3 perform zero network calls; "
                  "all LLM traffic targets local vLLM http://localhost:8000 only.")

A0_SYS = ("You are an operational risk analyst. Given a textual supplier or market warning, "
          "output regime probabilities directly. Respond ONLY with valid JSON:\n"
          '{"p_normal": <float 0-1>, "p_supplier_delay": <float 0-1>, "p_demand_surge": <float 0-1>}\n'
          "Benchmark prior: normal 0.35, supplier_delay 0.35, demand_surge 0.30. "
          "The three probabilities must sum to 1.0. No explanation, no retrieval, no revision.")
A1_SYS = ("You are an operational risk analyst. Follow this fixed procedure: (1) extract factual claims; "
          "(2) list which claims support each regime (normal/supplier_delay/demand_surge); "
          "(3) list contradictory or ambiguous evidence; (4) assign regime probabilities; (5) report. "
          "Respond ONLY with valid JSON:\n"
          '{"factual_claims": [...], "support": {"normal": [...], "supplier_delay": [...], "demand_surge": [...]}, '
          '"contradictions": [...], "probabilities": {"p_normal": <float>, "p_supplier_delay": <float>, '
          '"p_demand_surge": <float>}}\nBenchmark prior: normal 0.35, supplier_delay 0.35, demand_surge 0.30. '
          "Probabilities sum to 1.0.")
A2_SYS = ("You are a verification analyst. Given the warning and an initial structured belief, (1) re-inspect "
          "the warning; (2) list evidence potentially inconsistent with the initial conclusion; "
          "(3) revise probabilities exactly once. Respond ONLY with valid JSON:\n"
          '{"inconsistent_evidence": [...], "revised_probabilities": {"p_normal": <float>, '
          '"p_supplier_delay": <float>, "p_demand_surge": <float>}}\n'
          "Benchmark prior: normal 0.35, supplier_delay 0.35, demand_surge 0.30. Sum to 1.0. "
          "This is the sole permitted revision.")
R0_SYS = ("You are an operational risk analyst. Given a warning and frozen retrieved evidence (ranked excerpts), "
          "synthesize and output regime probabilities. Respond ONLY with valid JSON:\n"
          '{"p_normal": <float>, "p_supplier_delay": <float>, "p_demand_surge": <float>}\n'
          "Benchmark prior: normal 0.35, supplier_delay 0.35, demand_surge 0.30. Sum to 1.0. "
          "Do not issue further queries.")
A3H_SYS = ("You are an operational risk analyst with a retrieval tool. Given a warning, (1) state hypotheses "
           "per regime; (2) write ONE first retrieval query (verbatim warning text is acceptable). "
           "Respond ONLY with valid JSON:\n"
           '{"hypotheses": {"normal": [...], "supplier_delay": [...], "demand_surge": [...]}, "query_1": "<string>"}')
A3S_SYS = ("Given the warning, your hypotheses, and the retrieved evidence from at most two queries (optionally: "
           "one reformulated query `query_2`), synthesize final regime probabilities. Respond ONLY with valid JSON:\n"
           '{"query_2_used": <true|false>, "query_2": "<string or empty>", "used_evidence": ["<doc ids>"], '
           '"probabilities": {"p_normal": <float>, "p_supplier_delay": <float>, "p_demand_surge": <float>}}\n'
           "Benchmark prior: normal 0.35, supplier_delay 0.35, demand_surge 0.30. Sum to 1.0. Stop.")

OUT_ZIP = "/kaggle/working/cache_reasoning_smoke.zip"
OUT_MANIFEST = "/kaggle/working/smoke_manifest.json"

# ---------------------------------------------------------------- helpers
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(t):
    return _TOKEN.findall(t.lower())


def sha16(s):
    return hashlib.sha256(s.encode()).hexdigest()[:16]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in docs]
        self.n = len(self.docs)
        self.dl = [len(d) for d in self.docs]
        self.avg = sum(self.dl) / self.n if self.n else 0.0
        self.tf = [Counter(d) for d in self.docs]
        df = Counter()
        for d in self.docs:
            df.update(set(d))
        self.idf = {t: max(0.0, math.log(1.0 + (self.n - c + 0.5) / (c + 0.5))) for t, c in df.items()}

    def topk(self, q, k):
        import numpy as np
        qt = tokenize(q)
        out = np.zeros(self.n)
        for i in range(self.n):
            tf, dl = self.tf[i], self.dl[i]
            tot = 0.0
            for term in qt:
                f = tf.get(term, 0)
                if not f:
                    continue
                den = f + self.k1 * (1 - self.b + self.b * dl / (self.avg or 1.0))
                tot += self.idf.get(term, 0.0) * f * (self.k1 + 1) / den
            out[i] = tot
        return np.argsort(-out, kind="stable")[:k]


def find_input(name):
    for root in ("/kaggle/input", "."):
        for dp, _, files in os.walk(root):
            if name in files:
                return os.path.join(dp, name)
    raise FileNotFoundError(name)


def seal_inputs():
    seen = set()
    for root in ("/kaggle/input",):
        for dp, _, files in os.walk(root):
            for f in files:
                if f in ("test.tsv", "dev.tsv", "evidence_labels.parquet", "queries.jsonl"):
                    seen.add(os.path.join(dp, f))
    if seen:
        raise RuntimeError(f"SEAL VIOLATION: {sorted(seen)}")


def parse_strict(raw):
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(raw)


def belief_of(obj, arm):
    if arm in ("A0", "R0"):
        src = obj
    elif arm == "A1":
        src = obj["probabilities"]
    elif arm == "A2":
        src = obj["revised_probabilities"]
    elif arm == "A3":
        src = obj["probabilities"]
    else:
        raise ValueError(arm)
    p = {"normal": float(src["p_normal"]), "supplier_delay": float(src["p_supplier_delay"]),
         "demand_surge": float(src["p_demand_surge"])}
    assert all(0.0 <= v <= 1.0 for v in p.values()), p
    assert abs(sum(p.values()) - 1.0) <= 1e-6, p
    return p


def pkg_version(mod):
    try:
        import importlib.metadata as md
        return md.version(mod)
    except Exception as e:
        return f"unknown ({e})"


def main():
    t_all = time.time()
    run_log = {"network_policy": NETWORK_POLICY, "infra_exception_I1": "internet=True (see DESIGN §9)"}
    seal_inputs()
    print("seal ok: no qrels/evidence_labels/queries.jsonl in /kaggle/input", flush=True)
    shard = json.loads(open(find_input("shard.json")).read())
    warnings = shard["warnings"]
    assert all(set(w) == {"warning_id", "text"} for w in warnings), "shard must carry id+text only (F-5)"
    doc_ids, texts = [], []
    with open(find_input("corpus.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                doc_ids.append(d["_id"])
                texts.append(d["text"])  # text only
    bm25 = BM25(texts, K1, B)
    print(f"corpus ok: {len(texts)} docs; warnings: {len(warnings)}", flush=True)

    # Pinned infra installs (S-1). Internet used ONLY here + HF weights below.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    VLLM_PIN, OPENAI_PIN, TRANSFORMERS_PIN], check=True)
    # Fail fast if the server entry point moved between vLLM versions (seconds, not quota-hours).
    subprocess.run([sys.executable, "-c",
                    "import vllm.entrypoints.openai.api_server as m; print('entrypoint ok:', m.__name__)",
                    ], check=True)
    print(f"pinned ok: vllm={pkg_version('vllm')} openai={pkg_version('openai')} "
          f"transformers={pkg_version('transformers')}", flush=True)

    # GPU diagnostics (F-3a).
    gpu_info = {"torch_cuda": False}
    try:
        import torch
        gpu_info = {"torch_cuda": torch.cuda.is_available(),
                    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "device_count": torch.cuda.device_count(),
                    "total_mem_GB": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
                    if torch.cuda.is_available() else None}
    except Exception as e:
        gpu_info = {"torch_cuda": "unknown", "error": repr(e)}
    print(f"gpu: {gpu_info}", flush=True)

    t_srv0 = time.time()
    srv = subprocess.Popen([sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                            "--model", MODEL, "--revision", REVISION, "--quantization", "awq",
                            "--max-model-len", str(MAX_MODEL_LEN),
                            "--gpu-memory-utilization", str(GPU_MEM_UTIL),
                            "--port", "8000"])
    import urllib.request
    polls = 0
    for _ in range(120):
        polls += 1
        if srv.poll() is not None:
            raise RuntimeError(f"vLLM server process exited early (code {srv.returncode}); "
                               "see kernel log traceback above")
        try:
            if urllib.request.urlopen("http://localhost:8000/health", timeout=5).status == 200:
                break
        except Exception:
            time.sleep(15)
    else:
        raise RuntimeError("vLLM server failed to become healthy")
    srv_startup_s = time.time() - t_srv0
    print(f"vllm healthy after {srv_startup_s:.0f}s ({polls} polls)", flush=True)
    from openai import OpenAI
    client = OpenAI(api_key="local", base_url="http://localhost:8000/v1")

    # Served-model + HF weight identity (S-2).
    served = {}
    try:
        served = {"served_models": [m.id for m in client.models.list().data]}
    except Exception as e:
        served = {"served_models_error": repr(e)}
    hf_cache = {"status": "unprobed"}
    try:
        from huggingface_hub import scan_cache
        info = scan_cache()
        repos = [{"repo_id": r.repo_id, "revision": r.revision,
                  "files": sorted({f.file_name for s in r.snapshots for f in s.files})}
                 for r in info.repos if "Qwen3-8B-AWQ" in r.repo_id]
        hf_cache = {"status": "ok", "repos": repos}
    except Exception as e:
        hf_cache = {"status": "error", "error": repr(e)}

    latencies = []
    server_pid = srv.pid
    n_calls = 0
    results, logs, failures = {}, [], []

    def llm(system, user, tag):
        nonlocal n_calls
        t0 = time.time()
        r = client.chat.completions.create(
            model=MODEL, temperature=0, top_p=1.0, max_tokens=512,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        n_calls += 1
        u = r.usage
        usage = {"prompt_tokens": (u.prompt_tokens if u else 0),
                 "completion_tokens": (u.completion_tokens if u else 0)}
        fr = r.choices[0].finish_reason
        dt = time.time() - t0
        latencies.append(dt)
        raw = (r.choices[0].message.content or "")
        call = {"prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"],
                "latency_s": dt, "finish_reason": fr, "truncated": fr == "length"}
        raws[tag] = raw
        return raw.strip(), call

    raws = {}

    def ckpt():
        with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
            for k, v in results.items():
                z.writestr(k + ".json", json.dumps(v))

    # Resume: first-wins merge of any pre-existing checkpoint (F-4).
    resumed = 0
    if os.path.exists(OUT_ZIP):
        with zipfile.ZipFile(OUT_ZIP) as z:
            for n in z.namelist():
                if n.endswith(".json"):
                    results[n[:-5]] = json.loads(z.read(n))
                    resumed += 1
        print(f"resumed {resumed} completed arm-calls from checkpoint", flush=True)

    def done(key):
        return key in results

    for w in warnings:
        wid, wtext = w["warning_id"], w["text"]
        raws.clear()
        try:
            # A0
            if not done(f"{wid}||A0"):
                raw, c = llm(A0_SYS, "Warning:\n" + wtext, "A0")
                results[f"{wid}||A0"] = {"belief": belief_of(parse_strict(raw), "A0"),
                                         "raw": raw, "parsed": parse_strict(raw)}
                logs.append({"arm": "A0", "warning_id": wid, "retrieval_calls": 0,
                             "stages": 1, **c})
                ckpt()
            # A1
            if not done(f"{wid}||A1"):
                raw, c = llm(A1_SYS, "Warning:\n" + wtext, "A1")
                o = parse_strict(raw)
                results[f"{wid}||A1"] = {"belief": belief_of(o, "A1"), "raw": raw,
                                         "parsed": o,
                                         "struct": {k: o[k] for k in
                                                    ("factual_claims", "support", "contradictions")}}
                logs.append({"arm": "A1", "warning_id": wid, "retrieval_calls": 0,
                             "stages": 1, **c})
                ckpt()
            # A2 (one verification)
            if not done(f"{wid}||A2"):
                raw1, c1 = llm(A1_SYS, "Warning:\n" + wtext, "A2a")
                o = parse_strict(raw1)
                raw2, c2 = llm(A2_SYS, "Warning:\n" + wtext + "\n\nInitial structured belief:\n" + raw1, "A2b")
                o2 = parse_strict(raw2)
                results[f"{wid}||A2"] = {"belief": belief_of(o2, "A2"),
                                         "raw_initial": raw1, "raw_verification": raw2,
                                         "struct": {"initial": o, "verification": o2}}
                logs.append({"arm": "A2", "warning_id": wid, "retrieval_calls": 0, "stages": 2,
                             "prompt_tokens": c1["prompt_tokens"] + c2["prompt_tokens"],
                             "completion_tokens": c1["completion_tokens"] + c2["completion_tokens"],
                             "latency_s": c1["latency_s"] + c2["latency_s"],
                             "finish_reason": [c1["finish_reason"], c2["finish_reason"]],
                             "truncated": c1["truncated"] or c2["truncated"]})
                ckpt()
            # R0 (verbatim query, 1 retrieval)
            if not done(f"{wid}||R0"):
                idx = bm25.topk(wtext, TOP_K)
                docs = [{"doc_id": doc_ids[i], "text": texts[i],
                         "text_sha16": sha16(texts[i])} for i in idx]
                ev = "\n\n".join(f"[DOC {d['doc_id']}] {d['text']}" for d in docs)
                raw, c = llm(R0_SYS, "Warning:\n" + wtext + "\n\nRetrieved evidence:\n" + ev, "R0")
                results[f"{wid}||R0"] = {"belief": belief_of(parse_strict(raw), "R0"),
                                         "raw": raw, "parsed": parse_strict(raw),
                                         "retrieval_query": wtext,
                                         "evidence_ids": [d["doc_id"] for d in docs],
                                         "evidence_sha16": [d["text_sha16"] for d in docs]}
                logs.append({"arm": "R0", "warning_id": wid, "retrieval_calls": 1,
                             "stages": 1, **c})
                ckpt()
            # A3 (hypotheses -> q1 -> optional q2 -> synth; <=2 retrievals)
            if not done(f"{wid}||A3"):
                raw_h, ch = llm(A3H_SYS, "Warning:\n" + wtext, "A3h")
                h = parse_strict(raw_h)
                q1 = h.get("query_1") or wtext
                d1 = [{"doc_id": doc_ids[i], "text": texts[i], "text_sha16": sha16(texts[i])}
                      for i in bm25.topk(q1, TOP_K)]
                ev1 = "\n\n".join(f"[DOC {d['doc_id']}] {d['text']}" for d in d1)
                raw_s, cs = llm(A3S_SYS, "Warning:\n" + wtext + "\n\nHypotheses:\n"
                                 + json.dumps(h.get("hypotheses")) + "\n\nEvidence round 1:\n" + ev1, "A3s")
                s = parse_strict(raw_s)
                docs_all, nret = list(d1), 1
                if s.get("query_2_used") and s.get("query_2"):
                    d2 = [{"doc_id": doc_ids[i], "text": texts[i], "text_sha16": sha16(texts[i])}
                          for i in bm25.topk(s["query_2"], TOP_K)]
                    docs_all += d2
                    nret = 2
                assert nret <= 2
                results[f"{wid}||A3"] = {"belief": belief_of(s, "A3"),
                                         "raw_hypotheses": raw_h, "raw_synthesis": raw_s,
                                         "struct": {"hypotheses": h.get("hypotheses"), "query_1": q1,
                                                    "query_2_used": bool(s.get("query_2_used")),
                                                    "query_2": s.get("query_2", "")},
                                         "retrieval_query_1": q1,
                                         "retrieval_query_2": s.get("query_2", "") if s.get("query_2_used") else "",
                                         "evidence_ids": [d["doc_id"] for d in docs_all],
                                         "evidence_sha16": [d["text_sha16"] for d in docs_all]}
                logs.append({"arm": "A3", "warning_id": wid, "retrieval_calls": nret, "stages": 2,
                             "prompt_tokens": ch["prompt_tokens"] + cs["prompt_tokens"],
                             "completion_tokens": ch["completion_tokens"] + cs["completion_tokens"],
                             "latency_s": ch["latency_s"] + cs["latency_s"],
                             "finish_reason": [ch["finish_reason"], cs["finish_reason"]],
                             "truncated": ch["truncated"] or cs["truncated"]})
                ckpt()
        except Exception as e:
            failures.append({"warning_id": wid, "error": repr(e),
                             "raws": {k: v[:2000] for k, v in raws.items()}})
            print(f"WARNING_FAIL {wid}: {e!r} | raws: "
                  f"{json.dumps({k: v[:500] for k, v in raws.items()})}", flush=True)
            ckpt()

    import numpy as np
    lat = np.array(latencies) if latencies else np.array([0.0])
    peak_mem = None
    try:
        import torch
        if torch.cuda.is_available():
            peak_mem = {"peak_alloc_GB": round(torch.cuda.max_memory_allocated(0) / 1e9, 3)}
    except Exception:
        pass
    manifest = {"model": MODEL, "revision": REVISION, "enable_thinking": False, "temperature": 0,
                "infra": {"vllm_pin": VLLM_PIN, "vllm_resolved": pkg_version("vllm"),
                          "openai_pin": OPENAI_PIN, "openai_resolved": pkg_version("openai"),
                          "transformers_resolved": pkg_version("transformers"),
                          "max_model_len": MAX_MODEL_LEN, "gpu_mem_util": GPU_MEM_UTIL,
                          "quantization": "awq", "network_policy": NETWORK_POLICY, **run_log},
                "runtime": {"gpu": gpu_info, "peak_mem": peak_mem,
                            "vllm_pid": server_pid,
                            "vllm_startup_s": srv_startup_s, "health_polls": polls,
                            "served": served, "hf_cache": hf_cache,
                            "latency_s": {"n": len(latencies), "mean": float(lat.mean()),
                                          "p50": float(np.quantile(lat, 0.5)),
                                          "p95": float(np.quantile(lat, 0.95)),
                                          "max": float(lat.max())},
                            "n_truncated": int(sum(1 for l in logs if l.get("truncated"))),
                            "elapsed_s": time.time() - t_all},
                "retriever": {"algorithm": "in-repo-BM25", "k1": K1, "b": B, "top_k": TOP_K},
                "budgets": BUDGET, "n_warnings": len(warnings), "n_llm_calls": n_calls,
                "n_resumed": resumed,
                "n_failures": len(failures), "failures": failures,
                "prompt_shas": {k: hashlib.sha256(v.encode()).hexdigest()
                                for k, v in {"A0": A0_SYS, "A1": A1_SYS, "A2": A2_SYS,
                                             "R0": R0_SYS, "A3H": A3H_SYS, "A3S": A3S_SYS}.items()},
                "call_logs": logs,
                "gate_zero_parse_failures": "PASS" if not failures else "FAIL"}
    open(OUT_MANIFEST, "w").write(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("n_warnings", "n_llm_calls", "n_failures",
                                              "gate_zero_parse_failures")}, indent=2), flush=True)
    if failures:
        raise SystemExit(f"SMOKE FAIL: {len(failures)} parse/schema failures (tolerance is 0)")
    print("SMOKE PASS: zero parse/schema failures", flush=True)


if __name__ == "__main__":
    main()
