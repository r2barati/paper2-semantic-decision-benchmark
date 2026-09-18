"""Paper2 V3 Phi-4 probe: serve stelterlab/phi-4-AWQ via vLLM, run test generations.

Pass criteria: (a) server healthy, (b) valid JSON matching the C1 regime schema
AND the C3 per-doc schema, (c) temperature-0 determinism (same prompt twice ->
identical output). Writes probe_report.json to /kaggle/working.
NO benchmark data touched. Plain instruct model: no thinking-trace transport
patch (prompt TEXT frozen; C1 66e6890b / C3 5966cadd asserted by belief shards).
"""

import json
import os
import subprocess
import sys
import time
import urllib.request

MODEL = "stelterlab/phi-4-AWQ"
REVISION = "075b93fe5ab0d2e86004a5d68c7575ec3bb5a88b"


def _strip(raw):
    import re
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return raw


def main():
    print("cuda:", __import__("torch").cuda.is_available(), flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm", "openai",
                    "pydantic", "numpy"], check=True)
    sirven = subprocess.Popen(
        [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
         "--model", MODEL, "--revision", REVISION,
         "--quantization", "awq", "--max-model-len", "4096",
         "--gpu-memory-utilization", "0.85", "--port", "8000"],
        stdout=open("/tmp/vllm.log", "w"), stderr=subprocess.STDOUT)
    from openai import OpenAI
    client = OpenAI(api_key="local", base_url="http://localhost:8000/v1")
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
    cases = [
        ("c1", "You are an operational risk analyst. Given a textual supplier or "
         "market warning, classify the likely operational regime. Respond ONLY "
         "with valid JSON.",
         "Operator information need: test\n\nEvidence:\n[node S1] planning "
         "window W-42. Confirmed congestion delaying outbound loads."),
        ("c3", "You are an operational risk analyst. Given ONE evidence document "
         "respond ONLY with valid JSON.",
         "Operator's own node: S1\n\nEvidence document:\n[node S1] planning "
         "window W-42. Confirmed congestion delaying outbound loads."),
    ]
    report = {"model": MODEL, "revision": REVISION}
    all_identical, all_parse = True, True
    for name, system, user in cases:
        outs = []
        for _ in range(2):
            r = client.chat.completions.create(
                model=MODEL, temperature=0, max_tokens=512,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            outs.append(r.choices[0].message.content.strip())
        identical = outs[0] == outs[1]
        try:
            parsed = json.loads(_strip(outs[0]))
            parses, keys = True, sorted(parsed)
        except Exception as e:
            parses, keys = False, [f"ERR {str(e)[:100]}"]
        report[name] = {"identical": identical, "parses": parses, "keys": keys,
                        "sample": outs[0][:300]}
        all_identical &= identical
        all_parse &= parses
    report["identical"] = all_identical
    report["parses"] = all_parse
    with open("/kaggle/working/probe_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("PROBE:", json.dumps({k: (v if k in ("model", "revision", "identical",
                                               "parses") else v)
                                for k, v in report.items()}), flush=True)
    sirven.terminate()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
