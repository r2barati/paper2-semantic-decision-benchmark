"""Paper2 V3 Qwen14 probe: serve Qwen3-8B-AWQ via vLLM, run 3 test generations.

Pass criteria: (a) server healthy, (b) valid JSON matching the C3 per-doc schema,
(c) temperature-0 determinism (same prompt twice -> identical output).
Writes probe_report.json to /kaggle/working. NO benchmark data touched.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request

MODEL = "Qwen/Qwen3-14B-AWQ"
REVISION = "31c69efc29464b6bb0aee1398b5a7b50a99340c3"


def main():
    print("cuda:", __import__("torch").cuda.is_available(), flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm", "openai"],
                   check=True)
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
    prompt = ("Respond ONLY with valid JSON matching this exact schema:\n"
              '{"entity_match": <true|false>, "event": <"supplier_delay"|"demand_surge"|"normal"|"none">, '
              '"fresh": <true|false>, "stance": <"support"|"refute"|"na">, "confidence": <float 0-1>}')
    user = ("Operator's own node: S1\n\nEvidence document:\n"
            "[node S1] planning window W-42. Confirmed congestion delaying outbound loads.")
    outs = []
    for _ in range(2):
        r = client.chat.completions.create(
            model=MODEL, temperature=0, max_tokens=1024,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            messages=[{"role": "system", "content": prompt},
                      {"role": "user", "content": user}])
        outs.append(r.choices[0].message.content.strip())
    report = {"model": MODEL, "revision": REVISION, "identical": outs[0] == outs[1],
              "sample": outs[0][:500]}

    def _strip(raw):
        import re
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return raw
    try:
        parsed = json.loads(_strip(outs[0]))
        report["parses"] = True
        report["keys"] = sorted(parsed)
    except Exception as e:
        report["parses"] = False
        report["parse_error"] = str(e)[:200]
    with open("/kaggle/working/probe_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("PROBE:", json.dumps({k: v for k, v in report.items() if k != "sample"}),
          flush=True)
    sirven.terminate()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
