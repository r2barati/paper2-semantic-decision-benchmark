"""Verify and accept a non-result official Kaggle model source probe."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ATTEMPTS = ROOT / "versions/sem2act-v5/manifests/execution_attempts"
DEST = ROOT / "versions/sem2act-v5/manifests/cpu_source_probes"
EXPECTED = {
    "qwen": {"source_ref": "qwen-lm/qwen-3/Transformers/8b-awq", "owner": "qwen-lm", "publisher": "QwenLM", "version_number": 1, "version_id": 391615, "model_type": "qwen3", "architecture": "Qwen3ForCausalLM"},
    "llama": {"source_ref": "metaresearch/llama-3.1/Transformers/8b-instruct", "owner": "metaresearch", "publisher": "Meta", "version_number": 2, "version_id": 104449, "model_type": "llama", "architecture": "LlamaForCausalLM"},
}
def sha(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as h:
        for block in iter(lambda:h.read(1024*1024), b""): digest.update(block)
    return digest.hexdigest()
def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("family", choices=sorted(EXPECTED)); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--kaggle", default="/usr/local/bin/kaggle"); parser.add_argument("--fetch", action="store_true"); args=parser.parse_args()
    attempt_path=ATTEMPTS / f"cpu-model-source-probe-{args.family}.json"
    attempt=json.loads(attempt_path.read_text())
    if attempt.get("status") != "submitted-non-result-source-probe": raise SystemExit("source probe was not submitted")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.fetch:
        subprocess.run([args.kaggle,"kernels","output",attempt["slug"],"-p",str(args.output_dir),"--force"],check=True)
    remote_path=args.output_dir / "source_probe_manifest.json"
    if not remote_path.exists(): raise SystemExit("missing source_probe_manifest.json")
    remote=json.loads(remote_path.read_text())
    if remote.get("status") != "pass" or remote.get("source_only") is not True or remote.get("result_bearing_execution_started") is not False or remote.get("qrels_read") is not False: raise SystemExit("source probe did not pass fail-closed checks")
    source=remote.get("source") or {}
    for key,value in EXPECTED[args.family].items():
        if source.get(key) != value: raise SystemExit(f"source metadata mismatch: {key}")
    if not str(source.get("resolved_model_path","")).startswith("/kaggle/input/"): raise SystemExit("source was not mounted from Kaggle model input")
    snap=remote.get("source_model_snapshot") or {}
    if not snap.get("files_sha256") or not snap.get("file_count"): raise SystemExit("source content snapshot missing")
    if remote.get("inference_calls") != 0: raise SystemExit("source probe performed inference")
    dest=DEST / args.family
    if dest.exists(): shutil.rmtree(dest)
    shutil.copytree(args.output_dir,dest)
    accepted={"schema_version":1,"manifest_id":f"sem2act-v5-source-probe-accepted-{args.family}-v1","status":"accepted-non-result-source-probe","accepted_utc":datetime.now(timezone.utc).isoformat(),"family":args.family,"attempt_manifest_sha256":sha(attempt_path),"remote_manifest_sha256":sha(remote_path),"source":source,"source_model_snapshot":snap,"result_bearing_execution_started":False,"qrels_read":False}
    (dest / "acceptance_manifest.json").write_text(json.dumps(accepted,indent=2,sort_keys=True)+"\n")
    print(json.dumps(accepted,indent=2,sort_keys=True))
    return 0
if __name__ == "__main__": raise SystemExit(main())
