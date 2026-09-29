"""Assemble and validate frozen v5 belief outputs.

This step reads only model-output beliefs and run manifests. It never reads
constructed qrels or provenance labels.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "versions/sem2act-v5/manifests/protocol_freeze.json"
OUTPUTS = ROOT / "versions/sem2act-v5/runtime/model_outputs"
OUT = ROOT / "versions/sem2act-v5/runtime/assembled_beliefs.jsonl"
MODELS = {
    "Qwen/Qwen3-8B-AWQ": ("bm25", "rerank", "oracle"),
    "meta-llama/Llama-3.1-8B-Instruct": ("rerank", "oracle"),
    "mistralai/Mistral-7B-Instruct-v0.3": ("rerank", "oracle"),
}
CONSUMERS = ("C1", "C3")
REGIMES = ("normal", "supplier_delay", "demand_surge")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_keys(systems):
    return {
        (f"v5-lb-q{i:04d}", system, consumer)
        for i in range(1, 241)
        for system in systems
        for consumer in CONSUMERS
    }


def load_family(model_id: str, systems: tuple[str, ...]) -> tuple[list[dict], dict]:
    family = {
        "Qwen/Qwen3-8B-AWQ": "qwen",
        "meta-llama/Llama-3.1-8B-Instruct": "llama",
        "mistralai/Mistral-7B-Instruct-v0.3": "mistral",
    }[model_id]
    directory = OUTPUTS / family
    beliefs = directory / "beliefs.jsonl"
    run_manifest = directory / "run_manifest.json"
    if not beliefs.exists() or not run_manifest.exists():
        raise SystemExit(f"missing frozen {family} belief outputs")
    manifest = json.loads(run_manifest.read_text())
    if manifest.get("status") != "pass" or manifest.get("qrels_read") is not False:
        raise SystemExit(f"{family} run manifest is not accepted")
    rows = [json.loads(line) for line in beliefs.read_text().splitlines() if line]
    if len(rows) != len(expected_keys(systems)):
        raise SystemExit(f"{family}: expected {len(expected_keys(systems))} rows, got {len(rows)}")
    keys = {(row["query_id"], row["system"], row["consumer"]) for row in rows}
    if keys != expected_keys(systems) or len(keys) != len(rows):
        raise SystemExit(f"{family}: belief keyspace failed")
    for row in rows:
        if row.get("model") != model_id or row.get("k") != 3:
            raise SystemExit(f"{family}: model/k drift")
        if row.get("consumer") not in CONSUMERS or row.get("system") not in systems:
            raise SystemExit(f"{family}: unexpected arm")
        if row.get("true_regime") not in REGIMES:
            raise SystemExit(f"{family}: missing true regime")
        values = [float(row[name]) for name in (
            "p_normal", "p_supplier_delay", "p_demand_surge"
        )]
        if any(value < 0 or value > 1 for value in values) or abs(sum(values) - 1) > 1e-6:
            raise SystemExit(f"{family}: invalid belief simplex")
    return rows, {
        "model": model_id,
        "family": family,
        "beliefs_sha256": sha(beliefs),
        "run_manifest_sha256": sha(run_manifest),
        "n_rows": len(rows),
    }


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text())
    all_rows, families = [], []
    for model_id, systems in MODELS.items():
        rows, record = load_family(model_id, systems)
        all_rows.extend(rows)
        families.append(record)
    all_rows.sort(key=lambda row: (row["model"], row["system"], row["consumer"], row["query_id"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in all_rows))
    manifest = {
        "schema_version": 1,
        "experiment_id": "v5-belief-assembly",
        "status": "frozen",
        "protocol_hash": protocol["protocol_hash"],
        "qrels_read": False,
        "families": families,
        "n_rows": len(all_rows),
        "output_sha256": sha(OUT),
        "output": str(OUT.relative_to(ROOT)),
    }
    out_manifest = ROOT / "versions/sem2act-v5/manifests/belief_assembly.json"
    out_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
