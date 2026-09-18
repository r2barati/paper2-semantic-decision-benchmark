"""Merge fetched cache shards into a local cache dir (no API calls).

Deterministic first-wins policy: shards are merged in numeric shard-start
order; on filename collision with different bytes the FIRST shard's file is
kept and the event is recorded in <cache_dir>_nondeterminism.json. This
handles cross-session vLLM non-determinism on content-shared keys (same query
text + same docs across query IDs): the merge output is fully reproducible
given the same input zips, and every divergent case is auditable.

Then assemble with (all cache hits):
  PAPER2_V3_CACHE_DIR=results/v3main/beliefs_cache_awq \
  python3 -m src.beliefs_v3main --consumers C0 C1 C3 --model Qwen/Qwen3-8B-AWQ
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _shard_key(p: Path):
    m = re.match(r"cache_shard_(\d+)_(\d+)\.zip", p.name)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def _strip(raw: str) -> str:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return raw


def _belief_summary(payload: dict):
    """Comparable summary of parsed belief; None if unparseable."""
    try:
        p = json.loads(_strip(payload["raw"]))
        if "normal" in p:  # C1
            s = p["normal"] + p["supplier_delay"] + p["demand_surge"]
            return ("C1", round(p["normal"] / s, 6),
                    round(p["supplier_delay"] / s, 6),
                    round(p["demand_surge"] / s, 6))
        return ("C3", p.get("entity_match"), p.get("event"),
                p.get("fresh"), p.get("stance"), p.get("confidence"))
    except Exception:
        return None


def main(shard_dir=ROOT / "runs" / "v3main_awq",
         cache_dir=ROOT / "results" / "v3main" / "beliefs_cache_awq"):
    zips = sorted(Path(shard_dir).glob("cache_shard_*.zip"), key=_shard_key)
    if not zips:
        raise SystemExit(f"no shards in {shard_dir}")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    diverged = []
    for z in zips:
        with zipfile.ZipFile(z) as zh:
            names = zh.namelist()
            kept_new = kept_old = 0
            for fn in names:
                data = zh.read(fn)
                target = Path(cache_dir) / fn
                if target.exists():
                    old = target.read_bytes()
                    if old != data:
                        try:
                            old_j = json.loads(old)
                            new_j = json.loads(data)
                        except Exception:
                            old_j = new_j = {}
                        same_belief = (_belief_summary(old_j) is not None
                                       and _belief_summary(old_j) == _belief_summary(new_j))
                        diverged.append({
                            "file": fn, "kept_shard": "earlier",
                            "later_shard": z.name, "same_parsed_belief": same_belief,
                            "kept_raw": old_j.get("raw", "")[:300] if isinstance(old_j, dict) else "",
                            "later_raw": new_j.get("raw", "")[:300] if isinstance(new_j, dict) else "",
                        })
                        kept_old += 1
                        continue
                    kept_old += 1
                else:
                    target.write_bytes(data)
                    kept_new += 1
        print(f"{z.name}: {len(names)} entries ({kept_new} new, {kept_old} already present)")
    man_path = Path(cache_dir).parent / (Path(cache_dir).name + "_nondeterminism.json")
    man_path.write_text(json.dumps({
        "policy": "first-wins in numeric shard-start order",
        "n_diverged_files": len(diverged),
        "diverged": diverged,
    }, indent=1))
    print(f"cache: {len(list(Path(cache_dir).glob('*.json')))} files from {len(zips)} shards")
    print(f"diverged (kept first): {len(diverged)} -> {man_path.name}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--shard-dir", default=str(ROOT / "runs" / "v3main_awq"))
    ap.add_argument("--cache-dir",
                    default=str(ROOT / "results" / "v3main" / "beliefs_cache_awq"))
    a = ap.parse_args()
    if a.merge:
        main(a.shard_dir, a.cache_dir)
    else:
        print(__doc__)
