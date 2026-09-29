"""Generate the sealed Sem2Act v5 lockbox.

This generator has its own templates and provenance fields. It does not import
v3/v4 qrel-generation code. Output remains below the ignored v5 runtime tree.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "versions/sem2act-v5/runtime/lockbox"
SEED = 20260925
N_QUERIES = 240
REGIMES = ("supplier_delay", "demand_surge", "normal")
CURRENT = "planning window L-17"
STALE = "planning window L-04"

DELAY = (
    "Dock queue reports from {supplier} show outbound appointments slipping across the {site} lane.",
    "The carrier desk marked {supplier} capacity constrained while current loads await release.",
    "Receiving logs show {supplier} arrivals bunching behind the scheduled handoff.",
)
SURGE = (
    "Account teams report an unusual run of large replenishment requests for {site}.",
    "Order intake for the {site} network is running above its recent operating band.",
    "The commercial desk confirmed stronger pull from several customers this cycle.",
)
NORMAL = (
    "{supplier} is meeting the current delivery schedule and no unusual pull is reported.",
    "The {site} operating board remains within its expected supply and demand bands.",
    "Current receiving and order activity for {site} is routine.",
)
HEDGED = (
    "A preliminary note mentions possible movement, but the signal is not confirmed.",
    "One contact raised a watch item; no operational change is established yet.",
    "The latest note is inconclusive and asks planners to monitor the next update.",
)
CONTRA = (
    "Correction: the earlier alert was a tracking error and current operations are normal.",
    "Follow-up checks found no active disruption behind the earlier warning.",
)
OFFTOPIC = (
    "The facilities team is reviewing lighting upgrades for the next fiscal year.",
    "Finance circulated a revised expense coding guide for administrative purchases.",
    "The safety office scheduled a routine training session for warehouse staff.",
)
PREFIXES = ("NOVA", "ORBIT", "PRAIRIE", "SUMMIT", "EMBER", "CIRRUS",
            "LANTERN", "MOSAIC", "NIMBUS", "TIDELINE", "VISTA", "WILLOW")
SUFFIXES = ("POINT", "CROSSING", "RIDGE", "HARBOR")
SUPPLIERS = ("AXIOM-CARRIER", "BRIGHTLINE-LOGISTICS", "CANYON-FREIGHT",
             "DRIFTWOOD-TRANSIT", "EQUINOX-SHIPPING", "FARADAY-HAULAGE",
             "GATEWAY-CARRIERS", "HORIZON-LANE", "ION-FREIGHT",
             "JUNCTION-LOGISTICS", "KITEWAY-TRANSIT", "LUMEN-CARGO")


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_entities() -> list[dict[str, str]]:
    out = []
    for i, (prefix, suffix) in enumerate(
        ((p, s) for p in PREFIXES for s in SUFFIXES), start=1
    ):
        out.append({
            "node": f"LB{i:02d}",
            "site": f"{prefix}-{suffix}-DC",
            "supplier": SUPPLIERS[(i - 1) % len(SUPPLIERS)],
            "line": f"{7300 + i}",
        })
    return out


def make_doc(did, qid, ent, kind, regime, body, window, source):
    text = (
        f"[node {ent['node']} | site {ent['site']} | supplier {ent['supplier']} "
        f"| line {ent['line']}] {window}. {body}"
    )
    return {
        "_id": did,
        "title": source,
        "text": text,
        "metadata": {
            "query_id": qid,
            "entity_node": ent["node"],
            "site": ent["site"],
            "supplier": ent["supplier"],
            "kind": kind,
            "regime_described": regime,
            "window": window,
            "source": source,
        },
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    old_texts = set()
    old_entities = set()
    for path in (ROOT / "data/v3/corpus.jsonl", ROOT / "data/v3/queries.jsonl"):
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            row = json.loads(line)
            old_texts.add(norm(row.get("text", "")))
            meta = row.get("metadata", {})
            old_entities.update(str(meta.get(k, "")).lower()
                                for k in ("node", "entity_node", "site", "supplier")
                                if meta.get(k))
    entities = make_entities()
    new_identifiers = {
        x.lower() for e in entities for x in (e["node"], e["site"], e["supplier"])
    }
    if new_identifiers & old_entities:
        raise SystemExit("entity or supplier identifier overlap")
    if len({e["site"] for e in entities}) != 48:
        raise SystemExit("entity uniqueness failure")

    docs = []
    queries = []
    qrels = []
    provenance = []
    next_doc = 1
    regimes = [REGIMES[i // 80] for i in range(N_QUERIES)]
    order = rng.permutation(N_QUERIES)

    for ordinal, qi in enumerate(order):
        ent = entities[ordinal % len(entities)]
        regime = regimes[int(qi)]
        qid = f"v5-lb-q{int(qi) + 1:04d}"
        query = (
            f"Current supply and demand conditions for node {ent['node']} at "
            f"{ent['site']} with {ent['supplier']} during {CURRENT}."
        )
        qrow = {
            "_id": qid,
            "text": query,
            "metadata": {
                "true_regime": regime,
                "entity_node": ent["node"],
                "site": ent["site"],
                "supplier": ent["supplier"],
                "window": CURRENT,
                "lockbox_ordinal": int(qi),
            },
        }
        if norm(query) in old_texts:
            raise SystemExit(f"query overlap: {qid}")
        queries.append(qrow)

        choices = []
        for kind, grade, window, bank in (
            ("grade3", 3, CURRENT, {"supplier_delay": DELAY, "demand_surge": SURGE, "normal": NORMAL}[regime]),
            ("grade2", 2, CURRENT, HEDGED),
            ("stale", 1, STALE, {"supplier_delay": DELAY, "demand_surge": SURGE, "normal": NORMAL}[regime]),
            ("contradictory", 0, CURRENT, CONTRA),
        ):
            n = {"grade3": 3, "grade2": 3, "stale": 3, "contradictory": 2}[kind]
            for j in range(n):
                body = bank[(int(qi) + j) % len(bank)].format(**ent)
                choices.append((kind, grade, window, body, f"{kind}-{j}"))

        for j in range(3):
            other = entities[(ordinal + j + 1) % len(entities)]
            body = DELAY[j % len(DELAY)].format(**other)
            choices.append(("wrong_entity", 0, CURRENT, body, f"wrong-{j}"))
        for j in range(2):
            choices.append(("offtopic", 0, CURRENT,
                            OFFTOPIC[(int(qi) + j) % len(OFFTOPIC)], f"off-{j}"))

        for kind, grade, window, body, source in choices:
            did = f"v5-lb-d{next_doc:06d}"
            next_doc += 1
            owner = ent
            if kind == "wrong_entity":
                owner = entities[(ordinal + int(source[-1]) + 1) % len(entities)]
            d = make_doc(
                did, qid, owner, kind,
                regime if kind in {"grade3", "grade2", "stale"} else None,
                body, window, source,
            )
            if norm(d["text"]) in old_texts:
                raise SystemExit(f"document overlap: {did}")
            docs.append(d)
            qrels.append((qid, did, grade))
            provenance.append({
                "query_id": qid,
                "document_id": did,
                "query_entity": ent["node"],
                "document_entity": owner["node"],
                "kind": kind,
                "regime": regime,
                "window": window,
                "grade": grade,
                "source": source,
            })

    for i in range(240):
        did = f"v5-lb-d{next_doc:06d}"
        next_doc += 1
        ent = entities[i % len(entities)]
        docs.append(make_doc(
            did, "shared", ent, "shared", None,
            OFFTOPIC[i % len(OFFTOPIC)], CURRENT, "shared",
        ))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "private").mkdir(exist_ok=True)
    (OUT / "qrels_free").mkdir(exist_ok=True)
    (OUT / "queries.jsonl").write_text(
        "".join(json.dumps(q) + "\n" for q in queries)
    )
    (OUT / "corpus.jsonl").write_text(
        "".join(json.dumps(d) + "\n" for d in docs)
    )
    (OUT / "private/qrels.tsv").write_text(
        "query-id\tcorpus-id\tscore\n" +
        "".join(f"{q}\t{d}\t{g}\n" for q, d, g in qrels)
    )
    (OUT / "private/provenance.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in provenance)
    )
    (OUT / "qrels_free/queries.jsonl").write_text((OUT / "queries.jsonl").read_text())
    (OUT / "qrels_free/corpus.jsonl").write_text((OUT / "corpus.jsonl").read_text())

    manifest = {
        "schema_version": 1,
        "version_id": "sem2act-v5",
        "generation_seed": SEED,
        "n_queries": len(queries),
        "n_documents": len(docs),
        "regime_counts": {
            r: sum(q["metadata"]["true_regime"] == r for q in queries)
            for r in REGIMES
        },
        "n_entities": len(entities),
        "n_qrel_rows": len(qrels),
        "qrels_private": True,
        "source": str(Path(__file__).relative_to(ROOT)),
        "files": {
            str(p.relative_to(OUT)): sha(p)
            for p in (
                OUT / "queries.jsonl",
                OUT / "corpus.jsonl",
                OUT / "qrels_free/queries.jsonl",
                OUT / "qrels_free/corpus.jsonl",
                OUT / "private/qrels.tsv",
                OUT / "private/provenance.jsonl",
            )
        },
    }
    (OUT / "generated_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
