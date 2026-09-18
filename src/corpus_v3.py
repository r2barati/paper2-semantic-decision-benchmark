"""V3 Phase-1 prototype corpus (DecisionIR-Bench proto).

20 queries / ~300 docs with graded qrels (0-3, entity/recency/aboutness only)
and SEPARATE hidden operational labels. Fresh-authored styles dominate
(>=70%); existing V1/V2 template banks are usable as seed facts only and are
NOT imported here, so verbatim leakage is impossible by construction.

Deterministic given seed. All IDs persistent: v3p-q{:02d} / v3p-d{:04d}.
Prototype IDs are quarantined from the V3 full benchmark.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
RULES = yaml.safe_load((ROOT / "configs" / "v3" / "dataset_proto.yaml").read_text())

CURRENT_WINDOW = "planning window W-42"
PAST_WINDOW = "planning window W-31"

REGIMES = ["supplier_delay", "demand_surge", "normal"]

ENTITIES = [
    {"node": "R3", "site": "CEDAR-FALLS-DC", "supplier": "NORTHBRIDGE-LOGISTICS", "line": "7742"},
    {"node": "R1", "site": "ALDERGROVE-DC", "supplier": "WESTMARK-FREIGHT", "line": "3310"},
    {"node": "R2", "site": "BRIARWOOD-DC", "supplier": "COASTLINE-CARRIERS", "line": "5128"},
    {"node": "R4", "site": "DUNMORE-DC", "supplier": "STONEGATE-HAULAGE", "line": "9014"},  # held-out
]
HELD_OUT_NODE = "R4"


def _header(ent, window):
    return (f"[node {ent['node']} | site {ent['site']} | supplier {ent['supplier']} "
            f"| line {ent['line']}] {window}.")


# ---------------------------------------------------------------------------
# Fresh-authored content banks (style x regime). Slots: {site}, {supplier}.
# Deliberately partial: no text states regime + full parameters jointly.
# ---------------------------------------------------------------------------

_DELAY_CLEAR = [
    ("supplier status report",
     "Confirmed congestion at the {supplier} facility is delaying outbound loads. "
     "Dispatch lists show multi-day slips on most departures this week."),
    ("logistics notice",
     "URGENT: carrier {supplier} reports a facility disruption affecting scheduled pickups. "
     "Several loads are held pending dock availability."),
    ("email",
     "Subject: inbound delays. Team — {supplier} just notified us of congestion at their hub. "
     "Expect late arrivals on current bookings; I will update the ETA board as releases come in."),
    ("internal note",
     "Ops note: {supplier} hub throughput down. Planner advised to cover near-term needs "
     "from safety stock until flow normalises."),
    ("shipment notice",
     "Shipment notice amended: consignments tendered to {supplier} are departing behind schedule. "
     "Revised arrival windows to follow per load."),
    ("procurement update",
     "Procurement update: spot cover requested while {supplier} works through its backlog. "
     "No change to contracted volumes at this stage."),
    ("industry report",
     "Trade brief: port congestion upstream of {supplier} is cascading to inland hubs. "
     "Dwell times extended three days running."),
    ("email",
     "Subject: slipped loads. Two {supplier} trailers missed their windows this week. "
     "Carrier cites yard congestion; revised ETAs pending."),
    ("logistics notice",
     "Lane advisory: {site}-bound freight via {supplier} moving slow. "
     "Build buffer into this week's replenishment."),
    ("supplier status report",
     "Confirmed: {supplier} operating with reduced dock crews. Loading queues "
     "reported at morning and evening shifts."),
    ("internal note",
     "Ops note: inbound fill from {supplier} dropped below plan two days straight. "
     "Shortage risk flagged for fast movers."),
]

_DELAY_HEDGED = [
    ("supplier status report",
     "Unconfirmed reports of slower handling at the {supplier} facility. "
     "No revised ETAs issued yet; monitoring departure scans."),
    ("email",
     "Subject: possible slowness. Heard from two drivers that {supplier} docks are slow today. "
     "Could be nothing — keeping an eye on tomorrow's arrivals."),
    ("internal note",
     "Rumor log: {site} staff mention upstream transport rumblings. Nothing confirmed; "
     "no action beyond watch-listing next week's inbound."),
    ("logistics notice",
     "Lane watch: minor dwell-time uptick on {supplier} lanes. Could be weather; "
     "no revised ETAs at this time."),
    ("email",
     "Subject: fyi. {supplier} mentioned staffing gaps on our call. No impact stated — "
     "logging here in case it develops."),
]

_SURGE_CLEAR = [
    ("market report",
     "Confirmed: two major accounts placed unusually large orders for delivery over the coming weeks. "
     "Order desk volumes are running well above the recent norm."),
    ("email",
     "Subject: big week ahead. Sales closed a large contract — expect customer demand to run hot. "
     "Please make sure replenishment plans reflect the uplift."),
    ("procurement update",
     "Procurement update: forward cover increased after strong order intake. "
     "Customer pull is the driver, not a supply event."),
    ("internal note",
     "Ops note: pick volumes surging on the {site} line. Overtime authorised to keep pace with orders."),
    ("industry report",
     "Trade brief: regional buying surge reported across our customer segment this window. "
     "Distributors describe full order books."),
    ("market report",
     "Demand alert: inbound customer inquiries at multi-week highs. Conversion rates normal, "
     "so realised demand is expected to follow."),
    ("email",
     "Subject: restock rush. Key account doubling its usual pull for the next few cycles. "
     "Please expedite replenishment to avoid gaps."),
    ("industry report",
     "Sector note: sell-through accelerating across our category. Channel checks point "
     "to sustained rather than one-off buying."),
    ("internal note",
     "Ops note: backorders ticking up as order sizes grow. Pick capacity under review."),
    ("procurement update",
     "Procurement update: safety-stock targets raised after demand forecast revision. "
     "Driver is customer orders, confirmed by sales."),
]

_SURGE_HEDGED = [
    ("email",
     "Subject: maybe busy? A few customers asking about expedited slots. Might be noise — "
     "flagging in case the pattern holds."),
    ("internal note",
     "Watch item: {site} order inflow slightly elevated. Within recent variability so far."),
    ("market report",
     "Analyst chatter about firmer demand in the segment. No confirmed orders attributable yet."),
    ("email",
     "Subject: busy signals? Website traffic and quote requests both up this week. "
     "Too early to call it a trend."),
    ("internal note",
     "Watch item: repeat-order rate creeping up at {site}. Still inside planning bands."),
]

_NORMAL_CLEAR = [
    ("supplier status report",
     "Routine status: {supplier} operating normally. Departures on schedule, no backlog reported."),
    ("logistics notice",
     "Standard notice: all lanes to {site} running to plan. No disruptions expected this window."),
    ("internal note",
     "Ops note: {site} inventory within expected ranges. No unusual demand or supply signals."),
    ("shipment notice",
     "Shipment notice: loads tendered to {supplier} departing as booked. ETAs unchanged."),
    ("email",
     "Subject: weekly check. All quiet on inbound — suppliers on time, demand steady. No action needed."),
    ("supplier status report",
     "Scorecard: {supplier} met all service targets this window. On-time rate nominal."),
    ("logistics notice",
     "Lane review: {site} inbound balanced with outbound. No backlog, no expedites required."),
    ("internal note",
     "Ops note: cycle count reconciled. {site} positions healthy across SKUs."),
]

_STALE_PREFIX = "RESOLVED / CLOSED ({window}): "

_CONTRADICT = {
    "supplier_delay": [
        ("supplier status report",
         "Correction: earlier rumors of {supplier} congestion are unfounded. An audit of departure scans "
         "shows normal flow; no delays recorded this window."),
        ("email",
         "Subject: false alarm. {supplier} confirms operations normal — yesterday's slowdown was a "
         "system outage in tracking, not physical congestion."),
    ],
    "demand_surge": [
        ("market report",
         "Correction: the rumored large orders did not materialize; quoted volumes were not converted. "
         "Demand remains at normal levels."),
        ("internal note",
         "Ops note: order inflow back to baseline after a one-day data-entry duplication inflated the board. "
         "No surge in progress."),
    ],
    "normal": [
        ("email",
         "Subject: ignore rumor. Talk of a disruption affecting our lanes is baseless; "
         "all partners confirm normal operations."),
    ],
}

_OFFTOPIC_LEXICAL = [  # shares IR vocabulary, operationally void
    "Facilities memo: lead time for replacement forklift batteries is six weeks. Quotes attached for approval.",
    "Office supply order delayed: the stationery supplier cites printer-cartridge shortages. No impact on operations.",
    "IT notice: dashboard shipment-tracking tiles will load slowly during the weekend migration. Data unaffected.",
    "Travel desk: supplier-visit flights must be booked two weeks ahead under the revised policy.",
    "Canteen supplier changed its delivery slot to mornings. Vending restock unaffected.",
    "Procurement (MRO): delayed delivery of safety gloves from the uniform supplier. Warehouse ops continue normally.",
]

_OFFTOPIC_SEMANTIC = [  # operations-adjacent, different aspect
    "Safety audit scheduled at the warehouse next month. No operational changes required beforehand.",
    "Sustainability: scope-3 emissions re-baselining for logistics data. Reporting-only exercise.",
    "Warehouse expansion planning kickoff: capacity review for next fiscal year. No near-term moves.",
    "HR bulletin: forklift recertification window open. Rostering to accommodate training slots.",
    "Finance: freight accrual coding changes for quarterly close. Bookkeeping only.",
    "Legal: updated supplier code of conduct published. No action for existing contracts.",
]

_SHARED_ROUTINE = [
    "Payroll calendar published for the quarter. No change to deposit schedule.",
    "Second-floor printer replaced. Reinstall drivers from the software catalogue.",
    "Parking structure resurfacing continues. Overflow lot available.",
    "Benefits open enrolment closes end of month. Complete elections in the portal.",
    "Multi-factor authentication enforced for vendor portals. Contact the service desk.",
    "Annual safety certification window open for warehouse staff.",
    "Expense reports due Friday. Code freight accruals per the new schedule.",
    "Autumn campaign creative review moved. Regional teams submit assets on the revised date.",
    "Scope-3 reporting boundaries revised. Annual disclosure re-baselined.",
    "International travel now needs advance approval. Submit requests two weeks ahead.",
    "Updated code of conduct published. No action for existing contracts.",
    "Printer driver reinstallation required after the replacement program.",
    "Canteen menu rotation for the month. Allergen cards updated.",
    "Fire drill scheduled Thursday. Muster points unchanged.",
    "Holiday shutdown calendar circulated for planning purposes.",
    "Badge access audit next week. Report lost cards promptly.",
    "New starter inductions every Monday. Book rooms via facilities.",
    "Water cooler maintenance Tuesday morning. Brief outage expected.",
    "Charity drive collection point in the lobby through month end.",
    "Bike shelter cleaning scheduled. Move locks by Friday.",
]


@dataclass
class Doc:
    doc_id: str
    title: str
    text: str
    kind: str                 # grade3/grade2/stale/contra/wrong_entity/offtopic/shared
    regime_described: str | None
    entity_node: str | None
    window: str | None
    source: str               # style or bank
    contradicts_id: str | None = None
    provenance: str = "fresh"  # fresh | seedfact
    seed_fact: str | None = None

    def as_dict(self):
        return asdict(self)


@dataclass
class Query:
    query_id: str
    text: str
    true_regime: str
    entity_node: str
    split: str  # dev | test


_FORBIDDEN = [
    re.compile(r"lead time (of 4|from 4).{0,60}8 period", re.I),
    re.compile(r"period 20.{0,80}(10 period|last.{0,20}10)", re.I),
    re.compile(r"demand.{0,40}(8 units|normal 8).{0,40}14", re.I),
]


def check_giveaways(text):
    """Forbidden joint regime+parameter statements. Raises on violation."""
    return [p.pattern for p in _FORBIDDEN if p.search(text)]


def build_proto_corpus(seed=20260913):
    rng = np.random.default_rng(seed)
    docs: dict[str, Doc] = {}
    queries: list[Query] = []
    qrels: dict[str, dict[str, int]] = {}   # qid -> {did: grade}
    ev: list[dict] = []                      # operational labels (hidden)
    did = [0]
    seen_text: dict[str, str] = {}           # text sha -> doc_id (shared-pool dedup)

    def new_doc(title, text, kind, regime, ent, window, source, contradicts=None):
        h = sha256_text(text)
        if h in seen_text:
            return docs[seen_text[h]]  # identical bulletin shared across pools, one ID
        did[0] += 1
        d = Doc(f"v3p-d{did[0]:04d}", title, text, kind, regime, ent, window,
                source, contradicts)
        assert not check_giveaways(text), f"giveaway in {d.doc_id}"
        docs[d.doc_id] = d
        seen_text[h] = d.doc_id
        return d

    # query plan: 20 queries, stratified regimes, round-robin entities (R4 x4 held-out)
    regimes = (["supplier_delay"] * 8 + ["demand_surge"] * 8 + ["normal"] * 4)
    order = rng.permutation(len(regimes))
    ent_cycle = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 0, 3, 3]
    held_out_idx = {i for i, e in enumerate(ent_cycle) if e == 3}
    dev_pool = [i for i in range(20) if i not in held_out_idx]
    dev_idx = set(rng.choice(dev_pool, size=6, replace=False).tolist())

    for qi in range(20):
        reg = regimes[int(order[qi])]
        ent = ENTITIES[ent_cycle[qi]]
        qid = f"v3p-q{qi + 1:02d}"
        split = "dev" if qi in dev_idx else "test"
        qtext = (f"Current inbound supply, lead time and demand conditions for node {ent['node']} "
                 f"site {ent['site']} supplier {ent['supplier']} in {CURRENT_WINDOW} "
                 f"affecting replenishment now.")
        queries.append(Query(qid, qtext, reg, ent["node"], split))
        pool: dict[str, int] = {}

        def scoped(t, e, w=CURRENT_WINDOW):
            return f"{_header(e, w)} {t}"

        bank_clear = {"supplier_delay": _DELAY_CLEAR, "demand_surge": _SURGE_CLEAR,
                      "normal": _NORMAL_CLEAR}[reg]
        bank_hedge = {"supplier_delay": _DELAY_HEDGED, "demand_surge": _SURGE_HEDGED,
                      "normal": _NORMAL_CLEAR}[reg]

        # grade 3 (2): own node, current, faithful
        for style, t in [bank_clear[i % len(bank_clear)] for i in
                         rng.choice(len(bank_clear), size=2, replace=False)]:
            d = new_doc(f"{style} ({ent['node']})", scoped(t.format(**ent), ent),
                        "grade3", reg, ent["node"], CURRENT_WINDOW, style)
            pool[d.doc_id] = 3
        # grade 2 (2): own node, current, hedged
        for style, t in [bank_hedge[i % len(bank_hedge)] for i in
                         rng.choice(len(bank_hedge), size=2, replace=False)]:
            d = new_doc(f"{style} ({ent['node']})", scoped(t.format(**ent), ent),
                        "grade2", reg, ent["node"], CURRENT_WINDOW, style)
            pool[d.doc_id] = 2
        # grade 1 stale (2, or 1+contra handled below): own node, past window
        stale_src = bank_clear
        for i in rng.choice(len(stale_src), size=2, replace=False):
            style, t = stale_src[int(i)]
            body = _STALE_PREFIX.format(window=PAST_WINDOW) + t.format(**ent)
            d = new_doc(f"{style} [stale] ({ent['node']})",
                        f"{_header(ent, PAST_WINDOW)} {body}",
                        "stale", reg, ent["node"], PAST_WINDOW, style)
            pool[d.doc_id] = 1
        # contradictory (1) on 10 queries
        if qi % 2 == 0:
            cb = _CONTRADICT[reg]
            style, t = cb[int(rng.integers(len(cb)))]
            target = [k for k, v in pool.items() if v == 3][0]
            d = new_doc(f"{style} [correction] ({ent['node']})",
                        scoped(t.format(**ent), ent),
                        "contra", reg, ent["node"], CURRENT_WINDOW, style,
                        contradicts=target)
            docs[target].contradicts_id = d.doc_id
            pool[d.doc_id] = 0
        # wrong entity (3): other nodes; 1 misleading / 1 accidentally-correct / 1 neutral
        others = [e for e in ENTITIES if e["node"] != ent["node"]]
        err_regimes = [r for r in REGIMES if r != reg][:1] + [reg] + ["normal"]
        for j, er in enumerate(err_regimes):
            oe = others[int(rng.integers(len(others)))]
            eb = {"supplier_delay": _DELAY_CLEAR, "demand_surge": _SURGE_CLEAR,
                  "normal": _NORMAL_CLEAR}[er]
            style, t = eb[int(rng.integers(len(eb)))]
            d = new_doc(f"{style} ({oe['node']})", scoped(t.format(**oe), oe),
                        "wrong_entity", er, oe["node"], CURRENT_WINDOW, style)
            pool[d.doc_id] = 0
        # off-topic (5): 2 lexical + 2 semantic + 1 routine, own node (entity alone can't filter)
        for t in [ _OFFTOPIC_LEXICAL[i] for i in rng.choice(len(_OFFTOPIC_LEXICAL), size=2, replace=False)]:
            d = new_doc("Operations memo", scoped(t, ent), "offtopic", None,
                        ent["node"], CURRENT_WINDOW, "offtopic-lexical")
            pool[d.doc_id] = 0
        for t in [_OFFTOPIC_SEMANTIC[i] for i in rng.choice(len(_OFFTOPIC_SEMANTIC), size=2, replace=False)]:
            d = new_doc("Operations memo", scoped(t, ent), "offtopic", None,
                        ent["node"], CURRENT_WINDOW, "offtopic-semantic")
            pool[d.doc_id] = 0
        rt = _SHARED_ROUTINE[int(rng.integers(len(_SHARED_ROUTINE)))]
        d = new_doc("Corporate bulletin", scoped(rt, ent), "offtopic", None,
                    ent["node"], CURRENT_WINDOW, "routine")
        pool[d.doc_id] = 0

        qrels[qid] = pool
        for dd, g in pool.items():
            doc = docs[dd]
            ev.append({"query_id": qid, "doc_id": dd, "qrel": g,
                       "true_regime": reg, "entity_match": doc.entity_node == ent["node"],
                       "fresh": doc.window == CURRENT_WINDOW, "stance":
                       ("refute" if doc.kind == "contra" else "support" if g >= 2 else "na"),
                       "regime_described": doc.regime_described,
                       "misleading": (g == 0 and doc.regime_described not in (None, reg)),
                       "accidentally_correct": (g == 0 and doc.regime_described == reg),
                       "doc_kind": doc.kind, "provenance": doc.provenance})

    # shared routine distractors (属 corpus, unjudged for all queries -> grade 0 by pooling convention recorded)
    shared_ids = []
    for t in _SHARED_ROUTINE:
        d = new_doc("Corporate bulletin (shared)", t, "shared", None, None, None, "routine-shared")
        shared_ids.append(d.doc_id)

    return {"docs": docs, "queries": queries, "qrels": qrels, "evidence": ev,
            "shared_ids": shared_ids}


def sha256_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def write_beir(build, outdir):
    out = Path(outdir)
    (out / "qrels").mkdir(parents=True, exist_ok=True)
    (out / "splits").mkdir(parents=True, exist_ok=True)
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    docs, queries, qrels = build["docs"], build["queries"], build["qrels"]

    with open(out / "corpus.jsonl", "w") as f:
        for d in docs.values():
            f.write(json.dumps({"_id": d.doc_id, "title": d.title, "text": d.text,
                                "metadata": {"kind": d.kind, "provenance": d.provenance}}) + "\n")
    with open(out / "queries.jsonl", "w") as f:
        for q in queries:
            f.write(json.dumps({"_id": q.query_id, "text": q.text,
                                "metadata": {"true_regime": q.true_regime,
                                             "entity_node": q.entity_node, "split": q.split}}) + "\n")
    for split in ("dev", "test"):
        with open(out / "qrels" / f"{split}.tsv", "w") as f:
            f.write("query-id\tcorpus-id\tscore\n")
            for q in queries:
                if q.split != split:
                    continue
                for dd, g in sorted(qrels[q.query_id].items()):
                    f.write(f"{q.query_id}\t{dd}\t{g}\n")
    with open(out / "splits" / "splits.json", "w") as f:
        json.dump({"dev": [q.query_id for q in queries if q.split == "dev"],
                   "test": [q.query_id for q in queries if q.split == "test"],
                   "held_out_entities": [HELD_OUT_NODE],
                   "quarantine": "prototype IDs MUST NOT appear in V3 full benchmark"}, f, indent=2)
    try:
        import pandas as pd
        pd.DataFrame(build["evidence"]).to_parquet(out / "evidence_labels.parquet", index=False)
    except Exception:
        with open(out / "evidence_labels.json", "w") as f:
            json.dump(build["evidence"], f)
    manifest = {"seed": RULES["seed"], "files": {}}
    for p in [out / "corpus.jsonl", out / "queries.jsonl", out / "qrels" / "dev.tsv",
              out / "qrels" / "test.tsv", out / "splits" / "splits.json"]:
        manifest["files"][str(p.relative_to(out))] = sha256_text(p.read_text())
    (out / "manifests" / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
