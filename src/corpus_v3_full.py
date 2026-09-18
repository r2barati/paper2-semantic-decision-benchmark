"""V3 full-benchmark content wave 2 + build_full_corpus (200 queries / ~4k docs).

FROZEN RULES: configs/v3/dataset_full.yaml (read before generating anything).
Phase-1 code above is untouched; the prototype (v3p-*) is quarantined by test.

Uniqueness strategy: bank texts carry seeded SURFACE variation (dock/shift/day
reporting details) that changes no grade semantics, so shared operational reality
across sites still dedups only when truly identical. 100% freshly authored: zero
V1/V2 bank text is imported anywhere in this file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from src.corpus_v3 import (CURRENT_WINDOW, PAST_WINDOW, REGIMES, Doc, Query,
                           _CONTRADICT, _DELAY_CLEAR, _DELAY_HEDGED,
                           _NORMAL_CLEAR, _OFFTOPIC_LEXICAL, _OFFTOPIC_SEMANTIC,
                           _SHARED_ROUTINE, _SURGE_CLEAR, _SURGE_HEDGED,
                           _header, check_giveaways, sha256_text)

ROOT = Path(__file__).resolve().parent.parent
FULL_RULES = yaml.safe_load((ROOT / "configs" / "v3" / "dataset_full.yaml").read_text())

_STEMS = ["ALDER", "BRIAR", "CEDAR", "DUN", "ELM", "FAIR", "GABLE", "HARBOR",
           "IRON", "JUNIPE", "KEN", "LAKE", "MAPLE", "NORTH", "OVER", "PINE",
           "QUARRY", "RIVER", "STONE", "TIMBER", "UNDER", "VALLEY", "WEST",
           "YARROW", "ASH", "BROOK", "CLEAR", "DEER", "EAST", "FOX",
           "GREEN", "HILL", "COLD", "RED", "BLUE", "GRANITE"]
_TAILS = ["GROVE-DC", "WOOD-DC", "FALLS-DC", "MORE-DC", "HURST-DC", "BANK-DC",
          "VIEW-DC", "LIGHT-DC", "SHORE-DC", "GATE-DC", "LOOK-DC", "BEND-DC",
          "FIELD-DC", "CREEK-DC", "HILL-DC", "FORD-DC", "DALE-DC", "MONT-DC"]
_SUPS = ["WESTMARK-FREIGHT", "COASTLINE-CARRIERS", "NORTHBRIDGE-LOGISTICS",
         "STONEGATE-HAULAGE", "TRAILHEAD-TRANSPORT", "QUARRYSIDE-SHIPPING",
         "IRONHORSE-LOGISTICS", "BLUEFIN-FREIGHT", "COPPERLINE-CARRIERS",
         "SAGEBRUSH-FREIGHT", "RIVERSTONE-LOGISTICS", "FOGBANK-HAULAGE",
         "TIMBERLINE-TRANSPORT", "HIGHLAND-CARRIERS", "MEADOWLARK-FREIGHT"]


def _gen_entities(rng, n, prefix):
    used = set()
    out = []
    i = 0
    while len(out) < n:
        site = f"{rng.choice(_STEMS)}{rng.choice(_TAILS)}"
        if site in used:
            continue
        used.add(site)
        out.append({"node": f"{prefix}{i + 1}",
                    "site": site,
                    "supplier": str(rng.choice(_SUPS)),
                    "line": str(int(rng.integers(1000, 9999)))})
        i += 1
    return out

# --- wave-2 regime variants (same discipline: partial params only) ---
_DELAY_CLEAR_2 = [
    ("email", "Subject: {supplier} delays. Morning call confirmed extended dwell on our lanes. No revised schedule yet — will cascade ETAs as they land."),
    ("supplier status report", "Yard report: {supplier} trailer pool depleted; empty returns delayed. Loading appointments pushed to next cycle."),
    ("logistics notice", "Congestion surcharge notice from {supplier} suggests sustained pressure. Operational impact under review."),
    ("internal note", "Ops note: inbound appointments from {supplier} bunching up. Receiving overtime likely this week."),
    ("shipment notice", "Advance notice: {supplier} embargo on new bookings until backlog clears. Existing tenders stand."),
    ("procurement update", "Procurement update: secondary carrier engaged for overflow while {supplier} recovers. Premium rates apply."),
    ("industry report", "Analyst note: regional hub congestion hitting multiple shippers on {supplier} lanes. Recovery pace unclear."),
    ("email", "Subject: dock backlog. {supplier} reporting full yards at origin terminal. Our loads are in the queue."),
    ("supplier status report", "Service alert: {supplier} toggled to allocation mode. Volumes above baseline deferred."),
    ("logistics notice", "Detention risk rising on {supplier} equipment held at congested terminals. Clock starts at free-time expiry."),
    ("internal note", "Shift log: two {supplier} arrivals missed their windows overnight. Day shift to resequence receiving."),
    ("email", "Subject: heads-up. Freight forwarder warns {supplier} linehaul capacity is committed elsewhere this week."),
]
_DELAY_HEDGED_2 = [
    ("email", "Subject: watching. One late {supplier} arrival — probably weather. Flagging so it is on the radar."),
    ("internal note", "Logbook: {supplier} tracking timestamps look stale. Could be a feed issue rather than physical delay."),
    ("supplier status report", "Advisory: {supplier} mentions planned maintenance next month. No current impact stated."),
    ("logistics notice", "Weather desk: storms near {supplier} hub may slow linehaul. Contingency on standby, nothing moving yet."),
    ("email", "Subject: secondhand news. Customer claims {supplier} is struggling. No primary-source confirmation."),
]
_SURGE_CLEAR_2 = [
    ("email", "Subject: pipeline filling fast. Quote-to-order conversion spiking; planners should assume elevated pull."),
    ("market report", "Distributor survey: restocking orders up across the board. Our segment shows the steepest climb."),
    ("internal note", "Ops note: wave releases doubling in size. Labor plan adjusted for sustained volume."),
    ("procurement update", "Procurement update: raw-material call-offs accelerated to match finished-goods demand."),
    ("industry report", "Analyst upgrade: end-market demand revised upward. Channel inventory thin, reorder cycle active."),
    ("email", "Subject: promo hit. The promotion outperformed forecast threefold. Replenishment chase is on."),
    ("market report", "POS data: unit velocity climbing four weeks straight. No sign of plateau in latest read."),
    ("internal note", "Capacity note: overtime and weekend shifts scheduled to cover the order backlog."),
    ("email", "Subject: new listing. National account ranging our line — initial fill plus pipeline orders incoming."),
    ("procurement update", "Forecast update: statistical models lifted after persistent positive bias in order signals."),
    ("market report", "Seasonal pull-forward detected: buyers ordering ahead of announced price move."),
    ("internal note", "Backlog review: oldest open orders aging but volumes growing — demand-led, not fulfillment failure."),
]
_SURGE_HEDGED_2 = [
    ("email", "Subject: curious. Webstore conversion up two days running. Seasonal or real? Tracking."),
    ("internal note", "Huddle note: sales pipeline coverage improving. Slippage still possible on large deals."),
    ("market report", "Foot-traffic counters show modest gains. Translation to orders unconfirmed."),
    ("email", "Subject: sampling uptick. Field team reports more sample requests. Historically a soft lead indicator."),
    ("industry report", "Freight futures firming — sometimes a demand proxy, sometimes noise. Logged without action."),
]
_NORMAL_CLEAR_2 = [
    ("email", "Subject: green board. All {supplier} lanes green, order flow nominal. Enjoy the quiet week."),
    ("supplier status report", "QBR snippet: {supplier} service score steady quarter over quarter. No corrective actions open."),
    ("logistics notice", "Network notice: no weather, labor, or capacity advisories on {site} lanes this window."),
    ("internal note", "EOD summary: receipts matched plan, picks on pace, no exceptions carried overnight."),
    ("shipment notice", "Tender accepted: {supplier} confirmed capacity for the full weekly plan."),
    ("procurement update", "Procurement update: contract coverage adequate; no spot activity required."),
]
_CONTRADICT_2 = {
    "supplier_delay": [
        ("internal note", "Ops note: dock timestamps show {supplier} arrivals within tolerance all week. Delay narrative not supported by data."),
        ("email", "Subject: verified normal. Carrier scorecard for {supplier} shows on-time performance at target. Earlier alert retracted."),
    ],
    "demand_surge": [
        ("email", "Subject: pipeline review. Alleged mega-orders traced to duplicate CRM entries. True demand flat."),
        ("internal note", "Finance check: revenue run-rate unchanged. The 'surge' is a reporting artifact of the cutoff change."),
    ],
    "normal": [
        ("supplier status report", "Flash: minor {supplier} system glitch caused tracking gaps; physical flow unaffected. Remains a normal window."),
    ],
}
_OFFTOPIC_LEXICAL_2 = [
    "Mailroom: parcel supplier changed pickup times. Internal mail unaffected otherwise.",
    "Vending supplier delayed restocking the night shift line. Facilities notified.",
    "Courier contract renewal: same-day delivery SLA under renegotiation. Standard lanes unaffected.",
    "Office move: the logistics of relocating the planning team scheduled next quarter. No ops impact.",
    "PPE supplier backordered earplugs. Alternative brand approved by safety.",
    "Coffee supplier missed its slot; backup vendor covering. Break rooms restocked.",
    "Document shredding supplier pickup rescheduled to Thursdays. Compliance unaffected.",
    "Uniform supplier size exchange window extended. New hires unaffected.",
    "Water delivery supplier route changed. Coolers remain filled.",
    "Landscaping supplier delayed hedge trimming. Site operations continue.",
    "Badge printer toner on backorder. Temporary passes available at reception.",
    "Forklift tire supplier lead times extended. Maintenance scheduled around stock.",
    "Pallet supplier allocation tight this month. Pooling partner covering gaps.",
    "Cleaning supplier changed chemical dilution ratios. Safety data sheets updated.",
    "Stationery supplier catalog refresh. Order codes unchanged.",
    "Furniture supplier delivery for the new annex delayed. Existing space unaffected.",
    "HVAC filter supplier shipment late. No temperature excursions recorded.",
    "Lighting supplier retrofit kits arriving in phases. Warehouse lux levels nominal.",
    "Signage supplier proof round delayed. Wayfinding project timeline holds.",
    " telecoms supplier maintenance window Sunday. Warehouse systems on backup links.",
    "Elevator service supplier inspection passed. No downtime scheduled.",
    "Waste hauler supplier missed Friday pickup. Extra bins ordered.",
    "Pest control supplier visit completed. No findings in storage zones.",
    "Window cleaning supplier rained out. Rescheduled without disruption.",
]
_OFFTOPIC_SEMANTIC_2 = [
    "Energy audit: warehouse lighting load profiled. Retrofit business case in draft.",
    "Racking inspection passed with minor observations. Remediation booked.",
    "Cycle-count accuracy at 98.4%. Slotting review for slow movers proposed.",
    "Labor planning: agency coverage for peak season under negotiation.",
    "YMS upgrade user-acceptance testing begins Monday. Super-users assigned.",
    "Dock door maintenance rotation published. One door at a time, no capacity loss.",
    "Inventory valuation method review by auditors. No restatement expected.",
    "Carrier scorecard QBR deck due. Data pull from TMS in progress.",
    "Customs broker license renewal filed. Cross-border lanes unaffected.",
    "Claims log: two concealed-damage cases pending carrier response.",
    "Returns processing backlog cleared. Refurb queue within SLA.",
    "Slotting optimization pilot in aisle 12. Pick-path times being measured.",
    "Shrinkage review: within tolerance for the quarter. No investigation opened.",
    "Temperature mapping study for the ambient zone. Protocol drafted.",
    "Forklift fleet telematics rollout phase two. Fuel-use baselines forming.",
    "Packaging spec change for fragile SKUs. Trials scheduled off-peak.",
    "Appointment scheduling compliance at 91%. Late-arrival fees disputed.",
    "Demurrage invoice review: two charges under appeal with evidence attached.",
    "Network design study kickoff: lane cost benchmarking, no decisions this year.",
    "Business continuity tabletop exercise next month. Scenario: IT outage.",
    "KPI dashboard redesign workshop. Metric definitions under review.",
    "Peak-season playbook refresh: lessons-learned session minutes attached.",
    "Supplier diversity reporting: spend categorization updated.",
    "Carbon accounting: diesel-to-electric forklift transition model drafted.",
]
_ROUTINE_TOPICS = [
    "Fire drill {day}. Muster points unchanged.",
    "Payroll calendar published for the quarter. {extra}",
    "Benefits open enrolment closes {day}. Complete elections in the portal.",
    "Parking structure {activity} {day}. Overflow lot available.",
    "Canteen menu rotation. {extra}",
    "Holiday shutdown calendar circulated. {extra}",
    "Badge access audit {day}. Report lost cards promptly.",
    "Charity drive collection {day}. {extra}",
    "Water cooler maintenance {day}. Brief outage expected.",
    "Bike shelter cleaning {day}. Move locks beforehand.",
    "Printer fleet firmware {day}. Short spooler restart expected.",
    "Travel policy refresher {day}. {extra}",
    "Safety certification window open. {extra}",
    "Expense report deadline {day}. {extra}",
    "New starter inductions {day}. Book rooms via facilities.",
    "Fire extinguisher inspections {day}. {extra}",
    "Elevator maintenance {day}. Use stairs in block B.",
    "Landscaping works {day}. Deliveries use north gate.",
    "Audit walkthrough {day}. Tidy shared areas beforehand.",
    "Blood drive {day} in the conference suite. Sign-up sheet attached.",
]
_ROUTINE_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
_ROUTINE_EXTRA = ["No action required.", "Details on the intranet.", "Contact facilities with questions.",
                  "Applies to all shifts.", "See attached schedule.", "No change to operations."]
_ROUTINE_ACT = ["resurfacing continues", "cleaning scheduled", "line marking planned"]

# Surface variation: neutral reporting details (dock/shift/day/load). Grade-neutral by construction.
_DETAILS = ["Dock {dock}, {shift} shift, reported {day}.",
            "Load {load}, gate {gate}, logged {day}.",
            "Update {n}: {shift} shift notes, {day}."]
_DOCKS = ["3", "5", "7", "9", "12"]
_SHIFTS = ["morning", "evening", "night"]
_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday"]
_LOADS = ["L-1042", "L-2210", "L-3387", "L-4519", "L-5630"]
_GATES = ["A", "B", "C"]


def _vary(text, rng):
    d = rng.choice(_DETAILS)
    return (text + " " + d.format(dock=rng.choice(_DOCKS), shift=rng.choice(_SHIFTS),
                                  day=rng.choice(_DAYS), load=rng.choice(_LOADS),
                                  gate=rng.choice(_GATES),
                                  n=int(rng.integers(2, 6)))).strip()


def _bank_all():
    return {
        "delay_clear": _DELAY_CLEAR + _DELAY_CLEAR_2,
        "delay_hedged": _DELAY_HEDGED + _DELAY_HEDGED_2,
        "surge_clear": _SURGE_CLEAR + _SURGE_CLEAR_2,
        "surge_hedged": _SURGE_HEDGED + _SURGE_HEDGED_2,
        "normal_clear": _NORMAL_CLEAR + _NORMAL_CLEAR_2,
    }


def _test_only(idx):
    return idx % 5 == 0  # ~20% of bank items reserved test-only


def build_full_corpus(seed=20260914):
    """200 queries / ~4k pairs over 64 nodes (~3 queries/node). See dataset_full.yaml."""
    rng = np.random.default_rng(seed)
    std_ents = _gen_entities(np.random.default_rng(seed + 1), 48, "S")
    held_ents = _gen_entities(np.random.default_rng(seed + 2), 16, "H")
    held_nodes = {e["node"] for e in held_ents}
    ENTITIES = std_ents + held_ents
    banks = _bank_all()
    contra = {k: _CONTRADICT[k] + _CONTRADICT_2[k] for k in _CONTRADICT}
    lex = _OFFTOPIC_LEXICAL + _OFFTOPIC_LEXICAL_2
    sem = _OFFTOPIC_SEMANTIC + _OFFTOPIC_SEMANTIC_2

    docs, queries, qrels, ev = {}, [], {}, []
    did = [0]
    seen = {}

    def new_doc(title, text, kind, regime, ent, window, source, contradicts=None):
        h = sha256_text(text)
        if h in seen:
            return docs[seen[h]]
        for pat in check_giveaways(text):
            raise AssertionError(f"giveaway {pat}")
        did[0] += 1
        d = Doc(f"v3-d{did[0]:05d}", title, text, kind, regime, ent, window,
                source, contradicts)
        docs[d.doc_id] = d
        seen[h] = d.doc_id
        return d

    regimes = ["supplier_delay"] * 80 + ["demand_surge"] * 80 + ["normal"] * 40
    order = rng.permutation(200)
    std_idx = [i for i in range(200) if i % 4 != 3]   # 150 standard slots
    held_idx = [i for i in range(200) if i % 4 == 3]  # 50 held-out slots
    std_cycle = [std_ents[i % len(std_ents)] for i in range(len(std_idx))]
    held_cycle = [held_ents[i % len(held_ents)] for i in range(len(held_idx))]
    ent_for = {}
    for s, e in zip(sorted(std_idx), std_cycle):
        ent_for[s] = e
    for s, e in zip(sorted(held_idx), held_cycle):
        ent_for[s] = e
    HELD = held_nodes
    dev_pool = [i for i in std_idx]
    dev_idx = set(rng.choice(dev_pool, size=40, replace=False).tolist())
    # stratify dev 16/16/8 by swapping
    need = {"supplier_delay": 16, "demand_surge": 16, "normal": 8}
    have = {r: sum(1 for i in dev_idx if regimes[int(order[i])] == r) for r in need}
    for r in need:
        while have[r] < need[r]:
            out = next(i for i in dev_idx if regimes[int(order[i])] != r
                       and have[regimes[int(order[i])]] > need[regimes[int(order[i])]])
            inn = next(i for i in dev_pool if i not in dev_idx and regimes[int(order[i])] == r)
            dev_idx.discard(out)
            dev_idx.add(inn)
            have = {x: sum(1 for i in dev_idx if regimes[int(order[i])] == x) for x in need}

    sev = rng.choice(["major", "moderate"], size=200)
    tim = rng.choice(["early", "standard", "late"], size=200)

    for qi in range(200):
        reg = regimes[int(order[qi])]
        ent = ent_for[qi]
        qid = f"v3-q{qi + 1:03d}"
        split = "dev" if qi in dev_idx else "test"
        is_dev = qi in dev_idx
        qtext = (f"Current inbound supply, lead time and demand conditions for node {ent['node']} "
                 f"site {ent['site']} supplier {ent['supplier']} in {CURRENT_WINDOW} "
                 f"affecting replenishment now.")
        queries.append(Query(qid, qtext, reg, ent["node"], split))
        pool = {}

        def scoped(t, e, w=CURRENT_WINDOW):
            return f"{_header(e, w)} {t}"

        def pick(bank, n):
            idx = [i for i in range(len(bank)) if not (is_dev and _test_only(i))]
            sel = rng.choice(idx, size=min(n, len(idx)), replace=False)
            return [(int(i), bank[int(i)]) for i in np.atleast_1d(sel)]

        clear = banks[{"supplier_delay": "delay_clear", "demand_surge": "surge_clear",
                       "normal": "normal_clear"}[reg]]
        hedged = banks[{"supplier_delay": "delay_hedged", "demand_surge": "surge_hedged",
                        "normal": "normal_clear"}[reg]]
        for _, (style, t) in pick(clear, 2):
            d = new_doc(f"{style} ({ent['node']})",
                        scoped(_vary(t.format(**ent), rng), ent),
                        "grade3", reg, ent["node"], CURRENT_WINDOW, style)
            pool[d.doc_id] = 3
        for _, (style, t) in pick(hedged, 2):
            d = new_doc(f"{style} ({ent['node']})",
                        scoped(_vary(t.format(**ent), rng), ent),
                        "grade2", reg, ent["node"], CURRENT_WINDOW, style)
            pool[d.doc_id] = 2
        for _, (style, t) in pick(clear, 2):
            body = f"RESOLVED / CLOSED ({PAST_WINDOW}): {_vary(t.format(**ent), rng)}"
            d = new_doc(f"{style} [stale] ({ent['node']})",
                        f"{_header(ent, PAST_WINDOW)} {body}",
                        "stale", reg, ent["node"], PAST_WINDOW, style)
            pool[d.doc_id] = 1
        if reg != "normal" and qi % 2 == 0:
            cb = contra[reg]
            avail = [(i, cb[i]) for i in range(len(cb)) if not (is_dev and _test_only(i))]
            style, t = avail[int(rng.integers(len(avail)))][1]
            target = [k for k, v in pool.items() if v == 3][0]
            d = new_doc(f"{style} [correction] ({ent['node']})",
                        scoped(_vary(t.format(**ent), rng), ent),
                        "contra", reg, ent["node"], CURRENT_WINDOW, style,
                        contradicts=target)
            pool[d.doc_id] = 0
        others = [e for e in ENTITIES if e["node"] != ent["node"]]
        err_regimes = [r for r in REGIMES if r != reg][:1] + [reg] + ["normal",
                       [r for r in REGIMES if r != reg][0], reg]
        for j, er in enumerate(err_regimes[:5]):
            oe = others[int(rng.integers(len(others)))]
            eb = {"supplier_delay": banks["delay_clear"], "demand_surge": banks["surge_clear"],
                  "normal": banks["normal_clear"]}[er]
            avail = [(i, eb[i]) for i in range(len(eb)) if not (is_dev and _test_only(i))]
            style, t = avail[int(rng.integers(len(avail)))][1]
            d = new_doc(f"{style} ({oe['node']})", scoped(_vary(t.format(**oe), rng), oe),
                        "wrong_entity", er, oe["node"], CURRENT_WINDOW, style)
            pool[d.doc_id] = 0
        for t in [lex[i] for i in rng.choice(len(lex), size=3, replace=False)]:
            d = new_doc("Operations memo", scoped(t, ent), "offtopic", None,
                        ent["node"], CURRENT_WINDOW, "offtopic-lexical")
            pool[d.doc_id] = 0
        for t in [sem[i] for i in rng.choice(len(sem), size=3, replace=False)]:
            d = new_doc("Operations memo", scoped(t, ent), "offtopic", None,
                        ent["node"], CURRENT_WINDOW, "offtopic-semantic")
            pool[d.doc_id] = 0
        for t in [_ROUTINE_TOPICS[i] for i in rng.choice(len(_ROUTINE_TOPICS), size=2, replace=False)]:
            body = t.format(day=rng.choice(_ROUTINE_DAYS),
                            extra=rng.choice(_ROUTINE_EXTRA),
                            activity=rng.choice(_ROUTINE_ACT))
            d = new_doc("Corporate bulletin", scoped(body, ent), "offtopic", None,
                        ent["node"], CURRENT_WINDOW, "routine")
            pool[d.doc_id] = 0

        qrels[qid] = pool
        for dd, g in pool.items():
            doc = docs[dd]
            ev.append({"query_id": qid, "doc_id": dd, "qrel": g, "true_regime": reg,
                       "severity": str(sev[qi]), "timing": str(tim[qi]),
                       "entity_match": doc.entity_node == ent["node"],
                       "fresh": doc.window == CURRENT_WINDOW,
                       "stance": ("refute" if doc.kind == "contra"
                                  else "support" if g >= 2 else "na"),
                       "regime_described": doc.regime_described,
                       "misleading": (g == 0 and doc.regime_described not in (None, reg)),
                       "accidentally_correct": (g == 0 and doc.regime_described == reg),
                       "doc_kind": doc.kind})

    for i in range(120):
        t = _ROUTINE_TOPICS[int(rng.integers(len(_ROUTINE_TOPICS)))]
        body = t.format(day=rng.choice(_ROUTINE_DAYS), extra=rng.choice(_ROUTINE_EXTRA),
                        activity=rng.choice(_ROUTINE_ACT))
        new_doc("Corporate bulletin (shared)", body, "shared", None, None, None,
                "routine-shared")
    return {"docs": docs, "queries": queries, "qrels": qrels, "evidence": ev,
            "held_nodes": sorted(held_nodes)}


def write_full(build, outdir):
    out = Path(outdir)
    (out / "qrels").mkdir(parents=True, exist_ok=True)
    (out / "splits").mkdir(parents=True, exist_ok=True)
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    docs, queries, qrels = build["docs"], build["queries"], build["qrels"]
    with open(out / "corpus.jsonl", "w") as f:
        for d in docs.values():
            f.write(json.dumps({"_id": d.doc_id, "title": d.title, "text": d.text,
                                "metadata": {"kind": d.kind, "provenance": "fresh"}}) + "\n")
    with open(out / "queries.jsonl", "w") as f:
        for q in queries:
            f.write(json.dumps({"_id": q.query_id, "text": q.text,
                                "metadata": {"true_regime": q.true_regime,
                                             "entity_node": q.entity_node,
                                             "split": q.split}}) + "\n")
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
                   "held_out_entities": build.get("held_nodes", []),
                   "test_only_bank_rule": "bank index mod 5 == 0"}, f, indent=2)
    try:
        import pandas as pd
        pd.DataFrame(build["evidence"]).to_parquet(out / "evidence_labels.parquet", index=False)
    except Exception:
        with open(out / "evidence_labels.json", "w") as f:
            json.dump(build["evidence"], f)
    manifest = {"seed": FULL_RULES["seed"], "files": {}}
    for p in [out / "corpus.jsonl", out / "queries.jsonl", out / "qrels" / "dev.tsv",
              out / "qrels" / "test.tsv", out / "splits" / "splits.json"]:
        manifest["files"][str(p.relative_to(out))] = sha256_text(p.read_text())
    (out / "manifests" / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
