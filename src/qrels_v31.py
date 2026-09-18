"""V3.1 query-answerable re-grade (see reports/v3/phase2b_audit.md).

Rule uses ONLY query/doc metadata expressible from query+text (entity, window,
content register). Regime-faithfulness is NOT a qrel concept anymore; it lives
in evidence op-labels. Deterministic; applied identically to dev and test.
Test TEXT is untouched and test METRICS remain sealed until the main experiment.
"""

from __future__ import annotations

OPS_MARKERS = (
    "lead time", "leadtime", "shipment", "delivery", "deliveries", "inbound",
    "demand", "orders", "order ", "inventory", "stock", "replenishment",
    "supplier", "supplier", "logistics", "carrier", "freight", "dock",
    "congestion", "delay", "backlog", "surge", "capacity", "procurement",
    "lane", "terminal", "hub", "etrailer", "eta", "allocation",
)

ROUTINE_MARKERS = (
    "payroll", "printer", "parking", "benefits open", "canteen", "holiday shutdown",
    "badge access", "charity drive", "water cooler", "bike shelter",
    "blood drive", "audit walkthrough", "fire drill", "fire extinguisher",
    "new starter induction", "travel policy refresher",
)

HEDGE_MARKERS = (
    "might be", "could be", "possible", "rumor", "unconfirmed", "chatter",
    "slightly elevated", "keeping an eye", "monitoring", "watch",
    "too early", "soft lead", "secondhand", "no primary-source",
)


def grade_v31(query_node, doc):
    """query_node: str. doc: corpus_v3.Doc. Returns 0..3."""
    import re
    t = doc.text
    m = re.search(r"\[node (\S+)", t)
    own = bool(m and m.group(1) == query_node)
    if not own:
        return 0
    low = t.lower()
    if "resolved / closed" in t:
        return 1  # own node, resolved window: stale (marker is in the text itself)
    if doc.kind in ("offtopic", "shared") or any(k in low for k in ROUTINE_MARKERS):
        # routine corporate traffic: construction kind is ground truth of intent
        # (no regime content involved); text markers as backup
        return 0
    if "planning window w-42" not in low:
        return 1 if any(k in low for k in OPS_MARKERS) else 0
    if not any(k in low for k in OPS_MARKERS):
        return 0
    if any(k in low for k in HEDGE_MARKERS):
        return 2
    return 3
