"""V3 full-benchmark tests (200q/~4k). Frozen rules: configs/v3/dataset_full.yaml."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.corpus_v3_full import build_full_corpus
from tests.test_v3proto import _text_only_grade, _cohen_kappa


def test_full_scale_and_splits():
    b = build_full_corpus()
    assert len(b["queries"]) == 200
    assert len(b["docs"]) >= 2500, f"only {len(b['docs'])} unique docs"
    dev = [q for q in b["queries"] if q.split == "dev"]
    test = [q for q in b["queries"] if q.split == "test"]
    assert len(dev) == 40 and len(test) == 160
    assert Counter(q.true_regime for q in dev) == {"supplier_delay": 16, "demand_surge": 16, "normal": 8}
    held_nodes = set(b["held_nodes"])
    assert len(held_nodes) == 16
    assert not any(q.entity_node in held_nodes for q in dev)
    held = [q for q in b["queries"] if q.entity_node in held_nodes]
    assert len(held) == 50 and all(q.split == "test" for q in held)
    # competition ratio guard (validity): same-entity current docs per node must stay
    # well below the level that made within-entity ranking near-chance (~328 in v1 build)
    from collections import defaultdict as _dd
    per_ent = _dd(int)
    for d in b["docs"].values():
        if d.window == "planning window W-42" and d.entity_node:
            per_ent[d.entity_node] += 1
    worst = max(per_ent.values())
    print(f"\nmax same-entity current docs per node: {worst}")
    # Must stay an order below the ~328 level that made within-entity ranking
    # near-chance; ~70 expected at ~3 queries/node.
    assert worst <= 110, f"competition ratio too high: {worst}"


def test_full_annotation_agreement():
    b = build_full_corpus()
    qent = {q.query_id: q.entity_node for q in b["queries"]}
    gold, second = [], []
    for qid, pool in b["qrels"].items():
        for dd, g in pool.items():
            gold.append(g)
            second.append(_text_only_grade(b["docs"][dd], qent[qid]))
    kappa = _cohen_kappa(gold, second)
    print(f"\nfull annotation kappa: {kappa:.3f} (n={len(gold)})")
    assert kappa >= 0.6


def test_full_quarantine_and_provenance():
    b = build_full_corpus()
    assert not any(d.doc_id.startswith("v3p-") for d in b["docs"].values())
    assert not any(q.query_id.startswith("v3p-") for q in b["queries"])
    assert all(d.provenance == "fresh" for d in b["docs"].values())
    from src.events import REGIME_WARNING_TEMPLATES
    from src.confirmation_templates import CONFIRMATION_TEMPLATES
    bank = {t["text"] for t in REGIME_WARNING_TEMPLATES} | {t["text"] for t in CONFIRMATION_TEMPLATES}
    for d in b["docs"].values():
        for t in bank:
            assert t not in d.text, f"verbatim bank text in {d.doc_id}"


def test_full_no_exact_duplicates():
    from src.retrieval import tokenize
    b = build_full_corpus()
    seen = set()
    for d in b["docs"].values():
        key = tuple(sorted(set(tokenize(d.text))))
        assert key not in seen, f"exact duplicate text: {d.doc_id}"
        seen.add(key)


def test_full_grade_mix():
    b = build_full_corpus()
    grades = Counter(g for v in b["qrels"].values() for g in v.values())
    print(f"\nfull grades: {dict(grades)}")
    assert grades[3] >= 350 and grades[2] >= 350 and grades[1] >= 350 and grades[0] >= 2000
