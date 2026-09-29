"""Independent v5 lockbox provenance validator.

This file intentionally does not import the lockbox generator or any v3/v4 qrel
implementation. It reconstructs grades from frozen query/document metadata.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
LOCKBOX = ROOT / "versions/sem2act-v5/runtime/lockbox"
GATE = {"weighted_kappa": 0.60, "exact": 0.60, "within_one": 0.90}


def weighted_kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    if n == 0 or n != len(b):
        raise ValueError("label vectors must be equally non-empty")
    cats = range(4)
    observed = [[0.0 for _ in cats] for _ in cats]
    for x, y in zip(a, b):
        observed[x][y] += 1.0 / n
    pa = [sum(observed[i]) for i in cats]
    pb = [sum(observed[i][j] for i in cats) for j in cats]
    weight = lambda i, j: 1.0 - ((i - j) / 3.0) ** 2
    observed_score = sum(observed[i][j] * weight(i, j)
                         for i in cats for j in cats)
    expected_score = sum(pa[i] * pb[j] * weight(i, j)
                         for i in cats for j in cats)
    if math.isclose(expected_score, 1.0):
        return 1.0 if math.isclose(observed_score, 1.0) else 0.0
    return (observed_score - expected_score) / (1.0 - expected_score)


def reconstruct_grade(query: dict, doc: dict) -> int:
    qmeta = query["metadata"]
    dmeta = doc["metadata"]
    if dmeta.get("kind") in {"wrong_entity", "offtopic", "contradictory", "shared"}:
        return 0
    if dmeta.get("entity_node") != qmeta["entity_node"]:
        return 0
    if dmeta.get("window") != qmeta["window"]:
        return 1 if dmeta.get("kind") == "stale" else 0
    if dmeta.get("kind") == "grade3":
        return 3
    if dmeta.get("kind") == "grade2":
        return 2
    if dmeta.get("kind") == "stale":
        return 1
    return 0


def main() -> int:
    queries = {
        x["_id"]: x
        for x in map(json.loads, (LOCKBOX / "queries.jsonl").read_text().splitlines())
    }
    docs = {
        x["_id"]: x
        for x in map(json.loads, (LOCKBOX / "corpus.jsonl").read_text().splitlines())
    }
    rows = []
    with (LOCKBOX / "private/qrels.tsv").open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows.append((row["query-id"], row["corpus-id"], int(row["score"])))
    if len({q for q, _, _ in rows}) != 240:
        raise SystemExit("provenance gate requires all 240 queries")

    expected, actual, output = [], [], []
    for qid, did, grade in rows:
        if qid not in queries or did not in docs:
            raise SystemExit(f"unknown qrel key {qid}/{did}")
        oracle = reconstruct_grade(queries[qid], docs[did])
        expected.append(oracle)
        actual.append(grade)
        output.append({
            "query_id": qid,
            "document_id": did,
            "constructed_grade": grade,
            "provenance_grade": oracle,
        })

    exact = sum(a == b for a, b in zip(actual, expected)) / len(actual)
    within = sum(abs(a - b) <= 1 for a, b in zip(actual, expected)) / len(actual)
    kappa = weighted_kappa(actual, expected)
    confusion = [[0] * 4 for _ in range(4)]
    for a, b in zip(actual, expected):
        confusion[a][b] += 1
    passed = (
        kappa >= GATE["weighted_kappa"]
        and exact >= GATE["exact"]
        and within >= GATE["within_one"]
    )
    result = {
        "schema_version": 1,
        "experiment_id": "v5-lockbox-provenance",
        "n_pairs": len(rows),
        "metrics": {
            "weighted_kappa": kappa,
            "exact": exact,
            "within_one": within,
        },
        "confusion_matrix_constructed_rows_provenance_columns": confusion,
        "gate": GATE,
        "passed": passed,
        "qrels_unchanged": True,
        "source": str(Path(__file__).relative_to(ROOT)),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "labels_sha256": hashlib.sha256(
            "\n".join(json.dumps(x, sort_keys=True) for x in output).encode()
        ).hexdigest(),
        "failure_policy": "failed gate invalidates lockbox; qrels are never repaired",
    }
    (LOCKBOX / "provenance_labels.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in output)
    )
    (ROOT / "versions/sem2act-v5/manifests").mkdir(parents=True, exist_ok=True)
    (ROOT / "versions/sem2act-v5/manifests/provenance_validation.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
