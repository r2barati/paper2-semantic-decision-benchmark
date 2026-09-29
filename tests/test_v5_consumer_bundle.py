"""Synthetic-only determinism tests for the v5 consumer bundle assembler."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_v5_consumer_bundle import (
    FAMILIES,
    EXPECTED_QUERY_IDS,
    build_bundle,
)


class ConsumerBundleDeterminismTest(unittest.TestCase):
    def setUp(self) -> None:
        self.queries = [{
            "_id": qid,
            "text": f"synthetic development query for {qid}",
            "metadata": {
                "entity_node": f"synthetic-node-{index:04d}",
                "true_regime": "normal",
            },
        } for index, qid in enumerate(EXPECTED_QUERY_IDS, 1)]
        self.corpus = {
            f"synthetic-doc-{index:04d}": {
                "_id": f"synthetic-doc-{index:04d}",
                "text": f"synthetic development evidence document {index}",
            }
            for index in range(1, 4)
        }
        self.rankings = {
            system: {
                qid: [f"synthetic-doc-{index:04d}" for index in (1, 2, 3)]
                for qid in EXPECTED_QUERY_IDS
            }
            for system in ("bm25", "rerank", "oracle")
        }
        self.model_spec = {
            "model_id": "Qwen/Qwen3-8B-AWQ",
            "revision": "4da05a8edb55c6046cce958586c33b61da07bb79",
            "tokenizer_revision": "4da05a8edb55c6046cce958586c33b61da07bb79",
            "runtime_lock_id": "synthetic-dev-lock",
            "runtime": {"backend": "synthetic-dev", "quantization": {"method": "awq"}},
        }

    def build(self, out: Path) -> dict:
        return build_bundle(
            family="qwen", output_dir=out, queries=self.queries,
            corpus=self.corpus, rankings=self.rankings,
            model_spec=self.model_spec,
            runtime_lock_sha256="1" * 64,
            retrieval_manifest_sha256="2" * 64,
            reranker_manifest_sha256="3" * 64,
            reranker_verification_sha256="4" * 64,
            reranker_output_sha256="5" * 64,
            input_hashes={"synthetic.jsonl": "6" * 64},
            execution_sha="synthetic-dev-execution",
        )

    def test_workload_dimensions_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sem2act-v5-bundle-test-") as temp:
            out = Path(temp) / "qwen"
            result = self.build(out)
            rows = [json.loads(line) for line in (out / "evidence_inputs.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 720)
            self.assertEqual({row["query_id"] for row in rows}, set(EXPECTED_QUERY_IDS))
            self.assertEqual({system: sum(row["system"] == system for row in rows)
                              for system in ("bm25", "rerank", "oracle")},
                             {"bm25": 240, "rerank": 240, "oracle": 240})
            manifest_path = Path(result["manifest_path"])
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(result["manifest_sha256"], hashlib.sha256(manifest_path.read_bytes()).hexdigest())
            self.assertEqual(manifest["expected_workload"], {
                "queries": 240, "evidence_rows": 720, "calls": 2880,
            })
            self.assertEqual(manifest["source_reranker_output_sha256"], "5" * 64)
            self.assertEqual(manifest["evaluation_labels_included"], True)
            self.assertEqual(manifest["model_prompt_label_exposure"], False)
            self.assertEqual(manifest["protocol_hash"], "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022")
            for name, item in manifest["files"].items():
                data = (out / name).read_bytes()
                self.assertEqual(item["sha256"], hashlib.sha256(data).hexdigest())
                self.assertEqual(item["bytes"], len(data))

    def test_repeated_builds_are_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sem2act-v5-bundle-test-") as temp:
            root = Path(temp)
            first, second = root / "first", root / "second"
            self.build(first)
            self.build(second)
            def snapshot(path: Path) -> dict[str, bytes]:
                return {p.relative_to(path).as_posix(): p.read_bytes()
                        for p in sorted(path.iterdir()) if p.is_file()}
            self.assertEqual(snapshot(first), snapshot(second))

    def test_later_kaggle_families_keep_their_pinned_call_budgets(self) -> None:
        for family in ("llama", "mistral"):
            with self.subTest(family=family), tempfile.TemporaryDirectory(
                    prefix=f"sem2act-v5-{family}-bundle-test-") as temp:
                source = FAMILIES[family]
                model_spec = {
                    "model_id": source["model_id"],
                    "revision": source["revision"],
                    "tokenizer_revision": source["revision"],
                    "runtime_lock_id": "synthetic-dev-lock",
                    "runtime": {"backend": "synthetic-dev", "quantization": {"method": "nf4"}},
                }
                out = Path(temp) / family
                result = build_bundle(
                    family=family, output_dir=out, queries=self.queries,
                    corpus=self.corpus, rankings=self.rankings,
                    model_spec=model_spec, runtime_lock_sha256="1" * 64,
                    retrieval_manifest_sha256="2" * 64,
                    reranker_manifest_sha256="3" * 64,
                    reranker_verification_sha256="4" * 64,
                    reranker_output_sha256="5" * 64,
                    input_hashes={"synthetic.jsonl": "6" * 64},
                    execution_sha="synthetic-dev-execution",
                )
                manifest_path = Path(result["manifest_path"])
                manifest = json.loads(manifest_path.read_text())
                self.assertEqual(manifest["expected_workload"], {
                    "queries": 240, "evidence_rows": 720, "calls": 1920,
                })
                self.assertEqual(len((out / "evidence_inputs.jsonl").read_text().splitlines()), 720)
                self.assertEqual(result["manifest_sha256"], hashlib.sha256(manifest_path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
