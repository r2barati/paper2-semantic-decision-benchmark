"""Static tests for Moon staging and subprocess coverage; no model/job is run."""

from __future__ import annotations

import json
import resource
import tempfile
import unittest
from pathlib import Path

from scripts import run_v5_cpu_rerank_shard as worker
from scripts import run_v5_moon_reranker_pilot as pilot
from scripts import stage_v5_moon_reranker_model as staging


class MoonExecutionTests(unittest.TestCase):
    def test_model_allowlist_is_the_unchanged_accepted_twelve_files(self) -> None:
        model_id, revision, accepted, allowlist = staging.authorized_snapshot()
        self.assertEqual(model_id, "Qwen/Qwen3-Reranker-0.6B")
        self.assertEqual(revision, "e61197ed45024b0ed8a2d74b80b4d909f1255473")
        self.assertEqual(accepted["file_count"], 12)
        self.assertEqual(accepted["files_sha256"],
                         "5219a20fa423e1054c34b93a5a8c0ffa640ca52320181603be32d7afe26de353")
        self.assertEqual(allowlist, [entry["path"] for entry in accepted["files"]])
        self.assertEqual(len(allowlist), 12)
        self.assertNotIn("README.md", allowlist)
        self.assertNotIn(".gitattributes", allowlist)

    def test_model_validator_still_rejects_extra_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model"
            model_path.mkdir()
            (model_path / "README.md").write_text("not an accepted model file")
            staging_manifest = Path(directory) / "staging.json"
            with self.assertRaisesRegex(RuntimeError, "do not match the pinned accepted model snapshot"):
                pilot.validate_model(model_path, staging_manifest)

    def test_cpu_child_limits_keep_margin_below_observed_host_limit(self) -> None:
        self.assertEqual(pilot.cpu_child_limits(600, 1200), (540, 600))
        self.assertEqual(pilot.cpu_child_limits(resource.RLIM_INFINITY, resource.RLIM_INFINITY), (540, 600))
        with self.assertRaisesRegex(RuntimeError, "tighter than"):
            pilot.cpu_child_limits(539, 1200)
        with self.assertRaisesRegex(RuntimeError, "tighter than"):
            pilot.cpu_child_limits(600, 599)

    def test_query_chunk_plan_is_one_sequential_complete_query(self) -> None:
        qids = [f"v5-lb-q{i:04d}" for i in range(1, 21)]
        chunks = pilot.query_chunks(qids)
        self.assertEqual(len(chunks), 20)
        self.assertEqual([qid for _, ids in chunks for qid in ids], qids)
        self.assertTrue(all(len(ids) == 1 for _, ids in chunks))

    def test_pair_and_query_coverage_and_concatenation_are_deterministic(self) -> None:
        qids = ["q-1", "q-2"]
        candidates = {
            qid: [f"{qid}-d{index:02d}" for index in range(50)] for qid in qids
        }
        records = [
            {
                "key": f"reranker|{qid}",
                "query_id": qid,
                "ranking": list(reversed(candidates[qid])),
                "scores_sha256": f"score-{qid}",
            }
            for qid in qids
        ]
        first = pilot.deterministic_rankings(qids, candidates, records)
        second = pilot.deterministic_rankings(qids, candidates, records)
        self.assertEqual(first, second)
        self.assertEqual(len(first.decode().splitlines()), 2)
        with self.assertRaisesRegex(RuntimeError, "duplicate, omission, or permutation"):
            pilot.deterministic_rankings(qids, candidates, records[:1])
        bad = [dict(records[0]), records[1]]
        bad[0]["ranking"][-1] = bad[0]["ranking"][0]
        with self.assertRaisesRegex(RuntimeError, "coverage failed"):
            pilot.deterministic_rankings(qids, candidates, bad)

    def test_parity_gate_evidence_remains_mandatory(self) -> None:
        passing = {
            "status": "pass", "exact_repeat_equality": True,
            "ranking_equal": True, "authorized_threads": 16,
            "score_tolerance": 1e-6, "max_absolute_score_delta": 1e-6,
            "profiles": {
                "4": {"scores": [0.1, 0.2, 0.3], "ranking": ["a", "b", "c"], "repeat_equal": True},
                "16": {"scores": [0.1, 0.2, 0.3], "ranking": ["a", "b", "c"], "repeat_equal": True},
            },
        }
        worker.validate_moon_parity(passing)
        failed = dict(passing, ranking_equal=False)
        with self.assertRaisesRegex(RuntimeError, "incomplete or failed"):
            worker.validate_moon_parity(failed)

    def test_blocked_preflight_emits_manifest_without_claiming_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pilot"
            digest = pilot.write_blocked(
                output, "a" * 40, "model-snapshot-validation",
                RuntimeError("snapshot contained extra files"),
            )
            blocked_path = output / "pilot_blocked.json"
            blocked = json.loads(blocked_path.read_text())
            self.assertEqual(digest, pilot.sha(blocked_path))
            self.assertEqual(blocked["blocked_stage"], "model-snapshot-validation")
            self.assertEqual(blocked["status"], "blocked-before-lockbox-scoring")
            self.assertFalse(blocked["result_bearing_execution_started"])
            self.assertFalse(blocked["qrels_read"])


if __name__ == "__main__":
    unittest.main()
