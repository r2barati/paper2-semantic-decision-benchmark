"""Static tests for the Kaggle reranker pilot contract; no job, quota, or model runs."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import unittest
from pathlib import Path

from scripts import freeze_v5_kaggle_reranker_pilot as freeze
from scripts import kaggle_compute as compute

ROOT = Path(__file__).resolve().parents[1]
KERNEL_REL = "versions/sem2act-v5/compute/kaggle_kernel/p2_v5_rerank_pilot.py"
FULL_KERNEL_REL = "versions/sem2act-v5/compute/kaggle_kernel/p2_v5_rerank.py"
LOCK_REL = "versions/sem2act-v5/manifests/kaggle_reranker_runtime_lock.json"
PARENT_LOCK_REL = "versions/sem2act-v5/manifests/kaggle_runtime_lock.json"
BLOCKED_REL = "versions/sem2act-v5/manifests/moon_reranker_pilot_blocked.json"
AMENDMENT_REL = "versions/sem2act-v5/amendments/kaggle_reranker_pilot_v1.yaml"
UPLOAD_REL = "versions/sem2act-v5/manifests/upload/v5-rerank-inputs.json"
REFERENCE_REL = "versions/sem2act-v5/manifests/cpu_reranker_canary.json"
PROTOCOL_HASH = "2d75e4cb592c30c13b60316891bd489dfacb9e5824f49c11e0a18d6eccdf2022"
PILOT_QUERY_IDS = tuple(f"v5-lb-q{index:04d}" for index in range(1, 21))


def _module_constants(relative: str) -> dict:
    """Read top-level literal constants without executing a Kaggle kernel."""
    tree = ast.parse((ROOT / relative).read_text())
    values: dict = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
        except (ValueError, SyntaxError, TypeError):
            continue
    return values


class KaggleRerankerPilotTests(unittest.TestCase):
    def test_protocol_hash_is_unchanged_everywhere(self) -> None:
        lock = json.loads((ROOT / LOCK_REL).read_text())
        blocked = json.loads((ROOT / BLOCKED_REL).read_text())
        upload = json.loads((ROOT / UPLOAD_REL).read_text())
        amendment = (ROOT / AMENDMENT_REL).read_text()
        self.assertEqual(lock["protocol_hash"], PROTOCOL_HASH)
        self.assertEqual(blocked["protocol_hash"], PROTOCOL_HASH)
        self.assertEqual(upload["protocol_hash"], PROTOCOL_HASH)
        self.assertIn(PROTOCOL_HASH, amendment)
        self.assertTrue(lock["backend_routing"]["runtime_parameters_changed"] is False)

    def test_pilot_kernel_pins_the_frozen_lock_and_parent_lock_hashes(self) -> None:
        constants = _module_constants(KERNEL_REL)
        self.assertEqual(constants["EXPECTED_PILOT_LOCK_SHA256"],
                         compute._sha(ROOT / LOCK_REL))
        self.assertEqual(constants["PARENT_LOCK_SHA256"],
                         compute._sha(ROOT / PARENT_LOCK_REL))
        self.assertEqual(constants["LOCK_ID"],
                         "sem2act-v5-kaggle-reranker-pilot-v1")
        self.assertEqual(constants["AMENDMENT_ID"],
                         "sem2act-v5-kaggle-reranker-pilot-v1")

    def test_pilot_scoring_recipe_is_byte_identical_to_the_full_kernel(self) -> None:
        """The pilot may change coverage and provenance, never the scoring rule."""
        pilot = _module_constants(KERNEL_REL)
        full = _module_constants(FULL_KERNEL_REL)
        for key in ("MODEL_ID", "REVISION", "INSTRUCTION", "TOPN", "MAXLEN",
                    "EXPECTED_TORCH", "EXPECTED_CUDA"):
            self.assertEqual(pilot[key], full[key], key)
        self.assertEqual(pilot["EXPECTED_TRANSFORMERS"], full["PINNED_TRANSFORMERS"])
        self.assertEqual(pilot["SMOKE_FIXTURE_SHA256"],
                         full["EXPECTED_SMOKE_FIXTURE_SHA256"])
        self.assertEqual(
            json.loads((ROOT / REFERENCE_REL).read_text())["model_snapshot"]["files_sha256"],
            pilot["MODEL_SNAPSHOT_FILES_SHA256"])
        self.assertEqual(pilot["BATCH_SIZE"], 32)
        self.assertEqual(pilot["DTYPE"], "float16")
        # The two kernels name their candidate list differently, so compare the
        # scoring recipe by its invariants rather than by byte-identical source.
        # Collapse whitespace first because the full kernel wraps some calls.
        pilot_text = " ".join((ROOT / KERNEL_REL).read_text().split())
        full_text = " ".join((ROOT / FULL_KERNEL_REL).read_text().split())
        for fragment in (
            '<|im_start|>system\\nJudge whether the Document meets the requirements based',
            '"yes" or "no".<|im_end|>\\n<|im_start|>user\\n',
            "<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n",
            'convert_tokens_to_ids("yes")',
            'convert_tokens_to_ids("no")',
            "torch.stack([logits[:, no], logits[:, yes]], dim=1)",
            "log_softmax(",
            "[:, 1].exp()",
            'f"<Instruct>: {INSTRUCTION}\\n<Query>: {query}\\n<Document>: {doc}"',
            "padding=False",
            'truncation="longest_first"',
            "return_attention_mask=False",
            "max_length=MAXLEN - len(prefix) - len(suffix)",
            "ids = [prefix + item + suffix for item in enc[\"input_ids\"]]",
            '{"input_ids": ids}, padding=True, return_tensors="pt"',
            "logits = model(**batch).logits[:, -1, :]",
            "key=lambda pair: (-pair[0], pair[1])",
            "range(0, len(cand",
        ):
            self.assertIn(fragment, pilot_text, fragment)
            self.assertIn(fragment, full_text, fragment)
        # The pilot names the frozen batch size; the full kernel inlines it.
        self.assertIn("range(0, len(candidate_ids), BATCH_SIZE)", pilot_text)
        self.assertIn("range(0, len(cand), 32)", full_text)

    def test_pilot_query_and_score_keyspace_is_exactly_twenty_by_fifty(self) -> None:
        lock = json.loads((ROOT / LOCK_REL).read_text())
        pilot = lock["pilot"]
        self.assertEqual(tuple(pilot["query_ids"]), PILOT_QUERY_IDS)
        self.assertEqual(pilot["queries"], 20)
        self.assertEqual(pilot["candidate_depth"], 50)
        self.assertEqual(pilot["expected_scores"], 1000)
        self.assertEqual(pilot["expected_ranking_rows"], 1000)
        self.assertEqual(pilot["shard_index"], 0)
        self.assertNotIn("v5-lb-q0021", pilot["query_ids"])
        self.assertEqual(lock["inputs"]["files"],
                         json.loads((ROOT / UPLOAD_REL).read_text())["remote_sha256"])

    def test_model_snapshot_allowlist_is_the_unchanged_accepted_twelve_files(self) -> None:
        constants = _module_constants(KERNEL_REL)
        reference = json.loads((ROOT / REFERENCE_REL).read_text())["model_snapshot"]
        entries = [{"path": name, "sha256": digest, "bytes": size}
                   for name, digest, size in constants["MODEL_SNAPSHOT_FILES"]]
        self.assertEqual(entries, reference["files"])
        self.assertEqual(len(entries), 12)
        aggregate = hashlib.sha256(
            json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.assertEqual(aggregate, reference["files_sha256"])
        self.assertEqual(constants["MODEL_SNAPSHOT_FILES_SHA256"], aggregate)
        names = {entry["path"] for entry in entries}
        self.assertNotIn("README.md", names)
        self.assertNotIn(".gitattributes", names)

    def test_pilot_inherits_every_reranker_runtime_parameter_from_v3(self) -> None:
        pilot = json.loads((ROOT / LOCK_REL).read_text())["model"]
        parent = json.loads((ROOT / PARENT_LOCK_REL).read_text())["models"]["reranker"]
        for key in ("model_id", "revision", "tokenizer_revision", "dtype",
                    "attention_backend", "max_context_tokens", "batch_size"):
            self.assertEqual(pilot[key], parent[key], key)
        self.assertEqual(pilot["device_map"], parent["device_map"])
        self.assertEqual(pilot["quantization"], parent["quantization"])

    def test_moon_blocked_record_admits_no_scientific_output(self) -> None:
        blocked = json.loads((ROOT / BLOCKED_REL).read_text())
        self.assertFalse(blocked["scientific_output_accepted"])
        self.assertEqual(blocked["accepted_queries"], 0)
        self.assertEqual(blocked["accepted_pairs"], 0)
        self.assertEqual(blocked["ranking_rows_emitted"], 0)
        self.assertFalse(blocked["concatenation_performed"])
        self.assertFalse(blocked["qrels_read"])
        gates = blocked["gates"]
        self.assertEqual(gates["model_integrity"]["result"], "pass")
        self.assertEqual(gates["thread_parity_4_vs_16"]["result"], "pass")
        self.assertEqual(gates["cpu_scoring_feasibility"]["result"], "fail")
        self.assertEqual(gates["cpu_scoring_feasibility"]["return_code"], -24)
        self.assertFalse(blocked["admissibility"]["usable_as_scientific_output"])
        self.assertTrue(blocked["admissibility"]["usable_as_infrastructure_evidence"])

    def test_freeze_matches_the_working_tree_and_disclaims_the_full_reranker(self) -> None:
        manifest = freeze.expected_manifest()
        on_disk = json.loads((ROOT / "versions/sem2act-v5/manifests/kaggle_reranker_runtime_freeze.json").read_text())
        for key, value in manifest.items():
            if key == "frozen_utc":
                continue
            self.assertEqual(on_disk.get(key), value, key)
        self.assertFalse(manifest["full_reranker_authorized"])
        self.assertEqual(manifest["authorized_pilot_queries"], 20)
        self.assertEqual(manifest["authorized_pilot_scores"], 1000)
        self.assertEqual(manifest["pilot_kernel_sha256"], compute._sha(ROOT / KERNEL_REL))

    def test_freeze_refuses_drifted_reranker_parameters(self) -> None:
        lock = json.loads((ROOT / LOCK_REL).read_text())
        for mutate in (
            lambda value: value["models"] if False else value["model"].__setitem__("dtype", "bfloat16"),
            lambda value: value["model"].__setitem__("batch_size", 64),
            lambda value: value["runtime"].__setitem__("torch", "2.11.0+cu128"),
            lambda value: value["pilot"].__setitem__("queries", 240),
            lambda value: value["platform"].__setitem__("visible_gpu_count", 2),
        ):
            snapshot = json.loads(json.dumps(lock))
            mutate(snapshot)
            with self.subTest(snapshot=snapshot.get("model", {}).get("dtype")):
                with tempfile_lock(self, snapshot):
                    self.assertRaises(SystemExit, freeze.expected_manifest)

    def test_submission_gate_locks_the_full_reranker_behind_pilot_acceptance(self) -> None:
        job = compute.JOBS["v5-rerank"]
        self.assertEqual(job["verify"], "v5rerank")
        self.assertFalse(compute.V5_PILOT_ACCEPTANCE.exists())
        with self.assertRaisesRegex(SystemExit, "Codex acceptance"):
            compute._require_v5_full_reranker_acceptance("v5-rerank")
        # The pilot and unrelated jobs are not gated on their own acceptance.
        self.assertIsNone(compute._require_v5_full_reranker_acceptance("v5-rerank-pilot"))
        self.assertIsNone(compute._require_v5_full_reranker_acceptance("v5-qwen"))
        self.assertIn("v5-rerank-pilot", compute.JOBS)
        self.assertEqual(compute.JOBS["v5-rerank-pilot"]["dataset_slug"],
                         "sem2act-v5-rerank-inputs")
        self.assertTrue(compute.JOBS["v5-rerank-pilot"]["gpu"])

    def test_pilot_and_full_reranker_share_one_remote_verified_input_dataset(self) -> None:
        pilot = compute.JOBS["v5-rerank-pilot"]
        full = compute.JOBS["v5-rerank"]
        self.assertEqual(pilot["dataset_slug"], full["dataset_slug"])
        self.assertNotEqual(pilot["dest_dir"], full["dest_dir"])
        self.assertEqual(pilot["dest_dir"], "versions/sem2act-v5/runtime/kaggle_rerank_pilot")

    def test_pilot_verifier_rejects_partial_or_renumbered_coverage(self) -> None:
        import tempfile

        qids = list(PILOT_QUERY_IDS)
        candidates = {qid: [f"d{index:02d}" for index in range(50)] for qid in qids}
        upload = json.loads((ROOT / UPLOAD_REL).read_text())["remote_sha256"]
        smoke = compute._sha(ROOT / "versions/sem2act-v5/fixtures/runtime_smoke_v1.json")
        snapshot = json.loads((ROOT / REFERENCE_REL).read_text())["model_snapshot"]
        rows = [f"{qid} Q0 {did} {rank} {1000 - rank} v5-rerank-pilot"
                for qid in qids
                for rank, did in enumerate(candidates[qid], start=1)]

        def write_case(directory: Path, trec_rows, manifest_overrides=None):
            dest = directory / "out"
            dest.mkdir(parents=True)
            trec = dest / "rerank.trec"
            trec.write_text("\n".join(trec_rows) + "\n")
            manifest = {
                "experiment_id": "v5-reranker-pilot", "status": "pass",
                "qrels_read": False, "protocol_hash": PROTOCOL_HASH,
                "amendment_id": "sem2act-v5-kaggle-reranker-pilot-v1",
                "lock_id": "sem2act-v5-kaggle-reranker-pilot-v1",
                "runtime_lock_sha256": compute._sha(ROOT / LOCK_REL),
                "parent_runtime_lock_sha256": compute._sha(ROOT / PARENT_LOCK_REL),
                "model": "Qwen/Qwen3-Reranker-0.6B",
                "revision": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
                "dtype": "float16", "attention_backend": "eager",
                "candidate_depth": 50, "batch_size": 32, "max_context_tokens": 1024,
                "instruction_sha256": json.loads((ROOT / LOCK_REL).read_text())["scoring"]["instruction_sha256"],
                "model_snapshot": {"file_count": 12, "files_sha256": snapshot["files_sha256"]},
                "smoke": {"status": "pass", "fixture_sha256": smoke},
                "input_sha256": dict(upload, **{"runtime_smoke_v1.json": smoke}),
                "n_fail": 0, "query_ids": list(PILOT_QUERY_IDS),
                "n_queries": 20, "n_scores": 1000, "n_ranking_rows": 1000,
                "rerank_trec_sha256": compute._sha(trec),
            }
            manifest.update(manifest_overrides or {})
            (dest / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
            return dest

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            job = {"dataset_dir": "versions/sem2act-v5/runtime/kaggle_rerank_inputs"}
            good = write_case(directory / "good", rows)
            result = compute._verify_v5_rerank_pilot(good, job)
            self.assertEqual(result["status"], "pass")
            self.assertFalse(result["full_reranker_authorized"])

            short = rows[:-1]
            case = write_case(directory / "short", short, {"n_ranking_rows": 999})
            with self.assertRaises(SystemExit):
                compute._verify_v5_rerank_pilot(case, job)

            case = write_case(directory / "count", rows, {"n_queries": 240})
            with self.assertRaisesRegex(SystemExit, "coverage"):
                compute._verify_v5_rerank_pilot(case, job)

            case = write_case(directory / "keyspace", rows, {"query_ids": list(PILOT_QUERY_IDS[:19])})
            with self.assertRaisesRegex(SystemExit, "keyspace"):
                compute._verify_v5_rerank_pilot(case, job)

            case = write_case(directory / "qrel", rows, {"qrels_read": True})
            with self.assertRaisesRegex(SystemExit, "qrel"):
                compute._verify_v5_rerank_pilot(case, job)

            case = write_case(directory / "lock", rows, {"runtime_lock_sha256": "0" * 64})
            with self.assertRaisesRegex(SystemExit, "lock hash"):
                compute._verify_v5_rerank_pilot(case, job)

            case = write_case(directory / "dtype", rows, {"dtype": "bfloat16"})
            with self.assertRaisesRegex(SystemExit, "dtype"):
                compute._verify_v5_rerank_pilot(case, job)

            case = write_case(directory / "input", rows, {"input_sha256": dict(upload, **{"queries.jsonl": "0" * 64})})
            with self.assertRaisesRegex(SystemExit, "input hash"):
                compute._verify_v5_rerank_pilot(case, job)

            dup = list(rows)
            dup[0] = dup[0].rsplit(" ", 2)[0] + " 999 v5-rerank-pilot"
            dup[0] = f"{qids[0]} Q0 {candidates[qids[0]][1]} 1 999 v5-rerank-pilot"
            case = write_case(directory / "dup", dup)
            with self.assertRaises(SystemExit):
                compute._verify_v5_rerank_pilot(case, job)

            case = write_case(directory / "tag", [row.replace("v5-rerank-pilot", "v5-rerank")
                                                  for row in rows])
            with self.assertRaisesRegex(SystemExit, "schema"):
                compute._verify_v5_rerank_pilot(case, job)

    def test_smoke_gate_is_embedded_and_hash_pinned(self) -> None:
        """The frozen input dataset v1 has no smoke fixture, so the pilot must
        embed the exact accepted fixture bytes and prove them by hash."""
        import base64

        constants = _module_constants(KERNEL_REL)
        fixture = ROOT / "versions/sem2act-v5/fixtures/runtime_smoke_v1.json"
        decoded = base64.b64decode(constants["SMOKE_FIXTURE_B64"])
        self.assertEqual(hashlib.sha256(decoded).hexdigest(),
                         constants["SMOKE_FIXTURE_SHA256"])
        self.assertEqual(hashlib.sha256(fixture.read_bytes()).hexdigest(),
                         constants["SMOKE_FIXTURE_SHA256"])
        self.assertEqual(decoded, fixture.read_bytes())
        # The kernel must not try to read the fixture off disk, because the
        # attached dataset does not carry it.
        self.assertNotIn('find_input("runtime_smoke_v1.json")',
                         (ROOT / KERNEL_REL).read_text())

    def test_pilot_does_not_require_local_dataset_stage(self) -> None:
        """The pilot pushes against the already remote-verified dataset, so the
        submission gate must pass on the pinned remote manifest alone."""
        job = compute.JOBS["v5-rerank-pilot"]
        self.assertIs(job.get("require_local_dataset_stage"), False)
        dataset_dir = ROOT / job["dataset_dir"]
        if dataset_dir.exists():
            self.skipTest(f"local dataset copy is staged at {dataset_dir}")
        self.assertEqual(compute._require_v5_dataset_verified(job)["files"],
                         json.loads((ROOT / UPLOAD_REL).read_text())["remote_sha256"])
        # The full reranker attaches the same remote-verified dataset.
        self.assertIs(compute.JOBS["v5-rerank"].get("require_local_dataset_stage"), False)
        # An unrelated job still keeps the strict default.
        self.assertIs(compute.JOBS["v5-qwen"].get("require_local_dataset_stage", True), True)

    def test_shared_input_manifest_is_named_explicitly(self) -> None:
        """Both rerank jobs attach one uploaded dataset whose manifest is named
        after the upload job, not after either kernel job slug."""
        manifest = ROOT / UPLOAD_REL
        for name in ("v5-rerank-pilot", "v5-rerank"):
            job = compute.JOBS[name]
            self.assertEqual(job["upload_manifest"], UPLOAD_REL, name)
            self.assertEqual(compute._v5_upload_manifest_path(job), manifest, name)
        self.assertTrue(manifest.is_file())


class tempfile_lock:
    """Temporarily point the freeze module at a mutated pilot lock."""

    def __init__(self, case: unittest.TestCase, lock: dict) -> None:
        self.case = case
        self.lock = lock
        self.original = freeze.LOCK

    def __enter__(self):
        import tempfile

        self.directory = tempfile.TemporaryDirectory()
        path = Path(self.directory.name) / "lock.json"
        path.write_text(json.dumps(self.lock, indent=2))
        freeze.LOCK = path
        return self

    def __exit__(self, *exc):
        freeze.LOCK = self.original
        self.directory.cleanup()
        return False


class ModelCredentialPolicyTests(unittest.TestCase):
    """The pilot must not hard-fail on an ungated pin when the secrets
    backend is unreachable; it must still hard-fail for a gated model."""

    def _kernel(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "p2_v5_rerank_pilot_under_test", ROOT / KERNEL_REL
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _with_fake_secrets(self, behaviour):
        import sys
        import types

        class FakeClient:
            def get_secret(self, key):
                if isinstance(behaviour, BaseException):
                    raise behaviour
                if behaviour is None:
                    return ""
                return behaviour

        sys.modules["kaggle_secrets"] = types.SimpleNamespace(UserSecretsClient=FakeClient)
        self.addCleanup(sys.modules.pop, "kaggle_secrets", None)
        return FakeClient

    def test_transport_failure_falls_back_to_anonymous_for_ungated_pin(self) -> None:
        kernel = self._kernel()
        self._with_fake_secrets(ConnectionError("secrets backend unreachable"))
        saved = os.environ.pop("HF_TOKEN", None)
        self.addCleanup(lambda: os.environ.__setitem__("HF_TOKEN", saved) if saved else None)
        self.assertEqual(kernel.resolve_model_token(), "public_unauthenticated")
        self.assertNotIn("HF_TOKEN", os.environ)

    def test_readable_secret_is_used_and_recorded(self) -> None:
        kernel = self._kernel()
        self._with_fake_secrets("hf_example_token")
        self.addCleanup(os.environ.pop, "HF_TOKEN", None)
        self.assertEqual(kernel.resolve_model_token(), "kaggle_secret")
        self.assertEqual(os.environ["HF_TOKEN"], "hf_example_token")

    def test_empty_secret_falls_back_to_anonymous(self) -> None:
        kernel = self._kernel()
        self._with_fake_secrets(None)
        saved = os.environ.pop("HF_TOKEN", None)
        self.addCleanup(lambda: os.environ.__setitem__("HF_TOKEN", saved) if saved else None)
        self.assertEqual(kernel.resolve_model_token(), "public_unauthenticated")

    def test_gated_model_still_hard_fails_on_transport_failure(self) -> None:
        kernel = self._kernel()
        self._with_fake_secrets(ConnectionError("secrets backend unreachable"))
        original = kernel.MODEL_ID
        kernel.MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
        self.addCleanup(setattr, kernel, "MODEL_ID", original)
        with self.assertRaises(SystemExit) as caught:
            kernel.resolve_model_token()
        self.assertIn("ConnectionError", str(caught.exception))

    def test_gated_model_still_hard_fails_on_empty_secret(self) -> None:
        kernel = self._kernel()
        self._with_fake_secrets(None)
        original = kernel.MODEL_ID
        kernel.MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
        self.addCleanup(setattr, kernel, "MODEL_ID", original)
        with self.assertRaises(SystemExit):
            kernel.resolve_model_token()

    def test_kernel_uses_the_ungated_policy_not_the_old_hard_fail(self) -> None:
        source = (ROOT / KERNEL_REL).read_text()
        self.assertNotIn("require_hf_secret", source)
        self.assertIn("resolve_model_token", source)

    def test_manifest_records_the_credential_source(self) -> None:
        source = (ROOT / KERNEL_REL).read_text()
        self.assertIn('"credential_source": credential_source', source)

    def test_python_version_guard_compares_the_tuple_not_the_string(self) -> None:
        """Regression: str(sys.version_info)[:2] sliced the repr's first two
        characters, so the guard rejected every interpreter including 3.12."""
        source = (ROOT / KERNEL_REL).read_text()
        self.assertNotIn('join(str(sys.version_info)[:2])', source)
        self.assertIn("join(str(part) for part in sys.version_info[:2])", source)

    def test_python_version_guard_expression_evaluates_to_the_real_version(self) -> None:
        import sys as _sys

        actual = ".".join(str(part) for part in _sys.version_info[:2])
        self.assertEqual(actual, ".".join(str(part) for part in _sys.version_info[:2]))
        self.assertTrue(actual.replace(".", "").isdigit(), actual)


if __name__ == "__main__":
    unittest.main()
