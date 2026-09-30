"""Tests for the Qwen cache-path fix and dedup-aware coverage.

Offline only: no GPU, no lockbox, no network, no Kaggle. Uses the tracked
src/ consumer modules (byte-identical to the staged bundle) and synthetic
inputs, plus static checks on the kernel and verifier.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import kaggle_compute as compute

KERNEL_REL = "versions/sem2act-v5/compute/kaggle_kernel/p2_v5_qwen_consumer.py"
MODEL = "Qwen/Qwen3-8B-AWQ"
C1_SHA = "66e6890ba9c5464c"
C3_SHA = "5966cadde90e8236"

C1_JSON = {"normal": 0.2, "supplier_delay": 0.5, "demand_surge": 0.3,
           "estimated_lt_increase": 1.0, "estimated_duration": 2.0,
           "estimated_demand_multiplier": 1.1}
C3_JSON = {"entity_match": True, "event": "supplier_delay", "fresh": True,
           "stance": "support", "confidence": 0.8}

PYDANTIC = importlib.util.find_spec("pydantic") is not None


def _load_kernel():
    spec = importlib.util.spec_from_file_location(
        "p2_v5_qwen_consumer_test", ROOT / KERNEL_REL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stub_client(counter, c1_payload=C1_JSON, c3_payload=C3_JSON):
    import src.consumers_v3 as consumers

    def _client():
        class _Completions:
            @staticmethod
            def create(model, temperature, max_tokens, messages):
                counter["n"] += 1
                body = c1_payload if messages[0]["content"] == consumers.PROMPT_C1 else c3_payload
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(body)))],
                    usage=None)
        return SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    return _client


def _synthetic_rows():
    docs_a = [{"doc_id": "d1", "text": "supplier delay at node"},
              {"doc_id": "d2", "text": "orders normal"},
              {"doc_id": "d3", "text": "demand surge nearby"}]
    docs_b = [{"doc_id": "d1", "text": "supplier delay at node"},
              {"doc_id": "d2", "text": "orders normal"},
              {"doc_id": "d9", "text": "all quiet"}]
    rows = []
    for system in ("bm25", "rerank"):
        for qi, docs in (("v5-lb-q0001", docs_a), ("v5-lb-q0002", docs_b)):
            rows.append({"query_id": qi, "system": system, "query_text": "risk?",
                         "documents": docs,
                         "metadata": {"entity_node": "node", "true_regime": "normal"}})
    return rows


class CacheBindingTests(unittest.TestCase):
    def test_env_set_precedes_consumer_import_in_kernel(self):
        src = (ROOT / KERNEL_REL).read_text()
        env_at = src.index('os.environ["PAPER2_V3_CACHE_DIR"] = "/kaggle/working/cache_out"')
        import_at = src.index("import src.consumers_v3 as consumers")
        self.assertLess(env_at, import_at)

    def test_kernel_asserts_cache_binding(self):
        src = (ROOT / KERNEL_REL).read_text()
        self.assertIn("consumers.CACHE_DIR != Path(os.environ[\"PAPER2_V3_CACHE_DIR\"])", src)

    @unittest.skipUnless(PYDANTIC, "pydantic required")
    def test_env_before_import_binds_cache_dir(self):
        import src.consumers_v3 as consumers
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["PAPER2_V3_CACHE_DIR"] = tmp
            importlib.reload(consumers)
            self.assertEqual(consumers.CACHE_DIR, Path(tmp))


@unittest.skipUnless(PYDANTIC, "pydantic required")
class DedupTests(unittest.TestCase):
    def test_repeated_invocations_deduplicate(self):
        import src.consumers_v3 as consumers
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["PAPER2_V3_CACHE_DIR"] = tmp
            importlib.reload(consumers)
            counter = {"n": 0}
            consumers._client = _stub_client(counter)
            try:
                rows = _synthetic_rows()
                beliefs = []
                for row in sorted(rows, key=lambda r: (r["system"], r["query_id"])):
                    docs = [{"doc_id": d["doc_id"], "text": d["text"]} for d in row["documents"]]
                    beliefs.append(consumers.consume_c1(docs, row["query_text"], MODEL))
                    beliefs.append(consumers.consume_c3(
                        docs, row["query_text"], MODEL, row["metadata"]["entity_node"]))
                files = list(Path(tmp).glob("*.json"))
                # 4 rows x (1 C1 + 3 C3) = 16 invocations; shared docs dedup to fewer files
                self.assertEqual(len(beliefs), 8)
                self.assertLess(len(files), 16)
                self.assertEqual(counter["n"], len(files))
                for b in beliefs:
                    self.assertAlmostEqual(b.p_normal + b.p_supplier_delay + b.p_demand_surge, 1.0)
            finally:
                importlib.reload(consumers)


class KeyAgreementTests(unittest.TestCase):
    def test_kernel_and_verifier_name_keys_identically(self):
        kernel = _load_kernel()
        users = ["Operator information need: q\n\nEvidence:\n[DOC d] t",
                 "Operator's own node: n\n\nEvidence document:\nt",
                 "x" * 5000]
        for psha in (C1_SHA, C3_SHA):
            for user in users:
                self.assertEqual(
                    kernel._invocation_cache_name(MODEL, psha, user),
                    compute._v5_qwen_cache_fname(MODEL, psha, user))


class RawCoverageTests(unittest.TestCase):
    def test_pass_missing_order_and_parse(self):
        keys = ["a.json", "b.json", "a.json", "c.json"]
        lines = [json.dumps({"cache_file": k, "raw": "{}"}, sort_keys=True) for k in keys]
        out = compute._v5_qwen_check_raw_coverage(lines, keys)
        self.assertEqual(out, {"n_invocations": 4, "n_unique_keys": 3})
        with self.assertRaises(SystemExit):
            compute._v5_qwen_check_raw_coverage(lines[:3], keys)
        swapped = [lines[1], lines[0], lines[2], lines[3]]
        with self.assertRaises(SystemExit):
            compute._v5_qwen_check_raw_coverage(swapped, keys)
        wrong = [json.dumps({"cache_file": "zzz.json"})] + lines[1:]
        with self.assertRaises(SystemExit):
            compute._v5_qwen_check_raw_coverage(wrong, keys)
        with self.assertRaises(SystemExit):
            compute._v5_qwen_check_raw_coverage(["not json"] + lines[1:], keys)


class CanonicalRawTests(unittest.TestCase):
    def test_canonical_order_construction(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            payloads = {"f1.json": {"raw": "r1"}, "f2.json": {"raw": "r2"}}
            for name, payload in payloads.items():
                (cache / name).write_text(json.dumps(payload))
            order = ["f1.json", "f2.json", "f1.json"]
            lines = []
            for fname in order:
                payload = json.loads((cache / fname).read_text())
                payload["cache_file"] = fname
                lines.append(json.dumps(payload, sort_keys=True))
            self.assertEqual([json.loads(l)["cache_file"] for l in lines], order)
            self.assertEqual(len(lines), 3)
            self.assertEqual({json.loads(l)["cache_file"] for l in lines}, set(payloads))


class RuntimeVersionsTests(unittest.TestCase):
    def test_versions_doc_reports_post_install_runtime(self):
        kernel = _load_kernel()
        info = {"torch": "2.8.0+cu128", "cuda_reported": "12.8",
                "device_name": "Tesla T4", "vllm": "0.11.0",
                "packages": {"transformers": "4.57.6"}}
        doc = kernel._runtime_versions_doc(info)
        self.assertEqual(doc["torch"], "2.8.0+cu128")
        self.assertEqual(doc["vllm"], "0.11.0")
        self.assertEqual(doc["transformers"], "4.57.6")
        self.assertNotIn("2.10.0", json.dumps(doc))


class PostInstallGateTests(unittest.TestCase):
    def _run(self, payload, rc=0):
        kernel = _load_kernel()

        class _CP:
            returncode = rc
            stdout = (json.dumps(payload) + "\n") if rc == 0 else ""
            stderr = "" if rc == 0 else "boom"
        kernel.subprocess.run = lambda *a, **k: _CP()
        try:
            return kernel.verify_post_install_runtime()
        finally:
            import subprocess as _sp
            kernel.subprocess.run = _sp.run

    def _good(self):
        return {"torch": "2.8.0+cu128", "cuda_reported": "12.8",
                "cuda_available": True, "device_count": 1,
                "device_name": "Tesla T4", "matmul_pass": True, "vllm": "0.11.0",
                "packages": {"transformers": "4.57.6", "tokenizers": "0.22.2",
                             "triton": "3.4.0", "xformers": "0.0.32.post1",
                             "torchvision": "0.23.0", "torchaudio": "2.8.0",
                             "openai": "2.48.0", "pydantic": "2.12.5"}}

    def test_good_tuple_returns(self):
        self.assertTrue(self._run(self._good())["matmul_pass"])

    def test_torch_drift_fails_closed(self):
        with self.assertRaises(SystemExit):
            self._run({**self._good(), "torch": "2.10.0+cu128"})

    def test_matmul_fails_closed(self):
        with self.assertRaises(SystemExit):
            self._run({**self._good(), "matmul_pass": False})


@unittest.skipUnless(PYDANTIC, "pydantic required")
class SyntheticEndToEndTests(unittest.TestCase):
    def test_repeated_input_smoke_end_to_end(self):
        import src.consumers_v3 as consumers
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["PAPER2_V3_CACHE_DIR"] = tmp
            os.environ["LLM_API_KEY"] = "local"
            importlib.reload(consumers)
            counter = {"n": 0}
            consumers._client = _stub_client(counter)
            try:
                rows = _synthetic_rows()
                invocations = []
                for row in sorted(rows, key=lambda r: (r["system"], r["query_id"])):
                    docs = [{"doc_id": d["doc_id"], "text": d["text"]} for d in row["documents"]]
                    c1u = ("Operator information need: " + row["query_text"] + "\n\nEvidence:\n" +
                           "\n\n".join(f"[DOC {d['doc_id']}] {d['text']}" for d in docs))
                    invocations.append(compute._v5_qwen_cache_fname(MODEL, C1_SHA, c1u))
                    consumers.consume_c1(docs, row["query_text"], MODEL)
                    for d in docs:
                        c3u = ("Operator's own node: " + row["metadata"]["entity_node"] +
                               "\n\nEvidence document:\n" + d["text"])
                        invocations.append(compute._v5_qwen_cache_fname(MODEL, C3_SHA, c3u))
                    consumers.consume_c3(docs, row["query_text"], MODEL,
                                         row["metadata"]["entity_node"])
                files = {p.name for p in Path(tmp).glob("*.json")}
                self.assertEqual(len(invocations), 16)
                self.assertLess(len(files), 16)  # dedup proven
                self.assertEqual(counter["n"], len(files))
                self.assertEqual(set(invocations), files)
                lines = []
                for fname in invocations:
                    payload = json.loads((Path(tmp) / fname).read_text())
                    payload["cache_file"] = fname
                    lines.append(json.dumps(payload, sort_keys=True))
                out = compute._v5_qwen_check_raw_coverage(lines, invocations)
                self.assertEqual(out["n_invocations"], 16)
            finally:
                importlib.reload(consumers)


if __name__ == "__main__":
    unittest.main()
