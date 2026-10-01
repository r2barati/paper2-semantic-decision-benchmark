"""Unit tests for the cross-family (llama/mistral) credential amendment.

Offline only: stubs kaggle_secrets and the anonymous HF checks. Proves
secret-first behavior, public-only-for-whitelisted-pinned-pair fallback,
fail-closed for gated/non-whitelisted pairs, and the verifier gate.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import kaggle_compute as compute

KERNEL_REL = "versions/sem2act-v5/compute/kaggle_kernel/p2_v5_consumer.py"
MISTRAL = "mistralai/Mistral-7B-Instruct-v0.3"
MISTRAL_REV = "d79d1742f78eb0cc788c11e5b41a7539d7cb56ef"
LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_REV = "0e9e39f249a16976918f6564b8830bc894c89659"


def _load_kernel(name="p2_v5_consumer_test"):
    spec = importlib.util.spec_from_file_location(name, ROOT / KERNEL_REL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stub_secrets(mode):
    stub = ModuleType("kaggle_secrets")

    class Client:
        def get_secret(self, _name):
            if mode == "conn":
                raise ConnectionError("Connection error trying to communicate with service.")
            if mode == "empty":
                return ""
            if mode == "missing":
                raise ValueError("secret not found")
            return "tok"

    stub.UserSecretsClient = Client
    return stub


class CredentialTests(unittest.TestCase):
    def setUp(self):
        self.kernel = _load_kernel()
        self.kernel.time.sleep = lambda s: None
        import os
        os.environ.pop("HF_TOKEN", None)
        os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

    def _resolve(self, mode, model, rev, anon_ok=True):
        sys.modules["kaggle_secrets"] = _stub_secrets(mode)
        if anon_ok:
            self.kernel.verify_public_revision_anonymous = lambda m, r: None
        try:
            return self.kernel.resolve_credential(model, rev)
        finally:
            sys.modules.pop("kaggle_secrets", None)

    def test_secret_path(self):
        import os
        got = self._resolve("token", MISTRAL, MISTRAL_REV)
        self.assertEqual(got, "hf_token_kaggle_secret")
        self.assertEqual(os.environ.get("HF_TOKEN"), "tok")

    def test_empty_secret_fails_closed(self):
        with self.assertRaises(SystemExit):
            self._resolve("empty", MISTRAL, MISTRAL_REV)

    def test_nonchannel_error_fails_closed(self):
        with self.assertRaises(SystemExit):
            self._resolve("missing", MISTRAL, MISTRAL_REV)

    def test_dead_service_public_fallback_for_whitelisted_pair(self):
        import os
        got = self._resolve("conn", MISTRAL, MISTRAL_REV)
        self.assertEqual(got, "public_unauthenticated")
        self.assertNotIn("HF_TOKEN", os.environ)

    def test_gated_nonwhitelisted_pair_fails_closed(self):
        # real verifier: llama is not whitelisted -> fails before any network
        sys.modules["kaggle_secrets"] = _stub_secrets("conn")
        try:
            with self.assertRaises(SystemExit):
                self.kernel.resolve_credential(LLAMA, LLAMA_REV)
        finally:
            sys.modules.pop("kaggle_secrets", None)

    def test_wrong_revision_fails_closed(self):
        # real verifier: wrong revision fails the pin check before any network
        sys.modules["kaggle_secrets"] = _stub_secrets("conn")
        try:
            with self.assertRaises(SystemExit):
                self.kernel.resolve_credential(MISTRAL, "0" * 40)
        finally:
            sys.modules.pop("kaggle_secrets", None)

    def test_anonymous_verifier_pins_present(self):
        files = self.kernel.EXPECTED_PUBLIC_FILES[MISTRAL]
        self.assertEqual(files["config.json"],
                         "affafc6478ec0fd07a32f0ca57aa2fc57743f4d17d6730f86a96ac24d1507f99")
        self.assertEqual(files["tokenizer_config.json"],
                         "cf7d90917f89931e443b02253432a581abef86379533a5b962f7f2d3b7ac8d90")
        self.assertEqual(files["tokenizer.json"],
                         "e553af6fff7d7ad76e830608b218c5c0b0822998d5a1a96099a74cd3c1cb1a49")


class VerifierTests(unittest.TestCase):
    def _run(self, manifest):
        dest = Path(tempfile.mkdtemp())
        (dest / "run_manifest.json").write_text(json.dumps(manifest))
        (dest / "raw_outputs.jsonl").write_text("")
        (dest / "beliefs.jsonl").write_text("")
        try:
            compute._verify_v5_consumer(dest, {"slug": "x"})
            return "PASS(no-exit)"
        except SystemExit as exc:
            return str(exc)

    def _base(self):
        return {"experiment_id": "v5-cross-family-consumer", "status": "pass",
                "qrels_read": False,
                "runtime_lock_sha256": compute._sha(compute.V5_KAGGLE_LOCK),
                "smoke": {"status": "pass", "fixture_sha256": compute._sha(
                    compute.ROOT / "versions/sem2act-v5/fixtures/runtime_smoke_v1.json")},
                "expected_calls": 1920, "actual_calls": 1920, "n_fail": 0,
                "prompt_shas": {"C1": "66e6890ba9c5464c", "C3": "5966cadde90e8236"}}

    def test_missing_credential_fails(self):
        self.assertIn("credential source gate failed", self._run(self._base()))

    def test_public_credential_passes_gate(self):
        got = self._run({**self._base(), "credential_source": "public_unauthenticated"})
        # proceeds past credential gate to the next (raw-output) gate
        self.assertIn("raw-output coverage failed", got)


if __name__ == "__main__":
    unittest.main()
