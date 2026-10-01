"""Unit tests for the rented-GPU Llama launcher helpers.

Offline only: stubbed downloads, temp dirs, no network, no GPU, no tokens.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_llama_consumer_test",
    ROOT / "versions/sem2act-v5/compute/rented_gpu/run_llama_consumer.py",
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class IdentityTests(unittest.TestCase):
    def test_matching_files_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.json").write_bytes(b'{"x": 1}')
            want = {"a.json": hashlib.sha256(b'{"x": 1}').hexdigest()}
            self.assertIsNone(launcher.assert_file_identities(root, want))

    def test_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.json").write_bytes(b'{"x": 2}')
            want = {"a.json": hashlib.sha256(b'{"x": 1}').hexdigest()}
            with self.assertRaises(SystemExit):
                launcher.assert_file_identities(root, want)

    def test_missing_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                launcher.assert_file_identities(Path(tmp), {"gone.json": "00" * 32})


class CredentialChannelTests(unittest.TestCase):
    def test_missing_token_fails_closed(self):
        old = os.environ.pop("SEM2ACT_HF_TOKEN", None)
        try:
            with self.assertRaises(SystemExit):
                launcher.require_token()
        finally:
            if old is not None:
                os.environ["SEM2ACT_HF_TOKEN"] = old

    def test_stub_returns_token_without_touching_disk(self):
        launcher.install_stub_secrets("tok-value")
        try:
            from kaggle_secrets import UserSecretsClient
            self.assertEqual(UserSecretsClient().get_secret("HF_TOKEN"), "tok-value")
            with self.assertRaises(KeyError):
                UserSecretsClient().get_secret("OTHER")
        finally:
            sys.modules.pop("kaggle_secrets", None)

    def test_redact_strips_credentials(self):
        url = "https://user:secret@example.com/x?tok=abc"
        out = launcher.redact(url)
        self.assertNotIn("secret", out)
        self.assertNotIn("user", out.split("@")[0] if "@" in out else out)
        self.assertIn("example.com/x", out)


class LayoutTests(unittest.TestCase):
    def test_kaggle_layout_from_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            bundle.mkdir()
            for name in launcher.BUNDLE_FILES:
                (bundle / name).write_text("{}")
            inp, work = launcher.build_kaggle_layout(bundle, Path(tmp) / "kaggle")
            for name in launcher.BUNDLE_FILES:
                self.assertTrue((inp / name).is_file())
            self.assertTrue(work.is_dir())

    def test_incomplete_bundle_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                launcher.build_kaggle_layout(Path(tmp), Path(tmp) / "kaggle")


if __name__ == "__main__":
    unittest.main()
