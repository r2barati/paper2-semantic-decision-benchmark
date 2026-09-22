"""Validation for the sealed FINAL_BANK_V2 (EXP-P2-REASONING-AGENTIC-v1.1).

The bank file itself (results/reasoning_agentic/FINAL_BANK_V2.json) is LOCAL-ONLY
and never committed, so this test SKIPS when it is absent (e.g. fresh clones, CI).
Locally it re-verifies: schema, 3x3x3 balance, ID/verbatim-text disjointness from
all committed banks, and absence of regime label tokens.
"""

import hashlib
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BANK_PATH = ROOT / "results" / "reasoning_agentic" / "FINAL_BANK_V2.json"


def norm(t):
    return re.sub(r"\s+", " ", t.strip().lower())


@unittest.skipUnless(BANK_PATH.exists(), "local-only FINAL_BANK_V2.json absent")
class TestFinalBankV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = json.loads(BANK_PATH.read_text())
        cls.items = cls.bank["warnings"]

    def test_schema_and_balance(self):
        self.assertEqual(len(self.items), 27)
        from collections import Counter
        c = Counter((i["regime"], i["ambiguity_level"]) for i in self.items)
        self.assertEqual(len(c), 9)
        self.assertTrue(all(v == 3 for v in c.values()))
        for i in self.items:
            self.assertIn(i["regime"], ("supplier_delay", "demand_surge", "normal"))
            self.assertIn(i["ambiguity_level"], ("clear", "moderate", "vague"))
            self.assertTrue(i["template_id"].startswith("fx_"))
            self.assertNotIn("supplier_delay", i["text"])
            self.assertNotIn("demand_surge", i["text"])

    def test_disjoint_from_committed_banks(self):
        from src.events import REGIME_WARNING_TEMPLATES
        from src.confirmation_templates import CONFIRMATION_TEMPLATES
        from src.capacity_drop_templates import ALL_CAPACITY_DROP_TEMPLATES
        eids, etexts = set(), set()
        for t in REGIME_WARNING_TEMPLATES + CONFIRMATION_TEMPLATES + ALL_CAPACITY_DROP_TEMPLATES:
            eids.add(t["template_id"])
            etexts.add(norm(t["text"]))
        mine_ids = {i["template_id"] for i in self.items}
        self.assertFalse(mine_ids & eids)
        for i in self.items:
            self.assertNotIn(norm(i["text"]), etexts)


if __name__ == "__main__":
    unittest.main()
