from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LakehouseGovernanceTests(unittest.TestCase):
    def test_guardrails_policy_exists(self) -> None:
        policy = ROOT / "lakehouse" / "governance" / "medallion_guardrails.md"
        self.assertTrue(policy.exists())
        content = policy.read_text()
        self.assertIn("Bronze", content)
        self.assertIn("Silver", content)
        self.assertIn("Gold", content)

    def test_retention_and_compaction_scripts_exist(self) -> None:
        scripts = ROOT / "lakehouse" / "scripts"
        self.assertTrue((scripts / "retention_cleanup.sh").exists())
        self.assertTrue((scripts / "compact_silver_gold.sh").exists())

    def test_rollout_and_reproducibility_assets_exist(self) -> None:
        self.assertTrue((ROOT / "lakehouse" / "ROLLOUT_CHECKLIST.md").exists())
        self.assertTrue((ROOT / "lakehouse" / "scripts" / "validate_reproducibility.sh").exists())


if __name__ == "__main__":
    unittest.main()
