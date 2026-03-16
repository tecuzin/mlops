from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LakehouseFoundationTests(unittest.TestCase):
    def test_compose_declares_lakehouse_services(self) -> None:
        compose = (ROOT / "docker-compose.yml").read_text()
        self.assertIn("\n  minio:\n", compose)
        self.assertIn("\n  nessie:\n", compose)
        self.assertIn("\n  spark:\n", compose)

    def test_env_example_has_lakehouse_variables(self) -> None:
        env_example = (ROOT / ".env.example").read_text()
        self.assertIn("LAKEHOUSE_ENABLED=true", env_example)
        self.assertIn("LAKEHOUSE_S3_ENDPOINT=http://minio:9000", env_example)
        self.assertIn("LAKEHOUSE_CATALOG_URI=http://nessie:19120/api/v1", env_example)
        self.assertIn("LAKEHOUSE_METADATA_DIR=/app/lakehouse/metadata", env_example)
        self.assertIn("LAKEHOUSE_CATALOG_NAME=nessie", env_example)
        self.assertIn("LAKEHOUSE_CATALOG_REF=main", env_example)

    def test_runbook_exists(self) -> None:
        runbook = ROOT / "lakehouse" / "RUNBOOK.md"
        self.assertTrue(runbook.exists(), "lakehouse runbook must exist")


if __name__ == "__main__":
    unittest.main()
