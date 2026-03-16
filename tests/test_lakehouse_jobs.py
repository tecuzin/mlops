from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LakehouseJobsTests(unittest.TestCase):
    def test_medallion_jobs_exist(self) -> None:
        jobs_dir = ROOT / "lakehouse" / "jobs"
        self.assertTrue((jobs_dir / "bronze_ingest.py").exists())
        self.assertTrue((jobs_dir / "silver_transform.py").exists())
        self.assertTrue((jobs_dir / "gold_materialize.py").exists())

    def test_bronze_job_tracks_ingestion_metadata(self) -> None:
        bronze_job = (ROOT / "lakehouse" / "jobs" / "bronze_ingest.py").read_text()
        self.assertIn("ingestion_ts", bronze_job)
        self.assertIn("source_id", bronze_job)
        self.assertIn("batch_id", bronze_job)

    def test_contracts_and_quality_files_exist(self) -> None:
        contracts = ROOT / "lakehouse" / "contracts"
        checks = ROOT / "lakehouse" / "quality"
        self.assertTrue((contracts / "bronze_schema.json").exists())
        self.assertTrue((contracts / "source_to_bronze_mapping.json").exists())
        self.assertTrue((checks / "silver_quality.py").exists())

    def test_gold_writes_snapshot_metadata(self) -> None:
        gold_job = (ROOT / "lakehouse" / "jobs" / "gold_materialize.py").read_text()
        self.assertIn("snapshot", gold_job.lower())
        self.assertIn("metadata", gold_job.lower())


if __name__ == "__main__":
    unittest.main()
