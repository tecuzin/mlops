from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from api.schemas import RunCreateRequest


class LakehouseContractsTests(unittest.TestCase):
    def test_run_create_request_accepts_lakehouse_refs(self) -> None:
        payload = RunCreateRequest(
            model_name="mistral-7b-rag-qa",
            model_id="mistralai/Mistral-7B-v0.1",
            task_type="finetune",
            train_lakehouse_ref={
                "catalog": "nessie",
                "namespace": "gold",
                "table": "rag_qa_train_ready",
                "reference": "main",
                "snapshot_id": "snapshot-20260316",
            },
            eval_lakehouse_ref={
                "catalog": "nessie",
                "namespace": "gold",
                "table": "rag_qa_eval_ready",
                "reference": "main",
                "snapshot_id": "snapshot-20260316",
            },
        )
        self.assertEqual(payload.train_lakehouse_ref.snapshot_id, "snapshot-20260316")
        self.assertEqual(payload.eval_lakehouse_ref.reference, "main")

    def test_worker_lakehouse_ref_resolver(self) -> None:
        from workers.lakehouse_ref import resolve_lakehouse_dataset_path

        with tempfile.TemporaryDirectory() as tmp:
            metadata_dir = Path(tmp)
            metadata = {
                "table": "gold.rag_qa_train_ready",
                "snapshot_id": "snapshot-20260316",
                "training_export_path": "/tmp/train.jsonl",
                "evaluation_export_path": "/tmp/eval.jsonl",
            }
            (metadata_dir / "gold_rag_qa_train_ready.json").write_text(json.dumps(metadata))
            ref = {
                "namespace": "gold",
                "table": "rag_qa_train_ready",
                "snapshot_id": "snapshot-20260316",
            }

            train_path = resolve_lakehouse_dataset_path(ref, metadata_dir, role="train")
            eval_path = resolve_lakehouse_dataset_path(ref, metadata_dir, role="eval")
            self.assertEqual(train_path, "/tmp/train.jsonl")
            self.assertEqual(eval_path, "/tmp/eval.jsonl")


if __name__ == "__main__":
    unittest.main()
