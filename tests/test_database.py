"""
Unit tests for Database Audit Log & Analyst Review Module.
"""

import unittest
import os
import tempfile
from database import (
    init_db, log_transaction, get_transactions,
    get_transaction_by_id, update_analyst_review
)


class TestDatabase(unittest.TestCase):

    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(self.db_fd)
        init_db(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_log_and_get_transaction(self):
        tx_id = "TX-TEST-001"
        features = [0.1, -1.2, 3.4]
        res = log_transaction(
            tx_id=tx_id,
            features=features,
            probability=0.85,
            prediction=1,
            risk_tier="CRITICAL",
            decision="DECLINE",
            db_path=self.db_path
        )

        self.assertEqual(res["id"], tx_id)
        self.assertEqual(res["status"], "PENDING_REVIEW")

        item = get_transaction_by_id(tx_id, db_path=self.db_path)
        self.assertIsNotNone(item)
        self.assertEqual(item["id"], tx_id)
        self.assertEqual(item["probability"], 0.85)
        self.assertEqual(item["risk_tier"], "CRITICAL")
        self.assertEqual(item["features"], features)

    def test_get_transactions_filter(self):
        log_transaction("TX-01", [0.0], 0.05, 0, "LOW", "APPROVE", db_path=self.db_path)
        log_transaction("TX-02", [1.0], 0.90, 1, "CRITICAL", "DECLINE", db_path=self.db_path)

        all_tx = get_transactions(db_path=self.db_path)
        self.assertEqual(len(all_tx), 2)

        pending = get_transactions(status="PENDING_REVIEW", db_path=self.db_path)
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["id"], "TX-02")

    def test_update_analyst_review(self):
        tx_id = "TX-REVIEW-100"
        log_transaction(tx_id, [0.5], 0.75, 1, "HIGH", "REVIEW", db_path=self.db_path)

        updated = update_analyst_review(
            tx_id=tx_id,
            analyst_decision="APPROVED",
            analyst_notes="Customer confirmed transaction over phone",
            db_path=self.db_path
        )
        self.assertTrue(updated)

        item = get_transaction_by_id(tx_id, db_path=self.db_path)
        self.assertEqual(item["analyst_decision"], "APPROVED")
        self.assertEqual(item["analyst_notes"], "Customer confirmed transaction over phone")
        self.assertEqual(item["status"], "RESOLVED_APPROVED")


if __name__ == "__main__":
    unittest.main()
