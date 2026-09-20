"""
Unit & Integration Tests for Hybrid Rule Engine & Rule Database CRUD
"""

import unittest
import os
import tempfile
import sqlite3
from rule_engine import hybrid_engine, HybridRuleEngine
from database import (
    init_db, get_fraud_rules, add_fraud_rule,
    toggle_fraud_rule, delete_fraud_rule
)

class TestHybridRuleEngine(unittest.TestCase):
    def setUp(self):
        self.engine = HybridRuleEngine()
        # Create standard 30-feature vector (zeros)
        self.sample_features = [0.0] * 30

    def test_default_rules_evaluation_no_trigger(self):
        result = self.engine.evaluate(self.sample_features, ml_probability=0.10)
        self.assertEqual(result["triggered_count"], 0)
        self.assertEqual(result["rule_score_boost"], 0.0)
        self.assertEqual(result["hybrid_probability"], 0.10)
        self.assertEqual(result["final_decision"], "APPROVE")

    def test_v14_anomaly_rule_trigger(self):
        features = list(self.sample_features)
        features[14] = -5.2  # Trigger SEVERE_V14_ANOMALY (threshold < -4.5)
        result = self.engine.evaluate(features, ml_probability=0.20)
        self.assertGreater(result["triggered_count"], 0)
        self.assertIn("SEVERE_V14_ANOMALY", [r["code"] for r in result["triggered_rules"]])
        self.assertGreater(result["hybrid_probability"], 0.20)

    def test_hard_block_override(self):
        features = list(self.sample_features)
        result = self.engine.evaluate(features, ml_probability=0.95)
        self.assertEqual(result["override_action"], "HARD_BLOCK")
        self.assertEqual(result["final_decision"], "DECLINE")

    def test_custom_rule_evaluation(self):
        custom_rule = [{
            "id": "TEST_01",
            "code": "HIGH_V1",
            "name": "High V1 Anomaly",
            "feature_index": 1,
            "operator": ">",
            "threshold": 2.0,
            "score_impact": 0.40,
            "action": "BOOST_RISK",
            "enabled": True
        }]
        features = list(self.sample_features)
        features[1] = 2.8
        result = self.engine.evaluate(features, ml_probability=0.15, custom_rules=custom_rule)
        self.assertEqual(result["triggered_count"], 1)
        self.assertEqual(result["hybrid_probability"], 0.55)


class TestRuleDatabase(unittest.TestCase):
    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")

    def tearDown(self):
        os.close(self.db_fd)
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_init_db_seeds_rules(self):
        init_db(self.db_path)
        rules = get_fraud_rules(db_path=self.db_path)
        self.assertGreater(len(rules), 0)

    def test_add_and_delete_custom_rule(self):
        init_db(self.db_path)
        rule_data = {
            "code": "TEST_CUSTOM_RULE",
            "name": "Custom Test Rule",
            "description": "Test description",
            "feature_index": 0,
            "operator": ">",
            "threshold": 5.0,
            "score_impact": 0.15,
            "action": "BOOST_RISK",
            "enabled": True
        }
        added = add_fraud_rule(rule_data, db_path=self.db_path)
        self.assertIn("id", added)

        rules = get_fraud_rules(db_path=self.db_path)
        codes = [r["code"] for r in rules]
        self.assertIn("TEST_CUSTOM_RULE", codes)

        # Toggle rule
        toggled = toggle_fraud_rule(added["id"], False, db_path=self.db_path)
        self.assertTrue(toggled)

        # Delete rule
        deleted = delete_fraud_rule(added["id"], db_path=self.db_path)
        self.assertTrue(deleted)


if __name__ == "__main__":
    unittest.main()
