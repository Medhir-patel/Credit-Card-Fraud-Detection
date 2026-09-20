"""
Unit tests for the Risk Scoring and Decision Engine.
"""

import unittest
from risk_engine import assess_risk, RISK_THRESHOLDS


class TestRiskEngine(unittest.TestCase):

    def test_low_risk_boundary(self):
        result = assess_risk(0.05)
        self.assertEqual(result["tier"], "LOW")
        self.assertEqual(result["decision"], "APPROVE")
        self.assertFalse(result["requires_review"])
        self.assertEqual(result["score"], 5.0)

    def test_moderate_risk_boundary(self):
        result = assess_risk(0.35)
        self.assertEqual(result["tier"], "MODERATE")
        self.assertEqual(result["decision"], "CHALLENGE")
        self.assertFalse(result["requires_review"])
        self.assertEqual(result["score"], 35.0)

    def test_high_risk_boundary(self):
        result = assess_risk(0.65)
        self.assertEqual(result["tier"], "HIGH")
        self.assertEqual(result["decision"], "REVIEW")
        self.assertTrue(result["requires_review"])
        self.assertEqual(result["score"], 65.0)

    def test_critical_risk_boundary(self):
        result = assess_risk(0.95)
        self.assertEqual(result["tier"], "CRITICAL")
        self.assertEqual(result["decision"], "DECLINE")
        self.assertTrue(result["requires_review"])
        self.assertEqual(result["score"], 95.0)

    def test_clamping_negative_and_overflow(self):
        res_neg = assess_risk(-0.2)
        self.assertEqual(res_neg["score"], 0.0)
        self.assertEqual(res_neg["tier"], "LOW")

        res_over = assess_risk(1.5)
        self.assertEqual(res_over["score"], 100.0)
    def test_extract_top_risk_drivers(self):
        from risk_engine import extract_top_risk_drivers
        scaled_vec = [0.1, -4.5, 2.1, 0.05, -1.8]
        feat_names = ["V1", "V2", "V3", "V4", "V5"]

        drivers = extract_top_risk_drivers(scaled_vec, feat_names, top_k=3)
        self.assertEqual(len(drivers), 3)
        self.assertEqual(drivers[0]["feature"], "V2")
        self.assertEqual(drivers[0]["magnitude"], 4.5)
        self.assertEqual(drivers[0]["impact"], "CRITICAL")
        self.assertEqual(drivers[1]["feature"], "V3")
        self.assertEqual(drivers[2]["feature"], "V5")

    def test_extract_top_risk_drivers_empty(self):
        from risk_engine import extract_top_risk_drivers
        self.assertEqual(extract_top_risk_drivers([], []), [])
        self.assertEqual(extract_top_risk_drivers([1.0], []), [])

    def test_calibrated_threshold_assess_risk(self):
        # Low threshold (0.2)
        res_low_t = assess_risk(0.18, optimal_threshold=0.20)
        self.assertEqual(res_low_t["tier"], "MODERATE")
        self.assertEqual(res_low_t["decision"], "CHALLENGE")

        # High threshold (0.7)
        res_high_t = assess_risk(0.15, optimal_threshold=0.70)
        self.assertEqual(res_high_t["tier"], "LOW")
        self.assertEqual(res_high_t["decision"], "APPROVE")

    def test_extract_shap_risk_drivers_fallback(self):
        from risk_engine import extract_shap_risk_drivers
        scaled_vec = [0.1, -4.5, 2.1, 0.05, -1.8]
        feat_names = ["V1", "V2", "V3", "V4", "V5"]

        # When explainer is None, fall back cleanly
        drivers = extract_shap_risk_drivers(None, scaled_vec, feat_names, top_k=3)
        self.assertEqual(len(drivers), 3)
        self.assertEqual(drivers[0]["feature"], "V2")


if __name__ == "__main__":
    unittest.main()

