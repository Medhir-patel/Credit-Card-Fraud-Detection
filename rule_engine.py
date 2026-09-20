"""
Hybrid Rule Engine & Business Logic Evaluator
===============================================
Combines machine learning probability outputs with deterministic business rules,
velocity anomaly detection, and custom analyst heuristics for hybrid decisioning.
"""

import math
from typing import List, Dict, Any, Optional


class HybridRuleEngine:
    def __init__(self):
        # Default built-in rules
        self.default_rules = [
            {
                "id": "RULE_001",
                "code": "HIGH_AMOUNT_SURGE",
                "name": "High Transaction Amount Surge",
                "description": "Flags transactions where scaled amount > 3.5 standard deviations above normal.",
                "feature_index": 29,  # Amount is at index 29
                "operator": ">",
                "threshold": 3.5,
                "score_impact": 0.25,
                "action": "BOOST_RISK",
                "enabled": True
            },
            {
                "id": "RULE_002",
                "code": "SEVERE_V14_ANOMALY",
                "name": "Severe V14 Vector Anomaly Signature",
                "description": "Detects classic PCA V14 anomaly pattern common in compromised cards (V14 < -4.5).",
                "feature_index": 14,  # V14 is at index 14
                "operator": "<",
                "threshold": -4.5,
                "score_impact": 0.35,
                "action": "BOOST_RISK",
                "enabled": True
            },
            {
                "id": "RULE_003",
                "code": "V12_V14_COMBO_FRAUD",
                "name": "Dual V12 & V14 Anomaly Pattern",
                "description": "Correlates negative V12 and V14 vectors characteristic of organized fraud rings.",
                "feature_index": 12,  # V12 is at index 12
                "operator": "<",
                "threshold": -3.5,
                "score_impact": 0.20,
                "action": "BOOST_RISK",
                "enabled": True
            },
            {
                "id": "RULE_004",
                "code": "HIGH_PROBABILITY_HARD_BLOCK",
                "name": "Extreme ML Probability Hard Block",
                "description": "Triggers immediate card block if ML raw probability >= 90%.",
                "feature_index": -1,  # Uses ML prob directly
                "operator": ">=",
                "threshold": 0.90,
                "score_impact": 0.50,
                "action": "HARD_BLOCK",
                "enabled": True
            }
        ]

    def evaluate(
        self,
        features: List[float],
        ml_probability: float,
        custom_rules: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates a transaction vector and ML probability against active rules.

        Returns:
            Dict containing:
              - triggered_rules: list of triggered rule objects
              - rule_score_boost: float addition to ML probability
              - hybrid_probability: clamped composite risk probability
              - override_action: HARD_BLOCK, CHALLENGE_2FA, or None
              - final_decision: APPROVE, REVIEW, DECLINE
        """
        rules_to_eval = custom_rules if custom_rules is not None else self.default_rules
        triggered = []
        total_score_boost = 0.0
        override_action = None

        for r in rules_to_eval:
            if not r.get("enabled", True):
                continue

            feature_idx = r.get("feature_index", -1)
            op = r.get("operator", ">")
            thresh = r.get("threshold", 0.0)

            # Determine value to test
            if feature_idx == -1:
                val = ml_probability
            elif 0 <= feature_idx < len(features):
                val = float(features[feature_idx])
            else:
                continue

            # Evaluate operator condition
            is_triggered = False
            if op == ">" and val > thresh:
                is_triggered = True
            elif op == ">=" and val >= thresh:
                is_triggered = True
            elif op == "<" and val < thresh:
                is_triggered = True
            elif op == "<=" and val <= thresh:
                is_triggered = True
            elif op == "==" and math.isclose(val, thresh, abs_tol=1e-5):
                is_triggered = True

            if is_triggered:
                triggered.append({
                    "code": r.get("code", "CUSTOM_RULE"),
                    "name": r.get("name", "Custom Heuristic"),
                    "score_impact": r.get("score_impact", 0.1),
                    "action": r.get("action", "BOOST_RISK"),
                    "tested_val": round(val, 4),
                    "threshold": thresh
                })
                total_score_boost += r.get("score_impact", 0.0)
                if r.get("action") == "HARD_BLOCK":
                    override_action = "HARD_BLOCK"
                elif r.get("action") == "CHALLENGE_2FA" and override_action != "HARD_BLOCK":
                    override_action = "CHALLENGE_2FA"

        # Calculate hybrid composite probability
        hybrid_prob = min(1.0, max(0.0, ml_probability + total_score_boost))

        # Final decision logic
        if override_action == "HARD_BLOCK" or hybrid_prob >= 0.70:
            final_decision = "DECLINE"
        elif override_action == "CHALLENGE_2FA" or hybrid_prob >= 0.35:
            final_decision = "REVIEW"
        else:
            final_decision = "APPROVE"

        return {
            "triggered_rules": triggered,
            "triggered_count": len(triggered),
            "rule_score_boost": round(total_score_boost, 4),
            "hybrid_probability": round(hybrid_prob, 4),
            "override_action": override_action,
            "final_decision": final_decision
        }


# Global singleton instance
hybrid_engine = HybridRuleEngine()
