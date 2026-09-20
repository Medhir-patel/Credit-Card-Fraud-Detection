"""
Risk Scoring & Decision Engine for Fraud Detection
===================================================
Classifies transaction fraud probabilities into tiered risk categories,
recommends operational mitigations, and generates decision payloads.
"""

from typing import Dict, Any


RISK_THRESHOLDS = {
    "LOW_MAX": 0.15,
    "MODERATE_MAX": 0.50,
    "HIGH_MAX": 0.80,
}


def assess_risk(probability: float, prediction: int = 0, optimal_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Compute fine-grained risk tier, severity score, and automated mitigation steps,
    dynamically calibrated against the model's optimal decision threshold.
    
    Args:
        probability: Float between 0.0 and 1.0 (fraud probability from ML model)
        prediction: 0 (legitimate) or 1 (fraud)
        optimal_threshold: Calibrated decision boundary for positive fraud class.
        
    Returns:
        Dictionary containing risk classification, action advice, and UI metadata.
    """
    prob = max(0.0, min(1.0, float(probability)))
    opt_t = max(0.05, min(0.95, float(optimal_threshold)))

    low_cutoff = round(opt_t * 0.4, 4)
    mod_cutoff = round(opt_t, 4)
    high_cutoff = round(opt_t + (1.0 - opt_t) * 0.5, 4)

    if prob < low_cutoff:
        tier = "LOW"
        risk_score = round(prob * 100, 2)
        label = "Low Risk"
        action = "Automated Approval"
        action_detail = "Transaction matches standard customer behavioral profile. No friction needed."
        color = "#00d4aa"
        badge_class = "risk-low"
        requires_review = False
        decision = "APPROVE"

    elif prob < mod_cutoff:
        tier = "MODERATE"
        risk_score = round(prob * 100, 2)
        label = "Moderate Risk"
        action = "Step-Up Authentication"
        action_detail = "Slight deviation from normal spending pattern. Trigger 3D-Secure 2FA / OTP verification."
        color = "#ffa502"
        badge_class = "risk-moderate"
        requires_review = False
        decision = "CHALLENGE"

    elif prob < high_cutoff:
        tier = "HIGH"
        risk_score = round(prob * 100, 2)
        label = "High Risk"
        action = "Route to Fraud Analyst Queue"
        action_detail = "High anomaly indicators present. Temporarily hold settlement for manual review."
        color = "#ff793f"
        badge_class = "risk-high"
        requires_review = True
        decision = "REVIEW"

    else:
        tier = "CRITICAL"
        risk_score = round(prob * 100, 2)
        label = "Critical Risk"
        action = "Block & Alert Cardholder"
        action_detail = "Extreme fraud likelihood detected. Block transaction immediately and issue security notification."
        color = "#ff4757"
        badge_class = "risk-critical"
        requires_review = True
        decision = "DECLINE"

    return {
        "tier": tier,
        "label": label,
        "score": risk_score,
        "decision": decision,
        "action": action,
        "action_detail": action_detail,
        "color": color,
        "badge_class": badge_class,
        "requires_review": requires_review,
        "optimal_threshold": opt_t,
    }


def extract_shap_risk_drivers(explainer: Any, scaled_vector: list, feature_names: list, top_k: int = 4) -> list:
    """
    Extract exact SHAP (SHapley Additive exPlanations) feature attributions for explainable AI.
    
    Args:
        explainer: Trained SHAP Explainer instance (or None).
        scaled_vector: 1D array/list of scaled feature values.
        feature_names: List of feature names.
        top_k: Number of top risk drivers to return.
        
    Returns:
        List of dicts with feature name, SHAP value, direction, magnitude, and impact.
    """
    if explainer is None or not scaled_vector or not feature_names or len(scaled_vector) != len(feature_names):
        return extract_top_risk_drivers(scaled_vector, feature_names, top_k)

    try:
        import numpy as np
        X = np.array(scaled_vector, dtype=float).reshape(1, -1)
        sv = explainer.shap_values(X)

        if isinstance(sv, list):
            vals = sv[1][0] if len(sv) > 1 else sv[0][0]
        elif hasattr(sv, "values"):
            vals = sv.values[0]
            if np.ndim(vals) > 1:
                vals = vals[:, 1]
        else:
            vals = sv[0]
            if np.ndim(vals) > 1:
                vals = vals[:, 1]

        vals = np.array(vals, dtype=float).flatten()

        drivers = []
        for feat_name, raw_val, shap_val in zip(feature_names, scaled_vector, vals):
            abs_shap = abs(float(shap_val))
            direction = "Increases Fraud Risk" if shap_val > 0 else "Lowers Fraud Risk"
            impact = "CRITICAL" if abs_shap > 0.5 else ("HIGH" if abs_shap > 0.2 else "MODERATE")
            drivers.append({
                "feature": feat_name,
                "shap_value": round(float(shap_val), 4),
                "magnitude": round(abs_shap, 4),
                "raw_scaled": round(float(raw_val), 3),
                "direction": direction,
                "impact": impact,
                "explanation": f"Feature {feat_name} ({direction.lower()}) with SHAP impact score of {shap_val:+.4f}"
            })

        drivers.sort(key=lambda x: x["magnitude"], reverse=True)
        return drivers[:top_k]
    except Exception:
        return extract_top_risk_drivers(scaled_vector, feature_names, top_k)


def extract_top_risk_drivers(scaled_vector: list, feature_names: list, top_k: int = 4) -> list:
    """
    Fallback heuristic: Extract the most anomalous feature deviations for explainability.
    
    Args:
        scaled_vector: 1D array/list of scaled feature values for a single transaction.
        feature_names: List of feature names matching the vector.
        top_k: Number of top anomalous features to return.
        
    Returns:
        List of dicts with feature name, anomaly deviation score, and explanation.
    """
    if not scaled_vector or not feature_names or len(scaled_vector) != len(feature_names):
        return []

    deviations = []
    for feat_name, val in zip(feature_names, scaled_vector):
        abs_val = abs(float(val))
        deviations.append({
            "feature": feat_name,
            "magnitude": round(abs_val, 3),
            "raw_scaled": round(float(val), 3),
            "direction": "elevated" if val > 0 else "suppressed",
            "impact": "CRITICAL" if abs_val > 3.0 else ("HIGH" if abs_val > 1.5 else "MODERATE")
        })

    # Sort by absolute deviation magnitude descending
    deviations.sort(key=lambda x: x["magnitude"], reverse=True)
    return deviations[:top_k]

