"""
Alert Engine — Fraud Notification & Case Report Generator
==========================================================
Generates:
  - Cardholder SMS / push notification text
  - Analyst queue case notes
  - Recommended action step-by-step workflow
  - Complete alert payload for the frontend

Used by the /simulate route in app.py after the model produces a prediction.
"""

import datetime
import random
from typing import Dict, Any, List


# ── Recommended Workflow Steps by Decision ─────────────────────────────────

_WORKFLOW_STEPS: Dict[str, List[str]] = {
    "DECLINE": [
        "Transaction automatically blocked by fraud engine",
        "Real-time cardholder SMS/push alert dispatched",
        "Case escalated to Fraud Operations queue (Priority: HIGH)",
        "Temporary card freeze applied (renewable every 24h)",
        "Chargeback dispute window opened for cardholder",
    ],
    "REVIEW": [
        "Transaction placed in settlement hold (funds reserved)",
        "Fraud analyst assigned — SLA: 2 business hours",
        "Cardholder notified: asked to confirm or deny activity",
        "Auto-release on cardholder confirmation via app",
        "Auto-block if no response within 4 hours",
    ],
    "CHALLENGE": [
        "3D-Secure OTP challenge sent to registered mobile number",
        "Transaction pending cardholder verification",
        "Auto-approve on successful OTP (expires in 5 minutes)",
        "Auto-decline after 3 failed OTP attempts",
    ],
    "APPROVE": [
        "Transaction processed and settled normally",
        "Behavioral baseline updated for this card profile",
        "Monitoring activated for follow-up micro-transactions",
        "No action required from cardholder",
    ],
}

# ── Cardholder Alert Generation ────────────────────────────────────────────

def generate_cardholder_sms(tx_data: Dict[str, Any], prediction: Dict[str, Any]) -> str:
    """Generate the SMS/push notification text that would be sent to the cardholder."""
    amount      = tx_data.get("amount", 0.0)
    merchant    = tx_data.get("merchant_name", "Unknown Merchant")
    card_last4  = tx_data.get("card_last4", "****")
    decision    = prediction.get("risk", {}).get("decision", "APPROVE")
    ref_code    = f"FG{random.randint(100000, 999999)}"

    templates = {
        "DECLINE": (
            f"[FraudGuard ALERT] We BLOCKED a suspicious charge on your card ending {card_last4}.\n"
            f"${amount:,.2f} at {merchant} was declined.\n"
            f"NOT YOU? Your card is safe. Contact us: 1-800-FRAUD-01\n"
            f"Ref: {ref_code}"
        ),
        "REVIEW": (
            f"[FraudGuard] Unusual activity on card ending {card_last4}.\n"
            f"${amount:,.2f} at {merchant} is under review.\n"
            f"Reply YES to approve or NO to block. Ref: {ref_code}"
        ),
        "CHALLENGE": (
            f"[FraudGuard] Verification required for card ending {card_last4}.\n"
            f"${amount:,.2f} at {merchant}.\n"
            f"Your one-time code: {random.randint(100000, 999999)} (expires in 5 min)\n"
            f"Do NOT share this code with anyone."
        ),
        "APPROVE": (
            f"[FraudGuard] Purchase approved on card ending {card_last4}.\n"
            f"${amount:,.2f} at {merchant}.\n"
            f"Not you? Call 1-800-FRAUD-01 immediately. Ref: {ref_code}"
        ),
    }
    return templates.get(decision, templates["APPROVE"])


# ── Analyst Case Note ──────────────────────────────────────────────────────

def generate_analyst_note(
    tx_data: Dict[str, Any],
    prediction: Dict[str, Any],
    risk_factors: List[Dict],
) -> str:
    """Generate a structured summary note for the fraud analyst operations queue."""
    tx_id     = tx_data.get("tx_id", "TX-XXXX")
    amount    = tx_data.get("amount", 0.0)
    merchant  = tx_data.get("merchant_name", "Unknown")
    category  = tx_data.get("merchant_category", "unknown").replace("_", " ").title()
    card_last4 = tx_data.get("card_last4", "****")
    prob      = prediction.get("probability", 0.0) * 100
    tier      = prediction.get("risk", {}).get("tier", "UNKNOWN")
    decision  = prediction.get("risk", {}).get("decision", "APPROVE")
    action    = prediction.get("risk", {}).get("action", "N/A")

    factor_lines = "\n".join(
        f"  [{f['icon']}] {f['title']}: {f['detail']}"
        for f in risk_factors[:4]
    ) or "  • No significant risk factors triggered"

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return (
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"CASE {tx_id} | {tier} RISK | {decision}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Timestamp  : {timestamp}\n"
        f"Card       : •••• •••• •••• {card_last4}\n"
        f"Amount     : ${amount:,.2f}\n"
        f"Merchant   : {merchant} ({category})\n"
        f"Fraud Prob : {prob:.2f}%\n"
        f"Decision   : {action}\n"
        f"\nRisk Signals Triggered:\n{factor_lines}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )


# ── Full Alert Payload ─────────────────────────────────────────────────────

def build_alert_payload(
    tx_data: Dict[str, Any],
    prediction: Dict[str, Any],
    risk_factors: List[Dict],
) -> Dict[str, Any]:
    """
    Build the complete alert payload sent back to the frontend after a simulation.

    Returns a dict containing:
        cardholder_sms:    str   — Notification text for the cardholder
        analyst_note:      str   — Case note for the fraud ops queue
        risk_factors:      list  — Human-readable risk triggers for the UI
        workflow_steps:    list  — Step-by-step recommended actions
        case_id:           str   — Unique case reference
        timestamp:         str   — ISO timestamp
    """
    decision   = prediction.get("risk", {}).get("decision", "APPROVE")
    tier       = prediction.get("risk", {}).get("tier", "LOW")
    tx_id      = tx_data.get("tx_id", f"TX-{random.randint(1000, 9999)}")

    return {
        "cardholder_sms":  generate_cardholder_sms(tx_data, prediction),
        "analyst_note":    generate_analyst_note(tx_data, prediction, risk_factors),
        "risk_factors":    risk_factors,
        "workflow_steps":  _WORKFLOW_STEPS.get(decision, _WORKFLOW_STEPS["APPROVE"]),
        "case_id":         f"FG-{tx_id}-{tier[:3]}",
        "timestamp":       datetime.datetime.now().isoformat(),
    }
