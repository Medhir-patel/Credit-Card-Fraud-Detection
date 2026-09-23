"""
Transaction Simulator for FraudGuard AI
========================================
Converts human-readable transaction inputs (amount, merchant, time, location)
into the 30-feature vector required by the trained ML model (V1-V28, log_Amount, Hour).

Since V1-V28 are PCA-transformed and we cannot reverse PCA without the original
projection matrix, we use archetype blending:
  - Two archetypes: a verified legitimate transaction and a verified fraud transaction
  - A 'fraud_weight' [0.0–1.0] is computed from the combination of risk factors
  - The archetypes are blended by that weight and Gaussian noise is added for variation

This produces realistic, model-compatible feature vectors from ordinary user inputs.
"""

import numpy as np
from typing import Dict, Any, Tuple, List

# ── Archetype Feature Vectors (RAW / UNSCALED PCA space) ─────────────────────
# Row 0 from creditcard.csv — confirmed legitimate, no fraud history
_LEGIT_V = [
    -1.3598, -0.0728,  2.5363,  1.3782, -0.3383,  0.4624,  0.2396,  0.0987,
     0.3638,  0.0908, -0.5516, -0.6178, -0.9914, -0.3112,  1.4682, -0.4704,
     0.2080,  0.0258,  0.4040,  0.2514, -0.0183,  0.2778, -0.1105,  0.0669,
     0.1285, -0.1891,  0.1336, -0.0211,
]

# Row 6331 from creditcard.csv — high-confidence fraud (Class=1)
_FRAUD_V = [
     0.0084,  4.1378, -6.2407,  6.6757,  0.7683, -3.3531, -1.6317,  0.1546,
    -2.7959, -6.1879,  5.6644, -9.8545, -0.3062, -10.6912, -0.6385, -2.0420,
    -1.1291,  0.1165, -1.9347,  0.4884,  0.3645, -0.6081, -0.5395,  0.1289,
     1.4885,  0.5080,  0.7358,  0.5136,
]

LEGIT_V = np.array(_LEGIT_V, dtype=float)
FRAUD_V = np.array(_FRAUD_V, dtype=float)

# ── Category Amount Thresholds (USD) — define "normal" spend per category ────
CATEGORY_THRESHOLDS: Dict[str, float] = {
    "grocery":       60.0,
    "electronics":  600.0,
    "travel":      1500.0,
    "dining":        80.0,
    "atm":          300.0,
    "online":       250.0,
    "entertainment": 200.0,
    "healthcare":   500.0,
}

CATEGORY_LABELS: Dict[str, str] = {
    "grocery":       "Grocery / Supermarket",
    "electronics":   "Electronics & Gadgets",
    "travel":        "Travel & Accommodation",
    "dining":        "Dining & Restaurants",
    "atm":           "ATM Withdrawal",
    "online":        "Online Shopping",
    "entertainment": "Entertainment",
    "healthcare":    "Healthcare & Pharmacy",
}

# Fraud risk contribution by geographic region (0 = no added risk)
LOCATION_RISK: Dict[str, float] = {
    "domestic":  0.00,
    "europe":    0.12,
    "asia":      0.10,
    "americas":  0.08,
    "africa":    0.18,
    "oceania":   0.06,
}

# Hours considered high-risk (midnight to ~5 AM)
HIGH_RISK_HOURS = set(range(0, 5))


# ── Risk Factor Computation ──────────────────────────────────────────────────

def compute_fraud_weight(
    amount: float,
    merchant_category: str,
    transaction_type: str,
    hour: int,
    location: str,
) -> Tuple[float, List[Dict[str, str]]]:
    """
    Compute a continuous fraud_weight in [0.0, 1.0] and a list of
    human-readable risk factor explanations that drove the score.

    Args:
        amount:             Transaction amount in USD
        merchant_category:  e.g. 'grocery', 'electronics', 'travel' …
        transaction_type:   e.g. 'pos', 'online', 'atm', 'contactless'
        hour:               Hour of day (0–23)
        location:           e.g. 'domestic', 'europe', 'asia' …

    Returns:
        (fraud_weight: float, risk_factors: List[dict])
    """
    fw = 0.05   # small baseline — all transactions carry some inherent risk
    factors: List[Dict[str, str]] = []

    category  = merchant_category.lower().strip()
    tx_type   = transaction_type.lower().strip()
    loc       = location.lower().strip()
    threshold = CATEGORY_THRESHOLDS.get(category, 300.0)
    cat_label = CATEGORY_LABELS.get(category, category.replace("_", " ").title())

    # ── 1. Amount anomaly ────────────────────────────────────────────────────
    ratio = amount / threshold if threshold > 0 else 1.0
    if ratio > 5.0:
        fw += 0.55
        factors.append({
            "icon": "💰", "title": "Extreme Amount Anomaly",
            "detail": (
                f"${amount:,.2f} is {ratio:.1f}× above the typical limit "
                f"(${threshold:,.0f}) for {cat_label} transactions."
            ),
        })
    elif ratio > 3.0:
        fw += 0.35
        factors.append({
            "icon": "💸", "title": "Unusually High Amount",
            "detail": (
                f"${amount:,.2f} is {ratio:.1f}× above normal range for "
                f"{cat_label} (typical max: ${threshold:,.0f})."
            ),
        })
    elif ratio > 1.5:
        fw += 0.15
        factors.append({
            "icon": "⚠️", "title": "Above-Average Transaction Amount",
            "detail": (
                f"${amount:,.2f} is moderately elevated for {cat_label}. "
                f"Typical range: under ${threshold:,.0f}."
            ),
        })

    # ── 2. Unusual hour ──────────────────────────────────────────────────────
    if hour in HIGH_RISK_HOURS:
        if tx_type == "online":
            fw += 0.30
            factors.append({
                "icon": "🌙", "title": "Late-Night Online Transaction",
                "detail": (
                    f"Online purchase at {hour:02d}:00 AM — the 00:00–05:00 window "
                    f"accounts for a disproportionate share of card-not-present fraud."
                ),
            })
        elif tx_type == "atm":
            fw += 0.22
            factors.append({
                "icon": "🏧", "title": "Nocturnal ATM Withdrawal",
                "detail": (
                    f"ATM withdrawal at {hour:02d}:00 AM is uncommon and statistically "
                    f"elevated for cash-out fraud."
                ),
            })
        else:
            fw += 0.10
            factors.append({
                "icon": "🌙", "title": "Unusual Transaction Hour",
                "detail": (
                    f"Transaction at {hour:02d}:00 AM falls outside normal "
                    f"cardholder activity windows."
                ),
            })

    # ── 3. International location ────────────────────────────────────────────
    loc_risk = LOCATION_RISK.get(loc, 0.0)
    if loc_risk > 0:
        fw += loc_risk
        region = loc.replace("_", " ").title()
        factors.append({
            "icon": "🌍", "title": "International Transaction",
            "detail": (
                f"Transaction origin: {region}. Cross-border activity raises "
                f"fraud probability — especially without prior international history."
            ),
        })

    # ── 4. CNP (online) + International combo ────────────────────────────────
    if tx_type == "online" and loc_risk > 0:
        fw += 0.15
        factors.append({
            "icon": "🔗", "title": "International Card-Not-Present (CNP)",
            "detail": (
                "Combination of an online (CNP) transaction from an international "
                "origin is a well-known high-fraud pattern in financial crime."
            ),
        })

    # ── 5. High ATM withdrawal ───────────────────────────────────────────────
    if tx_type == "atm" and amount > 400:
        fw += 0.20
        factors.append({
            "icon": "💳", "title": "Large ATM Withdrawal",
            "detail": (
                f"${amount:,.2f} ATM withdrawal significantly exceeds typical "
                f"daily cash access limits (normal: <$300)."
            ),
        })

    # ── 6. High-value electronics + online + night triple-factor ─────────────
    if category == "electronics" and tx_type == "online" and hour in HIGH_RISK_HOURS:
        fw += 0.20
        factors.append({
            "icon": "🖥️", "title": "High-Value Online Electronics at Night",
            "detail": (
                "Purchasing electronics online during late-night hours is a known "
                "carding/cash-out fraud pattern detected in financial crime databases."
            ),
        })

    fw = float(np.clip(fw, 0.03, 0.97))
    return fw, factors


# ── Feature Vector Construction ──────────────────────────────────────────────

def build_feature_vector(
    amount: float,
    merchant_category: str,
    transaction_type: str,
    hour: int,
    location: str,
    seed: int = None,
) -> Tuple[List[float], float, List[Dict]]:
    """
    Build a 30-element feature vector [V1…V28, log_Amount, Hour] for the ML model.

    The V1–V28 features are produced by blending the two archetype vectors
    according to fraud_weight, then adding calibrated Gaussian noise.
    log_Amount and Hour are computed directly from the inputs.

    Returns:
        features:     List[float], shape (30,) — pass directly to /predict
        fraud_weight: float  — indicates how fraud-like the inputs are
        risk_factors: List[dict] — human-readable triggers
    """
    if seed is not None:
        np.random.seed(seed)

    fraud_weight, risk_factors = compute_fraud_weight(
        amount, merchant_category, transaction_type, hour, location
    )

    # Blend archetype V-feature vectors
    v_blended = fraud_weight * FRAUD_V + (1.0 - fraud_weight) * LEGIT_V

    # Noise is larger in the uncertain zone (fw ≈ 0.5) and smaller at extremes
    uncertainty = 1.0 - abs(fraud_weight - 0.5) * 2.0
    noise_scale = 0.35 + uncertainty * 0.55
    noise = np.random.normal(0.0, noise_scale, size=28)
    v_noisy = (v_blended + noise).tolist()

    # Append log-transformed amount and raw hour (model feature space)
    log_amount = float(np.log1p(amount))
    features = v_noisy + [log_amount, float(hour)]

    return features, fraud_weight, risk_factors


def simulate_transaction(tx_data: Dict[str, Any]) -> Tuple[List[float], float, List[Dict]]:
    """
    Main public API. Converts a human-readable transaction dict into a
    model-compatible 30-element feature vector.

    Expected keys:
        amount              float   Transaction amount in USD
        merchant_category   str     One of: grocery, electronics, travel,
                                    dining, atm, online, entertainment, healthcare
        transaction_type    str     One of: pos, online, atm, contactless
        hour                int     Hour of day (0–23)
        location            str     One of: domestic, europe, asia, americas,
                                    africa, oceania
        seed                int?    Optional random seed for reproducibility

    Returns:
        (features: List[float], fraud_weight: float, risk_factors: List[dict])
    """
    amount   = float(tx_data.get("amount", 50.0))
    category = str(tx_data.get("merchant_category", "online")).lower()
    tx_type  = str(tx_data.get("transaction_type", "pos")).lower()
    hour     = int(tx_data.get("hour", 12))
    location = str(tx_data.get("location", "domestic")).lower()
    seed     = tx_data.get("seed", None)

    return build_feature_vector(amount, category, tx_type, hour, location, seed=seed)
