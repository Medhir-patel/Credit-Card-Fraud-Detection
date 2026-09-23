"""
Credit Card Fraud Detection — Flask Web Application
====================================================
A premium web dashboard for real-time fraud prediction.

Usage:
    1. First train the model:  python train_model.py
    2. Then start the app:     python app.py
    3. Open: http://localhost:5000
"""

import os
import json
import time
import uuid
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify
from risk_engine import assess_risk, extract_top_risk_drivers, extract_shap_risk_drivers
from database import (
    init_db, log_transaction, get_transactions,
    get_transaction_by_id, update_analyst_review,
    get_fraud_rules, add_fraud_rule, toggle_fraud_rule, delete_fraud_rule
)
from rule_engine import hybrid_engine
from transaction_simulator import (
    simulate_transaction, build_feature_vector, CATEGORY_THRESHOLDS
)
from alert_engine import build_alert_payload

app = Flask(__name__)

# ── Load model artifacts & DB ──────────────────────────────────
MODEL = None
SCALER = None
METADATA = None
SHAP_EXPLAINER = None


def load_artifacts():
    global MODEL, SCALER, METADATA, SHAP_EXPLAINER
    init_db()
    model_path = os.path.join("models", "best_model.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")
    meta_path = os.path.join("models", "metadata.json")
    shap_path = os.path.join("models", "shap_explainer.pkl")

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        MODEL = joblib.load(model_path)
        SCALER = joblib.load(scaler_path)
        print("[OK] Model and scaler loaded successfully.")
    else:
        print("[!!] No trained model found. Run 'python train_model.py' first.")

    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            METADATA = json.load(f)
        print("[OK] Metadata loaded.")

    if os.path.exists(shap_path):
        try:
            SHAP_EXPLAINER = joblib.load(shap_path)
            print("[OK] SHAP explainer loaded.")
        except Exception as e:
            print(f"[!!] Could not load SHAP explainer: {e}")


# -- Routes -----------------------------------------------------
@app.route("/")
def index():
    model_loaded = MODEL is not None
    metadata = METADATA if METADATA else {}
    return render_template("index.html",
                           model_loaded=model_loaded,
                           metadata=metadata)


@app.route("/analyst")
def analyst_queue():
    """Fraud Analyst Review Queue Dashboard."""
    return render_template("analyst.html", metadata=METADATA if METADATA else {})


@app.route("/dashboard")
def dashboard_view():
    """Analytics Dashboard View."""
    return render_template("dashboard.html",
                           model_loaded=MODEL is not None,
                           metadata=METADATA if METADATA else {})


@app.route("/history")
def history_view():
    """Transaction Audit History View."""
    txs = get_transactions(limit=200)
    return render_template("history.html",
                           transactions=txs,
                           metadata=METADATA if METADATA else {})


@app.route("/simulate", methods=["GET", "POST"])
def simulate_interactive():
    """
    Interactive human-friendly transaction simulation route.
    Converts human inputs into PCA feature vectors, evaluates model & rules, and builds alert payload.
    """
    if request.method == "POST":
        try:
            tx_data = request.get_json() or {}
            features, fraud_weight, risk_factors = simulate_transaction(tx_data)

            if MODEL is not None and SCALER is not None:
                X = np.array(features, dtype=float).reshape(1, -1)
                X_scaled = SCALER.transform(X)
                probability = float(MODEL.predict_proba(X_scaled)[0][1])
                opt_thresh = float(METADATA.get("optimal_threshold", 0.5)) if METADATA else 0.5
                prediction = 1 if probability >= opt_thresh else 0
            else:
                probability = float(fraud_weight)
                prediction = 1 if fraud_weight > 0.5 else 0

            risk_data = assess_risk(probability, prediction)
            active_rules = get_fraud_rules(enabled_only=True)
            hybrid_eval = hybrid_engine.evaluate(features, probability, active_rules)

            pred_payload = {
                "prediction": prediction,
                "probability": probability,
                "risk": risk_data,
                "hybrid_eval": hybrid_eval
            }

            alert_payload = build_alert_payload(tx_data, pred_payload, risk_factors)

            tx_id = tx_data.get("tx_id", f"TX-SIM-{int(time.time()*1000):x}".upper())
            log_transaction(
                tx_id=tx_id,
                features=features,
                probability=probability,
                prediction=prediction,
                risk_tier=risk_data["tier"],
                decision=risk_data["decision"]
            )

            return jsonify({
                "tx_id": tx_id,
                "features": features,
                "probability": round(probability, 6),
                "prediction": prediction,
                "is_fraud": prediction == 1,
                "risk": risk_data,
                "hybrid_eval": hybrid_eval,
                "alert": alert_payload
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"message": "Use POST to submit transaction simulation parameters."})


@app.route("/api/analytics")
def api_analytics():
    """Return aggregated stats for the analytics dashboard."""
    history = get_transactions(limit=500)
    total = len(history)
    fraud_count = sum(1 for t in history if t.get("prediction") == 1 or t.get("risk_tier") in ("HIGH", "CRITICAL"))

    tier_counts = {"LOW": 0, "MODERATE": 0, "HIGH": 0, "CRITICAL": 0}
    for t in history:
        tier = t.get("risk_tier", "LOW")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    return jsonify({
        "total_transactions": total,
        "fraud_count": fraud_count,
        "fraud_rate_percent": round((fraud_count / total) * 100, 2) if total else 0.0,
        "risk_tier_breakdown": tier_counts
    })


_LIVE_MERCHANTS = {
    "grocery": ["Whole Foods", "Walmart", "Target", "Costco", "Kroger", "Aldi"],
    "electronics": ["Best Buy", "Apple Store", "Samsung", "Newegg", "B&H Photo"],
    "dining": ["Starbucks", "McDonald's", "Chipotle", "Olive Garden", "Sushi Palace"],
    "online": ["Amazon", "eBay", "Shopify Store", "Etsy", "Alibaba"],
    "travel": ["Marriott Hotels", "Delta Airlines", "Airbnb", "Booking.com", "Uber"],
    "atm": ["Chase ATM", "Wells Fargo ATM", "Bank of America ATM", "Citibank ATM"],
    "entertainment": ["Netflix", "Spotify", "AMC Theaters", "Steam", "PlayStation Store"],
    "healthcare": ["CVS Pharmacy", "Walgreens", "Kaiser Permanente", "LabCorp"],
}

_LIVE_NAMES = [
    "James Wilson", "Emma Johnson", "Liam Brown", "Olivia Davis", "Noah Garcia",
    "Ava Martinez", "Elijah Anderson", "Isabella Thomas", "Lucas Jackson", "Mia White",
    "Rajesh Kumar", "Priya Sharma", "Arun Patel", "Sunita Singh", "Vikram Mehta"
]

@app.route("/api/live-feed")
def api_live_feed():
    """Live stream feed polling endpoint for analytics dashboard."""
    import random
    category = random.choice(list(_LIVE_MERCHANTS.keys()))
    merchant = random.choice(_LIVE_MERCHANTS[category])
    name = random.choice(_LIVE_NAMES)
    tx_type = random.choice(["pos", "online", "atm", "contactless"])
    location = random.choice(["domestic", "europe", "asia", "americas"])

    base = CATEGORY_THRESHOLDS.get(category, 100.0)
    factor = random.choices([0.2, 0.5, 0.8, 1.2, 2.0, 4.0, 6.0], weights=[20, 30, 25, 12, 7, 4, 2])[0]
    amount = round(base * factor * random.uniform(0.8, 1.2), 2)
    hour = random.randint(0, 23)

    tx_data = {
        "amount": amount,
        "merchant_category": category,
        "transaction_type": tx_type,
        "hour": hour,
        "location": location,
        "merchant_name": merchant,
        "cardholder_name": name,
        "card_last4": f"{random.randint(1000, 9999)}"
    }

    features, fraud_weight, _ = simulate_transaction(tx_data)

    if MODEL is not None and SCALER is not None:
        X = np.array(features, dtype=float).reshape(1, -1)
        X_scaled = SCALER.transform(X)
        probability = float(MODEL.predict_proba(X_scaled)[0][1])
        prediction = 1 if probability >= float(METADATA.get("optimal_threshold", 0.5)) else 0
    else:
        probability = float(fraud_weight)
        prediction = 1 if fraud_weight > 0.5 else 0

    risk_data = assess_risk(probability, prediction)
    tx_id = f"TX-{random.randint(10000, 99999)}"

    return jsonify({
        "tx_id": tx_id,
        "cardholder_name": name,
        "card_last4": tx_data["card_last4"],
        "merchant_name": merchant,
        "merchant_category": category,
        "amount": amount,
        "transaction_type": tx_type,
        "hour": hour,
        "location": location,
        "prediction": prediction,
        "probability": round(probability, 4),
        "is_fraud": prediction == 1,
        "risk": risk_data,
        "timestamp": time.strftime("%H:%M:%S")
    })


@app.route("/api/health")
def api_health():
    """Service health check endpoint for monitoring and uptime probes."""
    return jsonify({
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "scaler_loaded": SCALER is not None,
        "metadata_loaded": METADATA is not None,
        "shap_explainer_loaded": SHAP_EXPLAINER is not None,
        "optimal_threshold": METADATA.get("optimal_threshold", 0.5) if METADATA else 0.5
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    if MODEL is None:
        return jsonify({"error": "Model not loaded. Train it first."}), 503

    try:
        data = request.get_json()
        features = data.get("features", [])
        custom_id = data.get("id")

        if len(features) != len(METADATA["feature_names"]):
            return jsonify({
                "error": f"Expected {len(METADATA['feature_names'])} features, "
                         f"got {len(features)}"
            }), 400

        X = np.array(features, dtype=float).reshape(1, -1)
        X_scaled = SCALER.transform(X)

        probability = float(MODEL.predict_proba(X_scaled)[0][1])
        opt_thresh = float(METADATA.get("optimal_threshold", 0.5)) if METADATA else 0.5
        prediction = 1 if probability >= opt_thresh else 0

        risk_data = assess_risk(probability, prediction, optimal_threshold=opt_thresh)
        risk_drivers = extract_shap_risk_drivers(SHAP_EXPLAINER, X_scaled[0].tolist(), METADATA["feature_names"], top_k=4)

        active_rules = get_fraud_rules(enabled_only=True)
        hybrid_eval = hybrid_engine.evaluate(features, probability, active_rules)

        tx_id = custom_id if custom_id else f"TX-{int(time.time() * 1000):x}".upper()
        db_record = log_transaction(
            tx_id=tx_id,
            features=features,
            probability=probability,
            prediction=prediction,
            risk_tier=risk_data["tier"],
            decision=risk_data["decision"]
        )

        return jsonify({
            "id": tx_id,
            "prediction": prediction,
            "probability": round(probability, 6),
            "is_fraud": prediction == 1,
            "optimal_threshold": opt_thresh,
            "confidence": round(max(probability, 1 - probability) * 100, 2),
            "label": "🚨 FRAUD DETECTED" if prediction == 1 else "✅ LEGITIMATE",
            "risk": risk_data,
            "risk_drivers": risk_drivers,
            "hybrid_eval": hybrid_eval,
            "db_status": db_record["status"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """Bulk prediction endpoint for scanning multiple transactions at once."""
    if MODEL is None or SCALER is None or METADATA is None:
        return jsonify({"error": "Model not loaded. Train it first."}), 503

    try:
        data = request.get_json()
        if not data or "transactions" not in data:
            return jsonify({"error": "Missing 'transactions' list in request body."}), 400

        raw_items = data["transactions"]
        if not isinstance(raw_items, list) or len(raw_items) == 0:
            return jsonify({"error": "'transactions' must be a non-empty list."}), 400

        expected_len = len(METADATA["feature_names"])
        matrix = []
        identifiers = []

        for idx, item in enumerate(raw_items):
            if isinstance(item, dict):
                tx_id = item.get("id", f"TX-BATCH-{int(time.time() * 1000):x}-{idx + 1:04d}".upper())
                feat = item.get("features", [])
            elif isinstance(item, list):
                tx_id = f"TX-BATCH-{int(time.time() * 1000):x}-{idx + 1:04d}".upper()
                feat = item
            else:
                return jsonify({"error": f"Invalid item format at index {idx}."}), 400

            if len(feat) != expected_len:
                return jsonify({
                    "error": f"Item {tx_id} (index {idx}) has {len(feat)} features, expected {expected_len}."
                }), 400

            matrix.append(feat)
            identifiers.append(tx_id)

        X = np.array(matrix, dtype=float)
        X_scaled = SCALER.transform(X)

        probabilities = MODEL.predict_proba(X_scaled)[:, 1]
        opt_thresh = float(METADATA.get("optimal_threshold", 0.5))

        results = []
        fraud_count = 0
        tier_counts = {"LOW": 0, "MODERATE": 0, "HIGH": 0, "CRITICAL": 0}

        for i in range(len(matrix)):
            prob = float(probabilities[i])
            pred = 1 if prob >= opt_thresh else 0
            is_fraud = (pred == 1)
            if is_fraud:
                fraud_count += 1

            risk_data = assess_risk(prob, pred, optimal_threshold=opt_thresh)
            tier_counts[risk_data["tier"]] = tier_counts.get(risk_data["tier"], 0) + 1

            db_rec = log_transaction(
                tx_id=identifiers[i],
                features=matrix[i],
                probability=prob,
                prediction=pred,
                risk_tier=risk_data["tier"],
                decision=risk_data["decision"]
            )

            results.append({
                "id": identifiers[i],
                "prediction": pred,
                "probability": round(prob, 6),
                "is_fraud": is_fraud,
                "confidence": round(max(prob, 1 - prob) * 100, 2),
                "label": "🚨 FRAUD DETECTED" if is_fraud else "✅ LEGITIMATE",
                "risk": risk_data,
                "db_status": db_rec["status"]
            })

        total = len(results)
        return jsonify({
            "summary": {
                "total_analyzed": total,
                "fraud_count": fraud_count,
                "legitimate_count": total - fraud_count,
                "fraud_rate_percent": round((fraud_count / total) * 100, 2),
                "optimal_threshold": opt_thresh,
                "risk_tier_breakdown": tier_counts
            },
            "results": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/transactions", methods=["GET"])
def api_transactions():
    """Retrieve logged transactions filtered by status or risk tier."""
    status = request.args.get("status")
    risk_tier = request.args.get("risk_tier")
    limit = request.args.get("limit", 100, type=int)

    items = get_transactions(status=status, risk_tier=risk_tier, limit=limit)
    return jsonify({
        "count": len(items),
        "transactions": items
    })


@app.route("/api/transactions/<tx_id>/review", methods=["POST"])
def api_review_transaction(tx_id):
    """Update analyst review decision for a flagged transaction."""
    data = request.get_json() or {}
    decision = data.get("decision")  # "APPROVED" or "CONFIRMED_FRAUD"
    notes = data.get("notes", "")

    if not decision or decision not in ("APPROVED", "CONFIRMED_FRAUD", "REJECTED"):
        return jsonify({"error": "Invalid or missing 'decision'. Must be 'APPROVED' or 'CONFIRMED_FRAUD'."}), 400

    try:
        success = update_analyst_review(tx_id, decision, notes)
        if not success:
            return jsonify({"error": f"Transaction '{tx_id}' not found."}), 404
        return jsonify({
            "status": "success",
            "message": f"Transaction '{tx_id}' resolved as {decision}.",
            "tx_id": tx_id,
            "analyst_decision": decision,
            "analyst_notes": notes
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rules", methods=["GET", "POST"])
def api_rules():
    """Get all fraud rules or create a new custom business logic rule."""
    if request.method == "GET":
        rules = get_fraud_rules()
        return jsonify({"count": len(rules), "rules": rules}), 200

    elif request.method == "POST":
        try:
            data = request.get_json() or {}
            required = ["code", "name", "feature_index", "operator", "threshold", "score_impact", "action"]
            for field in required:
                if field not in data:
                    return jsonify({"error": f"Missing required rule field '{field}'"}), 400

            new_rule = add_fraud_rule(data)
            return jsonify({"status": "success", "rule": new_rule}), 201
        except Exception as e:
            return jsonify({"error": str(e)}), 500


@app.route("/api/rules/<int:rule_id>", methods=["PATCH", "DELETE"])
def api_rule_detail(rule_id):
    """Toggle enabled status or delete a fraud rule."""
    if request.method == "PATCH":
        try:
            data = request.get_json() or {}
            enabled = data.get("enabled", True)
            success = toggle_fraud_rule(rule_id, enabled)
            if not success:
                return jsonify({"error": f"Rule ID '{rule_id}' not found."}), 404
            return jsonify({"status": "success", "rule_id": rule_id, "enabled": enabled}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    elif request.method == "DELETE":
        try:
            success = delete_fraud_rule(rule_id)
            if not success:
                return jsonify({"error": f"Rule ID '{rule_id}' not found."}), 404
            return jsonify({"status": "success", "message": f"Rule '{rule_id}' deleted."}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500


@app.route("/api/metadata")
def api_metadata():
    if METADATA is None:
        return jsonify({"error": "No metadata available"}), 404
    return jsonify(METADATA)


@app.route("/api/sample")
def api_sample():
    """Return a sample transaction for testing."""
    legit = [
        -1.3598, -0.0728, 2.5363, 1.3782, -0.3383, 0.4624, 0.2396, 0.0987, 0.3638,
        0.0908, -0.5516, -0.6178, -0.9914, -0.3112, 1.4682, -0.4704, 0.208, 0.0258,
        0.404, 0.2514, -0.0183, 0.2778, -0.1105, 0.0669, 0.1285, -0.1891, 0.1336, -0.0211,
        5.0148, 0,
    ]
    fraud = [
        0.0084, 4.1378, -6.2407, 6.6757, 0.7683, -3.3531, -1.6317, 0.1546, -2.7959,
        -6.1879, 5.6644, -9.8545, -0.3062, -10.6912, -0.6385, -2.042, -1.1291, 0.1165,
        -1.9347, 0.4884, 0.3645, -0.6081, -0.5395, 0.1289, 1.4885, 0.508, 0.7358, 0.5136,
        0.6931, 2,
    ]
    return jsonify({
        "legitimate": legit,
        "fraud": fraud,
        "feature_names": METADATA["feature_names"] if METADATA else []
    })


from flask import Flask, render_template, request, jsonify, Response
import io
import csv

@app.route("/docs")
def swagger_docs():
    """Interactive OpenAPI REST Documentation."""
    return render_template("swagger.html")


@app.route("/api/openapi.json")
def openapi_spec():
    """OpenAPI 3.0 specification for FraudGuard AI."""
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "FraudGuard AI REST API",
            "version": "2.0.0",
            "description": "Production-grade Credit Card Fraud Detection API with SHAP Explainability & Risk Engine"
        },
        "paths": {
            "/predict": {
                "post": {
                    "summary": "Single Transaction Evaluation",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string", "example": "TX-1001"},
                                        "features": {
                                            "type": "array",
                                            "items": {"type": "number"},
                                            "example": [-1.3598, -0.0728, 2.5363, 1.3782, -0.3383, 0.4624, 0.2396, 0.0987, 0.3638, 0.0908, -0.5516, -0.6178, -0.9914, -0.3112, 1.4682, -0.4704, 0.208, 0.0258, 0.404, 0.2514, -0.0183, 0.2778, -0.1105, 0.0669, 0.1285, -0.1891, 0.1336, -0.0211, 5.0148, 0]
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {"description": "Risk assessment payload with SHAP feature attributions."}
                    }
                }
            },
            "/predict/batch": {
                "post": {
                    "summary": "Bulk Batch Transaction Scanning",
                    "responses": {"200": {"description": "Batch summary breakdown and item list."}}
                }
            },
            "/predict/batch/export": {
                "post": {
                    "summary": "Export Batch Scan Results as CSV Report",
                    "responses": {"200": {"description": "Downloadable CSV audit report stream."}}
                }
            },
            "/api/transactions": {
                "get": {
                    "summary": "Get Audit Log Transactions",
                    "parameters": [
                        {"name": "status", "in": "query", "schema": {"type": "string"}},
                        {"name": "risk_tier", "in": "query", "schema": {"type": "string"}}
                    ],
                    "responses": {"200": {"description": "List of logged audit transactions."}}
                }
            },
            "/api/transactions/{tx_id}/review": {
                "post": {
                    "summary": "Submit Analyst Manual Review",
                    "parameters": [
                        {"name": "tx_id", "in": "path", "required": True, "schema": {"type": "string"}}
                    ],
                    "responses": {"200": {"description": "Review confirmation status."}}
                }
            },
            "/api/health": {
                "get": {
                    "summary": "Service Health Check Probe",
                    "responses": {"200": {"description": "Service health metrics."}}
                }
            }
        }
    }
    return jsonify(spec)


@app.route("/predict/batch/export", methods=["POST"])
def predict_batch_export():
    """Run bulk prediction and export results as a downloadable CSV report."""
    if MODEL is None or SCALER is None or METADATA is None:
        return jsonify({"error": "Model not loaded. Train it first."}), 503

    try:
        data = request.get_json()
        if not data or "transactions" not in data:
            return jsonify({"error": "Missing 'transactions' list in request body."}), 400

        raw_items = data["transactions"]
        if not isinstance(raw_items, list) or len(raw_items) == 0:
            return jsonify({"error": "'transactions' must be a non-empty list."}), 400

        expected_len = len(METADATA["feature_names"])
        matrix = []
        identifiers = []

        for idx, item in enumerate(raw_items):
            if isinstance(item, dict):
                tx_id = item.get("id", f"TX-EXP-{int(time.time() * 1000):x}-{idx + 1:04d}".upper())
                feat = item.get("features", [])
            elif isinstance(item, list):
                tx_id = f"TX-EXP-{int(time.time() * 1000):x}-{idx + 1:04d}".upper()
                feat = item
            else:
                return jsonify({"error": f"Invalid item format at index {idx}."}), 400

            if len(feat) != expected_len:
                return jsonify({
                    "error": f"Item {tx_id} has {len(feat)} features, expected {expected_len}."
                }), 400

            matrix.append(feat)
            identifiers.append(tx_id)

        X = np.array(matrix, dtype=float)
        X_scaled = SCALER.transform(X)

        probabilities = MODEL.predict_proba(X_scaled)[:, 1]
        opt_thresh = float(METADATA.get("optimal_threshold", 0.5))

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Transaction ID", "Fraud Probability (%)", "Prediction Label",
            "Risk Tier", "Automated Decision", "Action Advice", "Logged Status"
        ])

        for i in range(len(matrix)):
            prob = float(probabilities[i])
            pred = 1 if prob >= opt_thresh else 0
            risk_data = assess_risk(prob, pred, optimal_threshold=opt_thresh)

            db_rec = log_transaction(
                tx_id=identifiers[i],
                features=matrix[i],
                probability=prob,
                prediction=pred,
                risk_tier=risk_data["tier"],
                decision=risk_data["decision"]
            )

            writer.writerow([
                identifiers[i],
                f"{prob * 100:.4f}",
                "FRAUD DETECTED" if pred == 1 else "LEGITIMATE",
                risk_data["tier"],
                risk_data["decision"],
                risk_data["action"],
                db_rec["status"]
            ])

        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=fraud_scan_report.csv"}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/simulate", methods=["GET", "POST"])
def api_simulate():
    """Generate a realistic synthetic transaction in real time and evaluate fraud risk."""
    if MODEL is None or SCALER is None or METADATA is None:
        return jsonify({"error": "Model not loaded. Train it first."}), 503

    try:
        import random
        # 15% probability of generating an anomalous fraud vector
        is_fraud_sim = random.random() < 0.15

        v_feats = np.random.randn(28).tolist()
        if is_fraud_sim:
            # Inject fraud anomaly shifts into key PCA features
            v_feats[13] -= random.uniform(4.5, 7.0)  # V14
            v_feats[11] -= random.uniform(3.5, 6.0)  # V12
            v_feats[9]  -= random.uniform(3.5, 5.5)  # V10
            v_feats[16] -= random.uniform(3.0, 5.0)  # V17
            v_feats[3]  += random.uniform(3.5, 6.0)  # V4
            v_feats[10] += random.uniform(3.0, 5.0)  # V11
            raw_amount = random.uniform(400.0, 4500.0)
            hour = random.choice([2, 3, 4])
        else:
            raw_amount = float(np.random.exponential(scale=75.0))
            hour = random.randint(8, 22)

        log_amount = float(np.log1p(raw_amount))
        features = v_feats + [log_amount, hour]

        X = np.array(features, dtype=float).reshape(1, -1)
        X_scaled = SCALER.transform(X)

        probability = float(MODEL.predict_proba(X_scaled)[0][1])
        opt_thresh = float(METADATA.get("optimal_threshold", 0.5))
        prediction = 1 if probability >= opt_thresh else 0

        risk_data = assess_risk(probability, prediction, optimal_threshold=opt_thresh)
        risk_drivers = extract_shap_risk_drivers(SHAP_EXPLAINER, X_scaled[0].tolist(), METADATA["feature_names"], top_k=3)

        tx_id = f"SIM-{int(time.time() * 1000):X}"
        db_rec = log_transaction(
            tx_id=tx_id,
            features=features,
            probability=probability,
            prediction=prediction,
            risk_tier=risk_data["tier"],
            decision=risk_data["decision"]
        )

        return jsonify({
            "id": tx_id,
            "timestamp": time.strftime("%H:%M:%S"),
            "raw_amount_usd": round(raw_amount, 2),
            "hour": hour,
            "prediction": prediction,
            "probability": round(probability, 6),
            "is_fraud": prediction == 1,
            "optimal_threshold": opt_thresh,
            "confidence": round(max(probability, 1 - probability) * 100, 2),
            "label": "🚨 FRAUD DETECTED" if prediction == 1 else "✅ LEGITIMATE",
            "risk": risk_data,
            "risk_drivers": risk_drivers,
            "db_status": db_rec["status"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Main ───────────────────────────────────────────────────────
if __name__ == "__main__":
    load_artifacts()
    print("\n  Starting Credit Card Fraud Detection Web App...")
    print("  Open: http://localhost:5000\n")
    app.run(debug=True, port=5000)

