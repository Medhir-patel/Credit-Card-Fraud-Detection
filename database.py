"""
Database Persistence & Audit Logging for Fraud Detection
=========================================================
Manages SQLite storage for transaction logs, risk scores, and analyst reviews.
"""

import sqlite3
import json
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

DEFAULT_DB_PATH = "fraud_audit.db"


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initialize SQLite database, transactions table, and fraud rules table."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            features_json TEXT NOT NULL,
            probability REAL NOT NULL,
            prediction INTEGER NOT NULL,
            risk_tier TEXT NOT NULL,
            decision TEXT NOT NULL,
            status TEXT NOT NULL,
            analyst_decision TEXT,
            analyst_notes TEXT,
            updated_at TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fraud_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            feature_index INTEGER NOT NULL,
            operator TEXT NOT NULL,
            threshold REAL NOT NULL,
            score_impact REAL NOT NULL,
            action TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL
        )
    """)

    # Seed default rules if table is empty
    cursor.execute("SELECT COUNT(*) FROM fraud_rules")
    if cursor.fetchone()[0] == 0:
        from rule_engine import hybrid_engine
        now_iso = datetime.now(timezone.utc).isoformat()
        for r in hybrid_engine.default_rules:
            cursor.execute("""
                INSERT OR IGNORE INTO fraud_rules (
                    code, name, description, feature_index, operator,
                    threshold, score_impact, action, enabled, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                r["code"], r["name"], r["description"], r["feature_index"], r["operator"],
                r["threshold"], r["score_impact"], r["action"], 1 if r["enabled"] else 0, now_iso
            ))

    conn.commit()
    conn.close()


def log_transaction(
    tx_id: str,
    features: list,
    probability: float,
    prediction: int,
    risk_tier: str,
    decision: str,
    status: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> Dict[str, Any]:
    """
    Save an evaluated transaction into the audit log.
    """
    if status is None:
        if risk_tier in ("HIGH", "CRITICAL"):
            status = "PENDING_REVIEW"
        elif risk_tier == "MODERATE":
            status = "CHALLENGED_2FA"
        else:
            status = "AUTO_APPROVED"

    now_iso = datetime.now(timezone.utc).isoformat()
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO transactions (
            id, timestamp, features_json, probability, prediction,
            risk_tier, decision, status, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        tx_id, now_iso, json.dumps(features), float(probability), int(prediction),
        risk_tier, decision, status, now_iso
    ))

    conn.commit()
    conn.close()

    return {
        "id": tx_id,
        "timestamp": now_iso,
        "probability": probability,
        "prediction": prediction,
        "risk_tier": risk_tier,
        "decision": decision,
        "status": status
    }


def get_transactions(
    status: Optional[str] = None,
    risk_tier: Optional[str] = None,
    limit: int = 100,
    db_path: str = DEFAULT_DB_PATH
) -> List[Dict[str, Any]]:
    """
    Fetch logged transactions filtered by status or risk tier.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()

    query = "SELECT * FROM transactions"
    params = []
    conditions = []

    if status:
        conditions.append("status = ?")
        params.append(status)
    if risk_tier:
        conditions.append("risk_tier = ?")
        params.append(risk_tier)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    results = []
    for r in rows:
        row_dict = dict(r)
        if "features_json" in row_dict and row_dict["features_json"]:
            try:
                row_dict["features"] = json.loads(row_dict["features_json"])
            except Exception:
                row_dict["features"] = []
        results.append(row_dict)

    return results


def get_transaction_by_id(tx_id: str, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieve a single transaction by ID."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM transactions WHERE id = ?", (tx_id,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    row_dict = dict(row)
    if "features_json" in row_dict and row_dict["features_json"]:
        try:
            row_dict["features"] = json.loads(row_dict["features_json"])
        except Exception:
            row_dict["features"] = []
    return row_dict


def update_analyst_review(
    tx_id: str,
    analyst_decision: str,
    analyst_notes: str = "",
    db_path: str = DEFAULT_DB_PATH
) -> bool:
    """
    Record an analyst's manual review decision ('APPROVED' or 'CONFIRMED_FRAUD').
    """
    if analyst_decision not in ("APPROVED", "CONFIRMED_FRAUD", "REJECTED"):
        raise ValueError(f"Invalid analyst decision: {analyst_decision}")

    new_status = "RESOLVED_" + analyst_decision
    now_iso = datetime.now(timezone.utc).isoformat()

    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE transactions
        SET analyst_decision = ?, analyst_notes = ?, status = ?, updated_at = ?
        WHERE id = ?
    """, (analyst_decision, analyst_notes, new_status, now_iso, tx_id))

    rows_affected = cursor.rowcount
    conn.commit()
    conn.close()

    return rows_affected > 0


def get_fraud_rules(enabled_only: bool = False, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Fetch stored fraud rules from database."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    query = "SELECT * FROM fraud_rules"
    if enabled_only:
        query += " WHERE enabled = 1"
    query += " ORDER BY id ASC"

    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    results = []
    for r in rows:
        d = dict(r)
        d["enabled"] = bool(d["enabled"])
        results.append(d)
    return results


def add_fraud_rule(rule: Dict[str, Any], db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Add a new custom rule to database."""
    init_db(db_path)
    now_iso = datetime.now(timezone.utc).isoformat()
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO fraud_rules (
            code, name, description, feature_index, operator,
            threshold, score_impact, action, enabled, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        rule["code"], rule["name"], rule.get("description", ""),
        int(rule["feature_index"]), rule["operator"], float(rule["threshold"]),
        float(rule["score_impact"]), rule["action"], 1 if rule.get("enabled", True) else 0, now_iso
    ))

    rule_id = cursor.lastrowid
    conn.commit()
    conn.close()

    rule_copy = dict(rule)
    rule_copy["id"] = rule_id
    rule_copy["created_at"] = now_iso
    return rule_copy


def toggle_fraud_rule(rule_id: int, enabled: bool, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Toggle the enabled status of a rule."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("UPDATE fraud_rules SET enabled = ? WHERE id = ?", (1 if enabled else 0, rule_id))
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected > 0


def delete_fraud_rule(rule_id: int, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Delete a custom rule by ID."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM fraud_rules WHERE id = ?", (rule_id,))
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected > 0


if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")
