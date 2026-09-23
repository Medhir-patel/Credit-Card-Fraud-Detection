"""
Integration tests for Fraud Detection Flask Endpoints.
"""

import unittest
import json
from app import app, load_artifacts


class TestFraudDetectionAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        load_artifacts()
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_index_route(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"FraudGuard AI", response.data)

    def test_api_metadata(self):
        response = self.client.get("/api/metadata")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("best_model", data)
        self.assertIn("feature_names", data)
        self.assertEqual(len(data["feature_names"]), 30)

    def test_api_sample(self):
        response = self.client.get("/api/sample")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("legitimate", data)
        self.assertIn("fraud", data)
        self.assertEqual(len(data["legitimate"]), 30)
        self.assertEqual(len(data["fraud"]), 30)

    def test_single_prediction_legit(self):
        sample_resp = self.client.get("/api/sample")
        sample_data = sample_resp.get_json()
        legit_features = sample_data["legitimate"]

        response = self.client.post(
            "/predict",
            data=json.dumps({"features": legit_features}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("prediction", data)
        self.assertIn("probability", data)
        self.assertIn("risk", data)
        self.assertIn("risk_drivers", data)
        self.assertEqual(data["prediction"], 0)
        self.assertFalse(data["is_fraud"])
        self.assertEqual(data["risk"]["decision"], "APPROVE")
        self.assertTrue(len(data["risk_drivers"]) > 0)

    def test_single_prediction_fraud(self):
        sample_resp = self.client.get("/api/sample")
        sample_data = sample_resp.get_json()
        fraud_features = sample_data["fraud"]

        response = self.client.post(
            "/predict",
            data=json.dumps({"features": fraud_features}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("prediction", data)
        self.assertIn("probability", data)
        self.assertIn("risk", data)
        self.assertEqual(data["prediction"], 1)
        self.assertTrue(data["is_fraud"])

    def test_predict_invalid_feature_length(self):
        response = self.client.post(
            "/predict",
            data=json.dumps({"features": [1.0, 2.0, 3.0]}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)

    def test_batch_prediction(self):
        sample_resp = self.client.get("/api/sample")
        sample_data = sample_resp.get_json()
        legit = sample_data["legitimate"]
        fraud = sample_data["fraud"]

        batch_payload = {
            "transactions": [
                {"id": "TX-01", "features": legit},
                {"id": "TX-02", "features": fraud}
            ]
        }

        response = self.client.post(
            "/predict/batch",
            data=json.dumps(batch_payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("summary", data)
        self.assertIn("results", data)
        self.assertEqual(data["summary"]["total_analyzed"], 2)
        self.assertEqual(data["summary"]["fraud_count"], 1)
        self.assertEqual(data["summary"]["legitimate_count"], 1)
        self.assertEqual(len(data["results"]), 2)

    def test_swagger_docs_and_openapi(self):
        docs_resp = self.client.get("/docs")
        self.assertEqual(docs_resp.status_code, 200)
        self.assertIn(b"SwaggerUIBundle", docs_resp.data)

        spec_resp = self.client.get("/api/openapi.json")
        self.assertEqual(spec_resp.status_code, 200)
        spec_data = spec_resp.get_json()
        self.assertIn("openapi", spec_data)
        self.assertIn("/predict", spec_data["paths"])

    def test_batch_export_csv(self):
        sample_resp = self.client.get("/api/sample")
        sample_data = sample_resp.get_json()
        legit = sample_data["legitimate"]

        batch_payload = {
            "transactions": [
                {"id": "TX-EXP-1", "features": legit}
            ]
        }

        response = self.client.post(
            "/predict/batch/export",
            data=json.dumps(batch_payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "text/csv")
        self.assertIn(b"Transaction ID", response.data)
        self.assertIn(b"TX-EXP-1", response.data)

    def test_analyst_api_endpoints(self):
        analyst_page = self.client.get("/analyst")
        self.assertEqual(analyst_page.status_code, 200)

        txs_resp = self.client.get("/api/transactions")
        self.assertEqual(txs_resp.status_code, 200)
        data = txs_resp.get_json()
        self.assertIn("transactions", data)

    def test_simulate_endpoint(self):
        sim_resp = self.client.post("/api/simulate")
        self.assertEqual(sim_resp.status_code, 200)
        data = sim_resp.get_json()
        self.assertIn("id", data)
        self.assertIn("probability", data)
        self.assertIn("risk", data)
        self.assertIn("risk_drivers", data)
        self.assertTrue(data["id"].startswith("SIM-"))

    def test_dashboard_and_history_routes(self):
        dash_resp = self.client.get("/dashboard")
        self.assertEqual(dash_resp.status_code, 200)
        self.assertIn(b"Analytics Dashboard", dash_resp.data)

        hist_resp = self.client.get("/history")
        self.assertEqual(hist_resp.status_code, 200)
        self.assertIn(b"Transaction History", hist_resp.data)

    def test_interactive_simulate_route(self):
        payload = {
            "amount": 1250.0,
            "merchant_category": "electronics",
            "transaction_type": "online",
            "hour": 3,
            "location": "domestic"
        }
        res = self.client.post("/simulate", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("tx_id", data)
        self.assertIn("alert", data)
        self.assertIn("cardholder_sms", data["alert"])

    def test_live_feed_and_analytics_apis(self):
        feed_resp = self.client.get("/api/live-feed")
        self.assertEqual(feed_resp.status_code, 200)
        feed_data = feed_resp.get_json()
        self.assertIn("tx_id", feed_data)

        analytics_resp = self.client.get("/api/analytics")
        self.assertEqual(analytics_resp.status_code, 200)
        analytics_data = analytics_resp.get_json()
        self.assertIn("total_transactions", analytics_data)


if __name__ == "__main__":
    unittest.main()
