from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from lmp import Scorer
from service import create_app


class ServiceTests(unittest.TestCase):
    def test_startup_rejects_unapproved_model(self):
        with TemporaryDirectory() as root:
            artifact = Path(root) / "model.json"
            artifact.write_text('{"demo": false}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "release-approved"):
                Scorer(artifact, require_release=True)

    def test_startup_rejects_unsupported_encoder(self):
        with TemporaryDirectory() as root:
            artifact = Path(root) / "model.json"
            artifact.write_text('{"demo": false, "encoder": "unknown", '
                                '"release": {"policy_version": 2}}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported encoder"):
                Scorer(artifact, require_release=True)

    def test_health_auth_validation_and_success(self):
        token = "a" * 32
        payload = {"pairs": [{"language": "hi", "original": "किराया देय है।",
                              "simplified": "किराया देय है।"}]}
        headers = {"Authorization": f"Bearer {token}"}
        with patch.dict("os.environ", {"LMP_API_TOKEN": token}), patch("service.Scorer") as scorer:
            scorer.return_value.predict.side_effect = lambda rows: [
                {**row, "score": 10., "decision": "Preserved", "mode": "identical",
                 "evidence": None} for row in rows]
            with TestClient(create_app()) as client:
                self.assertEqual(client.get("/healthz").json(), {"status": "ready"})
                self.assertEqual(client.post("/v1/score", json=payload).status_code, 401)
                self.assertEqual(client.post("/v1/score", json={}, headers=headers).status_code, 422)
                response = client.post("/v1/score", headers={
                    **headers, "Content-Type": "application/json"}, content=b" " * (256 * 1024 + 1))
                self.assertEqual(response.status_code, 413)
                response = client.post("/v1/score", json=payload, headers=headers)
                self.assertEqual(response.status_code, 200)
                item = response.json()["predictions"][0]
                self.assertEqual(item["decision"], "Preserved")
                self.assertNotIn("original", item)
                self.assertEqual(scorer.call_count, 1)
                self.assertTrue(scorer.call_args.kwargs["require_release"])
