"""Opt-in HTTP smoke test with the real offline encoder and demo weights."""

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from lmp import Scorer
from service import create_app


@unittest.skipUnless(os.environ.get("LMP_RUN_INTEGRATION") == "1", "Enable cached model smoke test")
class EncoderIntegrationTests(unittest.TestCase):
    def test_bilingual_http_scoring_and_token_budget(self):
        model = Path(os.environ.get("LMP_TEST_MODEL", "out/demo/model.json"))
        scorer = Scorer(model, allow_demo=True, local_only=True)
        token = "a" * 32
        headers = {"Authorization": f"Bearer {token}"}
        pairs = [
            {"language": "en", "original": "Rent is due Monday.",
             "simplified": "Pay rent Monday."},
            {"language": "hi", "original": "किराया सोमवार को देना होगा।",
             "simplified": "सोमवार को किराया दें।", "context": "यह मासिक किराया है।"},
        ]
        with patch.dict(os.environ, {"LMP_API_TOKEN": token}), patch(
            "service.Scorer", return_value=scorer
        ), TestClient(create_app()) as client:
            response = client.post("/v1/score", json={"pairs": pairs}, headers=headers)
            self.assertEqual(response.status_code, 200, response.text)
            predictions = response.json()["predictions"]
            self.assertEqual([item["mode"] for item in predictions],
                             ["sentence_only", "with_context"])
            self.assertTrue(all(1 <= item["score"] <= 10 for item in predictions))
            pairs[0]["context"] = "notice " * 300
            response = client.post("/v1/score", json={"pairs": pairs}, headers=headers)
            self.assertEqual(response.status_code, 422)
