import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from lmp import LANGUAGES, prediction, release


class ReleaseTests(unittest.TestCase):
    def test_release_requires_independent_data_and_rejects_false_accepts(self):
        rows = [
            {"id": f"{lang}-{i}", "language": lang, "source": f"held-out-{i}",
             "original": "Original", "simplified": "Simplified", "context": "Context",
             "ratings": {"a": label, "b": label}, "rating_origin": "human"}
            for lang in LANGUAGES for i in range(30) for label in [3 if i < 15 else 9]
        ]
        with TemporaryDirectory() as root:
            model, dataset, out = (Path(root) / name for name in ("model", "data", "release"))
            model.write_text(json.dumps({"demo": False, "source_ids": ["dev"]}))
            dataset.write_text("\n".join(map(json.dumps, rows)))
            args = SimpleNamespace(model=model, dataset=dataset, out=out, local_only=True)
            with patch("lmp.Scorer") as scorer:
                scorer.return_value.predict.side_effect = lambda inputs: [
                    {**row, **prediction(row["ratings"]["a"] if row["context"] else 9)}
                    for row in inputs]
                with self.assertRaisesRegex(ValueError, "sentence_only/en: false acceptance"):
                    release(args)
                self.assertFalse(out.exists())
                scorer.return_value.predict.side_effect = None
                scorer.return_value.predict.return_value = [
                    {**row, **prediction(row["ratings"]["a"])} for row in rows]
                release(args)
                results = json.loads(out.read_text())["release"]["results"]
                self.assertEqual(results["sentence_only"]["hi"]["n"], 30)
                self.assertEqual(results["with_context"]["hi"]["n"], 30)
                model.write_text(json.dumps({"demo": False, "source_ids": ["held-out-0"]}))
                with self.assertRaisesRegex(ValueError, "absent"):
                    release(args)
