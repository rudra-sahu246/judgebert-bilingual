import unittest
from pathlib import Path

from lmp import agreement, evidence, overlap, prediction, read_rows, split_rows, validate


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.rows = read_rows(Path(__file__).parent / "data/demo.jsonl")

    def test_demo_cannot_enter_research_pipeline(self):
        validate(self.rows, demo=True)
        with self.assertRaisesRegex(ValueError, "human ratings"):
            validate(self.rows)

    def test_source_split_is_bilingual_and_disjoint(self):
        train, test = split_rows(self.rows, 42)
        self.assertFalse({r["source"] for r in train} & {r["source"] for r in test})
        for part in (train, test):
            self.assertEqual({r["language"] for r in part}, {"en", "hi"})

    def test_agreement(self):
        self.assertEqual(agreement([{"ratings": {"a": 1, "b": 1}},
                                    {"ratings": {"a": 10, "b": 10}}]), 1)
        self.assertIsNone(agreement([{"ratings": {"a": 5, "b": 5}}]))

    def test_hindi_overlap_and_evidence(self):
        self.assertEqual(overlap("किराया देय है।", "किराया देय है"), 1)
        self.assertEqual(overlap("किराया", "किराये"), 0)
        self.assertEqual(evidence(self.rows[1])["possible_change"], "number_or_date")
        self.assertEqual(evidence(self.rows[-1])["possible_change"], "scope")

    def test_decision_boundaries(self):
        for value, decision in [(1, "Not Preserved"), (2, "Partially Preserved"),
                                (6, "Partially Preserved"), (7, "Preserved")]:
            self.assertEqual(prediction(value)["decision"], decision)


if __name__ == "__main__":
    unittest.main()
