from pathlib import Path
import unittest

from lmp import read_rows, text_changes
from synthetic import SPLITS, audit, generate, supervision


class SyntheticTests(unittest.TestCase):
    def test_source_groups_and_challenge(self):
        splits = {name: generate(name) for name in SPLITS}
        splits["independent"] = read_rows(Path(__file__).parent / "data/independent.jsonl")
        report = audit(splits)
        self.assertEqual(report["train"]["pairs"], 616)
        self.assertEqual(report["independent"]["languages"], {"en": 22, "hi": 22})
        with self.assertRaisesRegex(ValueError, "Source leakage"):
            audit({"train": splits["train"], "leak": splits["train"]})

    def test_supervision_and_hindi_evidence(self):
        self.assertEqual(supervision([7, 2]), 1)
        self.assertEqual(supervision([3], contradiction=True), 2)
        row = {"original": "किरायेदार को 14/06/2027 को भुगतान नहीं करना होगा।",
               "simplified": "मकान मालिक को 15/06/2027 को भुगतान करना होगा।"}
        self.assertTrue({"date", "negation", "entity"} <= {
            item["possible_change"] for item in text_changes(row)})
