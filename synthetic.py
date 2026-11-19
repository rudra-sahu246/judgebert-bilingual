"""Generate deterministic, source-isolated English and Hindi legal exercises."""

from collections import Counter
import json
from pathlib import Path
import re
import unicodedata

from lmp import write_json


# Each topic is a fictional clause family. The translations share its group ID.
TOPICS = (
    ("inspection report", "निरीक्षण रिपोर्ट"), ("repair invoice", "मरम्मत का बिल"),
    ("rent receipt", "किराये की रसीद"), ("key inventory", "चाबियों की सूची"),
    ("deposit statement", "जमा राशि का विवरण"), ("access request", "प्रवेश का अनुरोध"),
    ("renewal form", "नवीनीकरण प्रपत्र"), ("parking permit", "पार्किंग अनुमति पत्र"),
    ("meter reading", "मीटर की रीडिंग"), ("damage report", "क्षति की रिपोर्ट"),
    ("insurance certificate", "बीमा प्रमाणपत्र"), ("pet application", "पालतू पशु का आवेदन"),
    ("visitor record", "आगंतुक का अभिलेख"), ("maintenance schedule", "रखरखाव की समयसूची"),
    ("handover record", "संपत्ति सौंपने का अभिलेख"), ("utility statement", "सेवा शुल्क का विवरण"),
    ("debit notice", "देय राशि की सूचना"), ("complaint record", "शिकायत का अभिलेख"),
    ("termination notice", "समाप्ति की सूचना"), ("fire safety certificate", "अग्नि सुरक्षा प्रमाणपत्र"),
    ("storage authorization", "भंडारण की अनुमति"), ("occupancy declaration", "अधिभोग की घोषणा"),
    ("lift inspection log", "लिफ्ट निरीक्षण की बही"), ("furniture checklist", "फर्नीचर की जाँच सूची"),
)
SPLITS = {"train": range(14), "validation": range(14, 17),
          "test": range(17, 20), "challenge": range(20, 24)}


def supervision(costs, *, contradiction=False):
    """Subtract legal-impact penalties, with a cap for explicit contradictions."""
    score = max(1, 10 - sum(costs))
    return min(score, 2) if contradiction else score


def variants(topic, language, days, amount):
    if language == "en":
        base = f"The tenant must submit the {topic} within {days} days if requested, except during emergency closure."
        plain = f"If asked, the tenant must send the {topic} within {days} days, unless the premises are closed for an emergency."
        moved = f"Except during emergency closure, the tenant must submit the {topic} within {days} days if requested."
        permitted = base.replace("must", "may")
        negative = base.replace("must submit", "must not submit")
        paid = f"The landlord must reimburse {amount} rupees for the {topic} after approval."
        ref = f"The tenant must submit the {topic} within the standard period."
        explicit = f"The tenant must send the {topic} within {days} days."
        context = f"For the {topic}, the standard period is {days} days after notice. Requests are made in writing."
        unrelated = f"The lift may be inspected by a technician during the annual service visit."
        condition = base.replace(" if requested", "")
        secondary = f"The tenant must retain the {topic} and, where available, its reference number."
        scope = f"The tenant must submit the {topic} for every unit."
        soft = base.replace("must", "should")
        return context, [
            ("identical", base, base, [], False),
            ("paraphrase", base, plain, [], False),
            ("restructure", base, moved, [], False),
            ("simplify", paid, paid.replace("reimburse", "pay back"), [], False),
            ("obligation_to_permission", base, permitted, [7], False),
            ("permission_to_obligation", permitted, base, [7], False),
            ("negation_removed", negative, base, [9], True),
            ("contradiction", base, negative, [9], True),
            ("deadline", base, base.replace(f"{days} days", f"{days+3} days"), [6], False),
            ("amount", paid, paid.replace(str(amount), str(amount+50)), [6], False),
            ("entity", base, base.replace("tenant", "landlord"), [7], False),
            ("condition", base, condition, [6], False),
            ("changed_condition", base, base.replace("if requested", "if approved"), [6], False),
            ("exception", base, base.replace(", except during emergency closure", ""), [6], False),
            ("secondary_omission", secondary, secondary.split(" and,")[0] + ".", [3], False),
            ("scope", scope, scope.replace("every unit", "the occupied unit"), [4], False),
            ("added_obligation", base, base + " The tenant must also pay an administrative fee.", [7], False),
            ("removed_obligation", secondary, f"The tenant should note the reference number for the {topic}.", [7, 2], False),
            ("weakened_obligation", base, soft, [6], False),
            ("context_resolution", ref, explicit, [], False),
            ("context_conflict", ref, explicit.replace(f"{days} days", f"{days+3} days"), [6], False),
            ("unrelated", base, unrelated, [9], False),
        ]
    base = f"अनुरोध मिलने पर किरायेदार को {days} दिनों के भीतर {topic} जमा करनी होगी, आपातकालीन बंदी को छोड़कर।"
    plain = f"अगर माँगा जाए तो किरायेदार {days} दिनों के अंदर {topic} भेजेगा, सिवाय आपातकालीन बंदी के।"
    moved = f"आपातकालीन बंदी को छोड़कर, अनुरोध मिलने पर किरायेदार को {days} दिनों के भीतर {topic} जमा करनी होगी।"
    permitted = base.replace("को ", "", 1).replace("जमा करनी होगी", "जमा कर सकता है")
    negative = base.replace("जमा करनी होगी", "जमा नहीं करनी होगी")
    paid = f"स्वीकृति के बाद मकान मालिक को {topic} के लिए {amount} रुपये की प्रतिपूर्ति करनी होगी।"
    ref = f"किरायेदार को तय समय सीमा में {topic} जमा करनी होगी।"
    explicit = f"किरायेदार को {days} दिनों के भीतर {topic} भेजनी होगी।"
    context = f"{topic} की तय समय सीमा सूचना मिलने के बाद {days} दिन है। अनुरोध लिखित रूप में होगा।"
    secondary = f"किरायेदार को {topic} और उपलब्ध होने पर उसकी संदर्भ संख्या रखनी होगी।"
    scope = f"किरायेदार को हर इकाई के लिए {topic} जमा करनी होगी।"
    return context, [
        ("identical", base, base, [], False),
        ("paraphrase", base, plain, [], False),
        ("restructure", base, moved, [], False),
        ("simplify", paid, paid.replace("प्रतिपूर्ति करनी होगी", "वापस देने होंगे"), [], False),
        ("obligation_to_permission", base, permitted, [7], False),
        ("permission_to_obligation", permitted, base, [7], False),
        ("negation_removed", negative, base, [9], True),
        ("contradiction", base, negative, [9], True),
        ("deadline", base, base.replace(f"{days} दिनों", f"{days+3} दिनों"), [6], False),
        ("amount", paid, paid.replace(str(amount), str(amount+50)), [6], False),
        ("entity", base, base.replace("किरायेदार", "मकान मालिक"), [7], False),
        ("condition", base, base.removeprefix("अनुरोध मिलने पर "), [6], False),
        ("changed_condition", base, base.replace("अनुरोध मिलने पर", "स्वीकृति मिलने पर"), [6], False),
        ("exception", base, base.replace(", आपातकालीन बंदी को छोड़कर", ""), [6], False),
        ("secondary_omission", secondary, f"किरायेदार को {topic} रखनी होगी।", [3], False),
        ("scope", scope, scope.replace("हर इकाई", "केवल इस्तेमाल की जा रही इकाई"), [4], False),
        ("added_obligation", base, base + " किरायेदार को प्रशासनिक शुल्क भी देना होगा।", [7], False),
        ("removed_obligation", secondary, f"किरायेदार को {topic} की संदर्भ संख्या लिख लेनी चाहिए।", [7, 2], False),
        ("weakened_obligation", base, base.replace("जमा करनी होगी", "जमा करनी चाहिए"), [6], False),
        ("context_resolution", ref, explicit, [], False),
        ("context_conflict", ref, explicit.replace(f"{days} दिनों", f"{days+3} दिनों"), [6], False),
        ("unrelated", base, "तकनीशियन वार्षिक सेवा के दौरान लिफ्ट की जाँच कर सकता है।", [9], False),
    ]


def generate(split, *, extra_round=0, categories=None):
    if extra_round and split != "train":
        raise ValueError("Augmentation is limited to the training split")
    rows = []
    for family in SPLITS[split]:
        for lang, topic in zip(("en", "hi"), TOPICS[family]):
            context, examples = variants(topic, lang, 5 + family + extra_round * 29,
                                         250 + family * 15 + extra_round * 365)
            for category, original, simplified, costs, contradiction in examples:
                if categories is not None and category not in categories:
                    continue
                rows.append({"id": f"{split}-{family}-{lang}-{category}-{extra_round}",
                             "source": f"lease-topic-{family}", "language": lang,
                             "category": category, "original": original, "simplified": simplified,
                             "context": context, "score": supervision(costs, contradiction=contradiction),
                             "rating_origin": "synthetic", "contradiction": contradiction})
    return rows


def canonical(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def audit(splits):
    keys, sources = set(), {}
    prior = {}
    report = {}
    for split, rows in splits.items():
        near = 0
        for row in rows:
            group = row["source"]
            if sources.setdefault(group, split) != split:
                raise ValueError(f"Source leakage: {group}")
            key = tuple(canonical(row[field]) for field in ("original", "simplified", "context"))
            if key in keys:
                raise ValueError(f"Duplicate example: {row['id']}")
            keys.add(key)
            tokens = tuple(set(re.findall(r"\S+", text)) for text in key[:2])
            for previous_split, previous_tokens in prior.get(row["language"], []):
                if previous_split != split and all(
                    len(left & right) / max(1, len(left | right)) >= .9
                    for left, right in zip(tokens, previous_tokens)
                ):
                    near += 1
                    break
            prior.setdefault(row["language"], []).append((split, tokens))
            if any(not text for text in key) or not 1 <= row["score"] <= 10:
                raise ValueError(f"Invalid example: {row['id']}")
            if (key[0] == key[1]) != (row["category"] == "identical"):
                raise ValueError(f"Unexpected identity: {row['id']}")
            if row["contradiction"] and row["score"] > 2:
                raise ValueError(f"Contradiction score out of range: {row['id']}")
            for field in ("original", "simplified", "context"):
                hindi = bool(re.search(r"[\u0900-\u097f]", row[field]))
                if hindi != (row["language"] == "hi"):
                    raise ValueError(f"Unexpected script: {row['id']}/{field}")
        report[split] = {"pairs": len(rows), "sources": len({r["source"] for r in rows}),
                         "near_duplicates_with_earlier_splits": near,
                         "languages": dict(Counter(r["language"] for r in rows)),
                         "categories": dict(Counter(r["category"] for r in rows)),
                         "scores": dict(Counter(r["score"] for r in rows))}
    return report


def write_dataset(root, *, varied=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    splits = {name: generate(name) for name in SPLITS}
    if varied:
        from varied import examples

        for name in ("train", "validation", "test"):
            splits[name].extend(examples(name))
    write_json(root / "audit.json", audit(splits))
    for split, rows in splits.items():
        (root / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return splits


if __name__ == "__main__":
    write_dataset("out/synthetic/data")
