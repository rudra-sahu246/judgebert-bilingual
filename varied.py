"""Additional development families with independently varied sentence structure."""

TOPICS = (
    ("close the gate", "गेट बंद करना", "गेट बंद करें", "gate closure", "गेट बंद करना"),
    ("clear the hallway", "गलियारा खाली करना", "गलियारा खाली करें", "hallway clearance", "गलियारा खाली करना"),
    ("label the spare key", "अतिरिक्त चाबी पर निशान लगाना", "अतिरिक्त चाबी पर निशान लगाएँ", "spare key labelling", "अतिरिक्त चाबी पर निशान लगाना"),
    ("record the water level", "पानी का स्तर दर्ज करना", "पानी का स्तर दर्ज करें", "water level recording", "पानी का स्तर दर्ज करना"),
    ("remove the old carpet", "पुराना कालीन हटाना", "पुराना कालीन हटाएँ", "old carpet removal", "पुराना कालीन हटाना"),
    ("clean the roof drain", "छत की नाली साफ करना", "छत की नाली साफ करें", "roof drain cleaning", "छत की नाली साफ करना"),
    ("mark the reserved space", "आरक्षित जगह पर निशान लगाना", "आरक्षित जगह पर निशान लगाएँ", "reserved space marking", "आरक्षित जगह पर निशान लगाना"),
    ("replace the smoke alarm battery", "धुआँ चेतावनी यंत्र की बैटरी बदलना", "धुआँ चेतावनी यंत्र की बैटरी बदलें", "battery replacement", "बैटरी बदलना"),
    ("dry the shared landing", "साझा चबूतरा सुखाना", "साझा चबूतरा सुखाएँ", "landing drying", "साझा चबूतरा सुखाना"),
    ("wash the kitchen filter", "रसोई का फिल्टर धोना", "रसोई का फिल्टर धोएँ", "filter washing", "रसोई का फिल्टर धोना"),
    ("secure the bicycle rack", "साइकिल का स्टैंड सुरक्षित करना", "साइकिल का स्टैंड सुरक्षित करें", "rack security", "साइकिल का स्टैंड सुरक्षित करना"),
)


def examples(split):
    groups = {"train": range(7), "validation": range(7, 9), "test": range(9, 11)}
    rows = []
    for index in groups.get(split, ()):
        verb, hi, hi_imperative, noun, hi_noun = TOPICS[index]
        days = index + 8
        english = f"Responsibility for {noun} rests with the leaseholder."
        hindi = f"{hi_noun} पट्टेदार की जिम्मेदारी है।"
        variants = {"en": [
            ("paraphrase", english, f"The leaseholder is responsible for {noun}.", 10),
            ("restructure", english, f"It is the leaseholder's duty to {verb}.", 10),
            ("simplify", f"The leaseholder is under an obligation to {verb}.",
             f"The leaseholder must {verb}.", 10),
            ("entity", english, english.replace("leaseholder", "owner"), 3),
            ("condition", f"Only after a warning, the leaseholder must {verb}.",
             f"The leaseholder must {verb} in all cases.", 4),
            ("exception", f"The leaseholder must {verb} unless access is blocked.",
             f"The leaseholder must {verb}, even when access is blocked.", 4),
            ("obligation_to_permission", english, f"The leaseholder is allowed to {verb}.", 3),
            ("contradiction", english, f"The leaseholder has no duty to {verb}.", 1),
            ("deadline", f"Within {days} days, {verb}.", f"Within {days+4} days, {verb}.", 4),
            ("unrelated", english, f"The community room closes after sunset on day {days}.", 1),
        ], "hi": [
            ("paraphrase", hindi, f"पट्टेदार {hi_noun} के लिए जिम्मेदार है।", 10),
            ("restructure", hindi, f"पट्टेदार का कर्तव्य है कि वह यह काम करे: {hi}।", 10),
            ("simplify", f"पट्टेदार पर यह दायित्व है: {hi}।", f"पट्टेदार को यह काम करना होगा: {hi}।", 10),
            ("entity", hindi, hindi.replace("पट्टेदार", "स्वामी"), 3),
            ("condition", f"चेतावनी मिलने के बाद ही पट्टेदार यह काम करे: {hi}।",
             f"पट्टेदार हर हाल में यह काम करे: {hi}।", 4),
            ("exception", f"पहुँच बाधित होने की स्थिति को छोड़कर पट्टेदार के लिए अनिवार्य है: {hi}।",
             f"पहुँच बाधित हो तब भी पट्टेदार के लिए अनिवार्य है: {hi}।", 4),
            ("obligation_to_permission", hindi, f"पट्टेदार को अनुमति है कि वह यह काम करे: {hi}।", 3),
            ("contradiction", hindi, f"यह पट्टेदार का कर्तव्य नहीं है: {hi}।", 1),
            ("deadline", f"{days} दिनों के भीतर {hi_imperative}।",
             f"{days+4} दिनों के भीतर {hi_imperative}।", 4),
            ("unrelated", hindi, f"सामुदायिक कक्ष दिन {days} को सूर्यास्त के बाद बंद होता है।", 1),
        ]}
        for language, items in variants.items():
            for category, original, simplified, score in items:
                rows.append({"id": f"style-{language}-{index}-{category}",
                             "source": f"style-topic-{index}", "language": language,
                             "category": category, "original": original, "simplified": simplified,
                             "context": ("The leaseholder is the person named as the occupant."
                                         if language == "en" else
                                         "पट्टेदार वह व्यक्ति है जिसका नाम निवासी के रूप में दर्ज है।"),
                             "score": score, "rating_origin": "synthetic",
                             "contradiction": category == "contradiction"})
    return rows
