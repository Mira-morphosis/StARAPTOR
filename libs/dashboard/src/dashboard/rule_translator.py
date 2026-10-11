#!/usr/bin/env python3
"""Turns mined association rules into human-readable English sentences.

The analyzer stores each rule as two itemsets joined by ``" AND "`` (see
``analyzer.analyzer._mine_transactions``), where every item is a categorical tag
produced by ``data_standardizer.DataStandardizer.transactional``:

* score pillars  -> ``gameplay_3``, ``monetizationModel_2``  (a rating out of 5)
* booleans       -> ``voted_up_True``, ``actuallyRecommendsTitle_False`` (qualitative)
* playtime       -> ``playtime_heavy`` (a coarse playtime bucket)
* language       -> ``lang_english``
* hardware       -> ``hw_os_Windows_11``, ``hw_ram_32GB``, ``hw_vram_16GB``
* KPI trend      -> ``kpi_retention_growth``

Each tag is rendered as a past-tense predicate describing what those reviewers
did / had, so that an itemset reads naturally both on its own (the reviews-modal
filter) and inside the full rule sentence.
"""
import re

# --- Score pillars: tag name -> readable label -------------------------------
PILLARS = {
    "gameplay": "gameplay",
    "community": "community",
    "immersion": "immersion",
    "story": "story",
    "audio": "audio",
    "graphics": "graphics",
    "controls": "controls",
    "difficulty": "difficulty",
    "customization": "customization",
    "replayability": "replayability",
    "monetizationModel": "monetization model",
    "multiplayer": "multiplayer",
}

# --- Boolean (qualitative) attributes: tag name -> {True: phrase, False: phrase}
BOOLEANS = {
    "voted_up": {
        "True": "left a positive review",
        "False": "left a negative review",
    },
    "actuallyRecommendsTitle": {
        "True": "recommended the game",
        "False": "did not recommend the game",
    },
    "containsPositiveAspects": {
        "True": "highlighted positive aspects",
        "False": "did not highlight any positive aspects",
    },
    "containsNegativeAspects": {
        "True": "highlighted negative aspects",
        "False": "did not highlight any negative aspects",
    },
    "containsBugDescription": {
        "True": "reported bugs in their review",
        "False": "did not report any bugs",
    },
    "isUseful": {
        "True": "wrote a review others found helpful",
        "False": "wrote a review others did not find helpful",
    },
    "steam_purchase": {
        "True": "purchased the game on Steam",
        "False": "had not purchased the game on Steam",
    },
    "received_for_free": {
        "True": "received the game for free",
        "False": "did not receive the game for free",
    },
    "primarily_steam_deck": {
        "True": "primarily played on Steam Deck",
        "False": "did not primarily play on Steam Deck",
    },
}

# --- Playtime buckets: tag suffix -> phrase ----------------------------------
PLAYTIME = {
    "refund_zone": "played under 2 hours",
    "casual": "played under 10 hours",
    "mid_core": "played between 10 and 50 hours",
    "heavy": "played over 50 hours",
    "hardcore": "played over 200 hours",
}

# --- KPI retention trends: tag suffix -> phrase ------------------------------
KPI_TRENDS = {
    "retention_growth": "joined during a period of rising player numbers",
    "retention_drop": "joined during a period of falling player numbers",
    "retention_stable": "joined during a period of stable player numbers",
}

# --- Language codes -> readable names (fallback: title-cased code) -----------
LANG_NAMES = {
    "english": "English",
    "schinese": "Simplified Chinese",
    "tchinese": "Traditional Chinese",
    "russian": "Russian",
    "turkish": "Turkish",
    "french": "French",
    "german": "German",
    "spanish": "Spanish",
    "brazilian": "Brazilian Portuguese",
    "italian": "Italian",
    "japanese": "Japanese",
    "korean": "Korean",
    "polish": "Polish",
    "portuguese": "Portuguese",
    "dutch": "Dutch",
    "swedish": "Swedish",
    "norwegian": "Norwegian",
    "danish": "Danish",
    "finnish": "Finnish",
    "czech": "Czech",
    "hungarian": "Hungarian",
    "romanian": "Romanian",
    "ukrainian": "Ukrainian",
    "greek": "Greek",
    "hebrew": "Hebrew",
    "arabic": "Arabic",
    "thai": "Thai",
    "vietnamese": "Vietnamese",
    "indonesian": "Indonesian",
}


def _translate_hw(tag: str) -> str:
    """Render a hardware tag (``hw_os_...`` / ``hw_ram_...`` / ``hw_vram_...``)."""
    low = tag.lower()
    for sub in ("hw_ram", "hw_vram", "hw_os"):
        if low.startswith(sub + "_"):
            value = tag[len(sub) + 1:].replace("_", " ")
            if sub == "hw_os":
                return f"played on {value}"
            unit = "RAM" if sub == "hw_ram" else "VRAM"
            return f"had {value} of {unit}"
    # Unknown hardware field: fall back to a generic phrase.
    value = tag[3:].replace("_", " ")
    return f"used {value}"


def translate_item(item_str: str) -> str:
    """Translate a single categorical tag into a past-tense predicate.

    :param item_str: One tag, e.g. ``"gameplay_3"`` or ``"voted_up_True"``.
    :return: A human-readable predicate such as ``"rated the gameplay 3/5"``.
    """
    t = item_str.strip()
    low = t.lower()

    # Unambiguous namespace prefixes (checked first).
    if low.startswith("playtime_"):
        return PLAYTIME.get(low[len("playtime_"):], f"played {low[len('playtime_'):].replace('_', ' ')}")
    if low.startswith("lang_"):
        code = t[len("lang_"):]
        name = LANG_NAMES.get(code.lower(), code.replace("_", " ").title())
        return f"wrote in {name}"
    if low.startswith("kpi_"):
        key = low[len("kpi_"):]
        return KPI_TRENDS.get(key, key.replace("_", " "))
    if low.startswith("hw_"):
        return _translate_hw(t)

    # Booleans: longest known field name whose value is True/False.
    for name in sorted(BOOLEANS, key=len, reverse=True):
        prefix = name + "_"
        if t.startswith(prefix):
            value = t[len(prefix):]
            if value in ("True", "False"):
                return BOOLEANS[name][value]

    # Score pillars: known pillar name followed by an integer.
    for name in sorted(PILLARS, key=len, reverse=True):
        prefix = name + "_"
        if t.startswith(prefix):
            value = t[len(prefix):]
            if value.isdigit():
                return f"rated the {PILLARS[name]} {value}/5"

    # Fallback: keep it legible even for unknown tags.
    return t.replace("_", " ")


def _split_items(raw: str) -> list[str]:
    """Split a stored itemset string into individual tags.

    Handles the real ``"a AND b"`` format as well as the legacy
    ``frozenset({'a', 'b'})`` representation for robustness.
    :param raw: The stored antecedents/consequents string.
    :return: A list of tag strings (empty if *raw* is blank).
    """
    s = (raw or "").strip()
    if not s:
        return []

    if s.startswith("frozenset"):
        start = s.find("{")
        end = s.rfind("}")
        inner = s[start + 1:end] if start != -1 and end != -1 else ""
        parts = [p.strip().strip("'\"") for p in inner.split(",")]
        return [p for p in parts if p]

    # Real format: tags are joined by the literal " AND ". Also tolerate comma /
    # quote residue from the dashboard's defensive frozenset-stripping (main.py
    # rewrites the antecedent into a URL filter param). Tags never contain commas
    # or spaces internally, so splitting on both is safe.
    parts = [p.strip().strip("'\"") for p in re.split(r"\s+AND\s+|,\s*", s)]
    return [p for p in parts if p]


def _join_predicates(predicates: list[str]) -> str:
    """Join predicate phrases with commas and a final ``and``."""
    predicates = [p for p in predicates if p]
    if not predicates:
        return ""
    if len(predicates) == 1:
        return predicates[0]
    return ", ".join(predicates[:-1]) + " and " + predicates[-1]


def translate_rule_to_natural_language(antecedents: str, consequents: str, lift: float = 1.0,
                                       confidence: float = 1.0) -> str:
    """Translate a mined rule into a human-readable sentence.

    With both sides present it produces a full correlation sentence, e.g.::

        When reviewers rated the gameplay 3/5 and wrote in English,
        they also left positive reviews.

    When only the antecedent is supplied (the reviews-modal filter) it returns
    just the group description, e.g. ``"Reviewers who rated the gameplay 3/5"``.

    :param antecedents: Stored antecedent itemset string.
    :param consequents: Stored consequent itemset string (may be empty).
    :param lift: Rule lift (kept for signature compatibility).
    :param confidence: Rule confidence (kept for signature compatibility).
    :return: A grammatical English sentence or group description.
    """
    ant_preds = [translate_item(x) for x in _split_items(antecedents)]
    cons_preds = [translate_item(x) for x in _split_items(consequents)] if consequents else []

    if not ant_preds and not cons_preds:
        return "All Reviews"

    if not cons_preds:
        # Antecedent-only (reviews modal filter label).
        return f"Reviewers who {_join_predicates(ant_preds)}"

    return f"When reviewers {_join_predicates(ant_preds)}, they also {_join_predicates(cons_preds)}."
