"""
model/glossary.py
=================
Character-level glossary mapping MCS Chinese characters to English words.

Used by generate.py and other scripts to display model output in English
for readers who do not read Chinese.
"""

# Character-level mapping
CHAR_TO_EN = {
    # Animals
    "虎": "tiger",
    "狐": "fox",
    "狗": "dog",
    "鸡": "chicken",
    "羊": "sheep",
    "牛": "cow",
    "虫": "worm",

    # Plants and food
    "草": "grass",
    "麦": "wheat",
    "肉": "meat",

    # Verbs
    "吃": "eat",
    "追": "chase",
    "赶": "drive",
    "抓": "catch",

    # Adjectives
    "肥": "fat",
    "饿": "hungry",

    # Punctuation
    "。": ".",
    "，": ",",
    "、": ",",
    "：": ":",
    "；": ";",
}


def char_to_en(ch):
    """Return the English gloss for a single Chinese character."""
    return CHAR_TO_EN.get(ch)


def gloss_text(text):
    """
    Convert a Chinese string into an English gloss.

    Example:
        "虎吃羊。" -> "tiger eat sheep."
    """
    tokens = []
    for ch in text:
        if ch.isspace():
            # Preserve spaces, tabs, and newlines without adding '?'
            tokens.append(ch)
            continue
        en = CHAR_TO_EN.get(ch)
        if en is None:
            # Unknown character: keep it and mark with '?'
            tokens.append(f"{ch}?")
        else:
            tokens.append(en)
    return " ".join(tokens)