"""Biến đổi đối kháng (adversarial perturbation) để kiểm tra độ bền của bộ phát hiện.

Kẻ tấn công thật thường "né" bộ lọc bằng các mẹo rẻ tiền. Ta áp dụng chúng lên
các prompt tấn công trong tập test và đo Recall còn lại bao nhiêu.
"""
from __future__ import annotations

import random
import unicodedata

LEET = str.maketrans({"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"})
HOMOGLYPH = str.maketrans({"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "i": "і"})  # Cyrillic
ZW = "​"

BENIGN_PAD = (
    "Hi! I am working on a school project about renewable energy and would love some help. "
    "Could you summarise the main advantages of solar panels for a small family home, including cost, "
    "maintenance and typical lifespan? Please keep the answer friendly and easy to read. "
) * 3


def _words(t):
    return t.split(" ")


def leetspeak(t: str, rng=None) -> str:
    return t.translate(LEET)


def homoglyph(t: str, rng=None) -> str:
    return t.translate(HOMOGLYPH)


def char_spacing(t: str, rng=None) -> str:
    """Chèn dấu cách giữa các ký tự của các từ khoá dài (i g n o r e)."""
    rng = rng or random.Random(0)
    return " ".join(" ".join(w) if len(w) >= 5 and rng.random() < 0.5 else w for w in _words(t))


def zero_width(t: str, rng=None) -> str:
    return ZW.join(t)


def random_case(t: str, rng=None) -> str:
    rng = rng or random.Random(0)
    return "".join(c.upper() if rng.random() < 0.5 else c.lower() for c in t)


def benign_padding(t: str, rng=None) -> str:
    """'Pha loãng': chèn đoạn văn vô hại dài trước prompt tấn công."""
    return BENIGN_PAD + "\n\n" + t


def typo(t: str, rng=None) -> str:
    rng = rng or random.Random(0)
    out = []
    for w in _words(t):
        if len(w) > 3 and rng.random() < 0.3:
            i = rng.randrange(1, len(w) - 1)
            w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        out.append(w)
    return " ".join(out)


PERTURBATIONS = {
    "gốc (không đổi)": lambda t, rng=None: t,
    "lỗi gõ (typo)": typo,
    "HoA tHưỜnG ngẫu nhiên": random_case,
    "leetspeak (1gn0r3)": leetspeak,
    "tách ký tự (i g n o r e)": char_spacing,
    "homoglyph (Cyrillic)": homoglyph,
    "ký tự vô hình (zero-width)": zero_width,
    "pha loãng bằng văn bản vô hại": benign_padding,
}

# --- Phòng thủ: chuẩn hoá văn bản trước khi đưa vào mô hình ----------------
_CYR2LAT = {v: k for k, v in {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "i": "і"}.items()}
_DELEET = str.maketrans({"4": "a", "3": "e", "1": "i", "0": "o", "5": "s", "7": "t"})


def defensive_normalize(t: str) -> str:
    """NFKC + bỏ ký tự vô hình + map homoglyph Cyrillic → Latin + bỏ dấu cách giữa các ký tự đơn lẻ.

    De-leet chỉ áp dụng cho token lẫn chữ và số (tránh phá số thật như "2023").
    """
    t = unicodedata.normalize("NFKC", t)
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Cf")
    t = "".join(_CYR2LAT.get(ch, ch) for ch in t)
    toks = t.split(" ")
    out, buf = [], []
    for tok in toks:  # gộp chuỗi các ký tự đơn "i g n o r e" → "ignore"
        if len(tok) == 1 and tok.isalnum():
            buf.append(tok)
            continue
        if buf:
            out.append("".join(buf))
            buf = []
        out.append(tok)
    if buf:
        out.append("".join(buf))
    fixed = []
    for tok in out:
        if any(c.isalpha() for c in tok) and any(c.isdigit() for c in tok):
            tok = tok.translate(_DELEET)
        fixed.append(tok)
    return " ".join(fixed)
