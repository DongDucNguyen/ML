"""Biểu diễn văn bản thành vector đặc trưng.

* TF-IDF theo từ (1–2 gram): bắt cụm như "ignore previous", "developer mode".
* TF-IDF theo ký tự (3–5 gram, char_wb): bền hơn với lỗi chính tả, viết tắt,
  leetspeak ("1gn0re"), và không phụ thuộc ngôn ngữ (deepset có tiếng Đức).
* Đặc trưng thủ công (handcrafted): độ dài, tỉ lệ chữ hoa, ký tự đặc biệt,
  số dòng, từ khoá tấn công, dấu hiệu nhập vai, mã hoá base64...

Tất cả đều không âm (cần cho Naive Bayes) và được fit **bên trong** Pipeline
nên mỗi fold cross-validation tự học lại từ điển → không rò rỉ dữ liệu.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler

from . import config as C


def _truncate(texts):
    return [str(t)[: C.MAX_CHARS] for t in texts]


Truncate = FunctionTransformer(_truncate, validate=False)

# Từ khoá / mẫu thường gặp trong prompt injection & jailbreak
ATTACK_PATTERNS = {
    "kw_ignore": r"\b(ignore|disregard|forget|override|bypass)\b",
    "kw_previous": r"\b(previous|prior|above|earlier|initial)\s+(instructions?|prompts?|rules?|directions?)",
    "kw_system_prompt": r"system\s*prompt|hidden\s+(instructions?|prompt)|your\s+(instructions|rules|guidelines)",
    "kw_reveal": r"\b(reveal|leak|disclose|print|show|repeat)\b.{0,30}\b(password|secret|prompt|instructions?|key)\b",
    "kw_roleplay": r"\b(pretend|act as|you are now|roleplay|role-play|simulate|imagine you)\b",
    "kw_persona": r"\b(dan|do anything now|developer mode|jailbreak|jailbroken|aim|evil|unfiltered|uncensored)\b",
    "kw_no_rules": r"\b(no (rules|restrictions|limits|filters|ethics|morals)|without (any )?(restrictions|limitations|filters))\b",
    "kw_policy": r"\b(openai|content policy|guidelines|ethical|moral|policy)\b",
    "kw_never_refuse": r"(never|don'?t|do not|will not|won'?t)\s+(refuse|decline|say (no|sorry)|warn)",
    "kw_stay_in_char": r"stay in character|remain in character|break character",
    "kw_password": r"\b(password|passcode|secret (key|word))\b",
    "kw_two_responses": r"(two|2) (different )?(responses|answers|paragraphs)|\[(🔓|🔒|gpt|jailbreak|classic)",
}
_COMPILED = {k: re.compile(v, re.IGNORECASE) for k, v in ATTACK_PATTERNS.items()}
_B64 = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")
_URL = re.compile(r"https?://")


class HandcraftedFeatures(BaseEstimator, TransformerMixin):
    """Đặc trưng thống kê + từ khoá, dễ giải thích khi phân tích lỗi."""

    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self, input_features=None):
        return np.array(
            [
                "log_len_chars", "log_len_words", "n_lines", "upper_ratio", "digit_ratio",
                "punct_ratio", "non_ascii_ratio", "n_brackets", "n_quotes", "has_base64",
                "has_url", "has_code_block", "has_separator", "n_imperatives",
            ]
            + list(_COMPILED)
        )

    def transform(self, X):
        rows = []
        for t in X:
            t = str(t)
            n = max(len(t), 1)
            words = t.split()
            low = t.lower()
            feats = [
                np.log1p(len(t)),
                np.log1p(len(words)),
                min(t.count("\n"), 200) / 10.0,
                sum(c.isupper() for c in t) / n,
                sum(c.isdigit() for c in t) / n,
                sum((not c.isalnum()) and (not c.isspace()) for c in t) / n,
                sum(ord(c) > 127 for c in t) / n,
                min(t.count("[") + t.count("{") + t.count("<"), 100) / 10.0,
                min(t.count('"') + t.count("'"), 100) / 10.0,
                float(bool(_B64.search(t))),
                float(bool(_URL.search(t))),
                float("```" in t),
                float(bool(re.search(r"(^|\n)\s*(#{2,}|-{3,}|={3,})", t))),
                min(len(re.findall(r"\b(you must|you will|you should|always|never|do not|don't)\b", low)), 50) / 5.0,
            ]
            feats += [min(len(p.findall(t)), 20) / 2.0 for p in _COMPILED.values()]
            rows.append(feats)
        return np.asarray(rows, dtype=np.float32)


def word_tfidf(max_features: int = 50_000, min_df: int = 2) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.9,
        max_features=max_features,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b|[^\w\s]",  # giữ cả dấu câu như ':', '[', '#'
        dtype=np.float32,
    )


def char_tfidf(max_features: int = 100_000, min_df: int = 3) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=min_df,
        max_df=0.95,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )


def text_features(kind: str = "word+char+hand", word_max: int = 50_000, char_max: int = 100_000) -> Pipeline:
    """Tạo khối trích đặc trưng.

    kind: tổ hợp của "word", "char", "hand" nối bằng "+".
    """
    parts = []
    tokens = set(kind.split("+"))
    if "word" in tokens:
        parts.append(("word", word_tfidf(word_max)))
    if "char" in tokens:
        parts.append(("char", char_tfidf(char_max)))
    if "hand" in tokens:
        parts.append(("hand", Pipeline([("hc", HandcraftedFeatures()), ("scale", MaxAbsScaler())])))
    if not parts:
        raise ValueError(f"kind không hợp lệ: {kind}")
    return Pipeline([("truncate", Truncate), ("union", FeatureUnion(parts))])


def feature_names(feat_pipe: Pipeline) -> np.ndarray:
    """Tên đặc trưng (để giải thích hệ số Logistic / độ quan trọng của RF)."""
    union: FeatureUnion = feat_pipe.named_steps["union"]
    names = []
    for name, tr in union.transformer_list:
        if isinstance(tr, Pipeline):
            sub = tr.steps[0][1].get_feature_names_out()
        else:
            sub = tr.get_feature_names_out()
        names.extend(f"{name}:{s}" for s in sub)
    return np.asarray(names)
