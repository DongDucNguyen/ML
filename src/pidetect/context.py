"""Đặc trưng ngữ cảnh cho SPML: mối quan hệ giữa *user prompt* và *system prompt*.

Prompt injection là khái niệm phụ thuộc ngữ cảnh: cùng một câu "hãy nói
chuyện thô lỗ" là hợp lệ với chatbot hài nhưng là tấn công với chatbot y tế.
Các đặc trưng ở đây đo mức độ user prompt "lệch" khỏi system prompt.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from .features import text_features

_RULE = re.compile(r"\b(never|must|always|do not|don't|avoid|should not|refrain|only)\b", re.I)
_REDEFINE = re.compile(r"\b(you are|your role|you will now|from now on|as an? (ai|assistant|bot))\b", re.I)


class ContextFeatures(BaseEstimator, TransformerMixin):
    """Nhận DataFrame 2 cột [text, system_prompt] → vector đặc trưng ngữ cảnh."""

    def fit(self, X, y=None):
        corpus = list(X.iloc[:, 0].astype(str)) + list(X.iloc[:, 1].astype(str).unique())
        self.vec_ = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True).fit(corpus)
        return self

    def get_feature_names_out(self, input_features=None):
        return np.array(["ctx_cosine", "ctx_word_overlap", "ctx_len_ratio", "ctx_n_rules_system",
                         "ctx_n_rules_user", "ctx_redefine_role", "ctx_has_hash_sep", "ctx_user_rule_density"])

    def transform(self, X):
        u = X.iloc[:, 0].astype(str).tolist()
        s = X.iloc[:, 1].astype(str).tolist()
        U = self.vec_.transform(u)
        S = self.vec_.transform(s)
        cos = np.asarray(U.multiply(S).sum(1)).ravel()
        feats = []
        for ui, si, c in zip(u, s, cos):
            wu, ws = set(re.findall(r"\w+", ui.lower())), set(re.findall(r"\w+", si.lower()))
            overlap = len(wu & ws) / max(len(wu), 1)
            nr_u = len(_RULE.findall(ui))
            feats.append([
                c, overlap, np.log1p(len(ui)) - np.log1p(len(si)), len(_RULE.findall(si)) / 10,
                nr_u / 5, float(bool(_REDEFINE.search(ui))), float("###" in ui),
                nr_u / max(len(ui.split()), 1) * 10,
            ])
        return np.asarray(feats, dtype=np.float32)


def user_plus_context_features(kind: str = "word+char+hand", **kw) -> ColumnTransformer:
    """Đặc trưng = TF-IDF(user prompt) ⊕ đặc trưng ngữ cảnh(user, system)."""
    return ColumnTransformer(
        [
            ("user", text_features(kind, **kw), "text"),
            ("ctx", Pipeline([("cf", ContextFeatures()), ("scale", MaxAbsScaler())]), ["text", "system_prompt"]),
        ],
        sparse_threshold=1.0,
    )


def user_only_features(kind: str = "word+char+hand", **kw) -> ColumnTransformer:
    return ColumnTransformer([("user", text_features(kind, **kw), "text")], sparse_threshold=1.0)
