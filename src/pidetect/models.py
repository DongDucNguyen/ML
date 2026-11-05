"""Bộ mô hình (model zoo) bám theo các slide đã học + mô hình bonus.

| Khoá           | Slide                    | Mô hình                                         |
|----------------|--------------------------|-------------------------------------------------|
| lpm            | 2 – Linear Regression    | Hồi quy tuyến tính (Ridge) dùng như bộ phân loại |
| logreg         | 3 – Logistic Regression  | Logistic Regression (L2, chọn C)                |
| knn            | 4 – KNN                  | TF-IDF → SVD → chuẩn hoá L2 → KNN (cosine)      |
| nb             | 5 – Probabilistic models | Complement / Multinomial Naive Bayes            |
| tree           | 6 – Decision Tree        | Cây quyết định (gini / entropy)                 |
| rf             | 6 – Random Forest        | Rừng ngẫu nhiên (bagging + chọn đặc trưng ngẫu nhiên) |
| svm_linear     | 7 – SVM                  | Linear SVM (hinge loss, chọn C)                 |
| svm_rbf        | 7 – SVM (kernel)         | SVD → chuẩn hoá → SVM kernel RBF (C, γ)         |
| lgbm (bonus)   | ngoài chương trình       | Gradient boosting LightGBM                      |
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from . import config as C
from .features import text_features


class LinearProbabilityModel(ClassifierMixin, BaseEstimator):
    """Hồi quy tuyến tính trên nhãn 0/1 (Linear Probability Model).

    Dùng để minh hoạ vì sao slide 3 chuyển sang Logistic: đầu ra của hồi quy
    tuyến tính *không bị chặn trong [0, 1]*. ``decision_function`` trả về giá
    trị thô; ``predict_proba`` cắt (clip) về [0, 1].
    Dùng Ridge (L2) vì TF-IDF có hàng trăm nghìn chiều > số mẫu → OLS thuần
    không xác định duy nhất nghiệm.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.reg_ = Ridge(alpha=self.alpha, random_state=C.SEED).fit(X, np.asarray(y, dtype=float))
        return self

    def decision_function(self, X):
        return self.reg_.predict(X)

    def predict_proba(self, X):
        p = np.clip(self.decision_function(X), 0.0, 1.0)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.decision_function(X) >= 0.5).astype(int)


@dataclass
class ModelSpec:
    key: str
    name_vi: str
    slide: str
    build: Callable[[], Pipeline]
    grid: dict = field(default_factory=dict)
    max_train: int | None = None  # giới hạn số mẫu train (mô hình chậm)
    bonus: bool = False


def _pipe(features, *steps):
    return Pipeline([("feats", features), *steps])


def _lgbm():
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31, colsample_bytree=0.3,
        subsample=0.8, subsample_freq=1, min_child_samples=10, reg_lambda=1.0,
        random_state=C.SEED, n_jobs=C.N_JOBS, verbose=-1, force_row_wise=True, max_bin=63,
    )


MODEL_SPECS: dict[str, ModelSpec] = {
    "lpm": ModelSpec(
        "lpm", "Hồi quy tuyến tính (LPM)", "Slide 2 – Linear Regression",
        lambda: _pipe(text_features(), ("clf", LinearProbabilityModel())),
        {"clf__alpha": [0.3, 1.0, 3.0]},
    ),
    "logreg": ModelSpec(
        "logreg", "Logistic Regression", "Slide 3 – Logistic Regression",
        lambda: _pipe(text_features(), ("clf", LogisticRegression(max_iter=3000))),
        {"clf__C": [1.0, 4.0, 16.0], "clf__class_weight": [None, "balanced"]},
    ),
    "knn": ModelSpec(
        "knn", "K-Nearest Neighbors", "Slide 4 – KNN",
        lambda: _pipe(
            text_features("word+char"),
            ("svd", TruncatedSVD(n_components=200, random_state=C.SEED)),
            ("norm", Normalizer()),  # vector đơn vị → khoảng cách Euclid ~ cosine
            ("clf", KNeighborsClassifier(n_jobs=C.N_JOBS)),
        ),
        {"clf__n_neighbors": [5, 15, 31], "clf__weights": ["uniform", "distance"]},
    ),
    "nb": ModelSpec(
        "nb", "Naive Bayes", "Slide 5 – Probabilistic models",
        lambda: _pipe(text_features(), ("clf", ComplementNB())),
        {"clf": [ComplementNB(), MultinomialNB()], "clf__alpha": [0.03, 0.1, 0.3]},
    ),
    "tree": ModelSpec(
        "tree", "Decision Tree", "Slide 6 – Decision Tree",
        lambda: _pipe(
            text_features("word+hand", word_max=20_000),
            ("clf", DecisionTreeClassifier(random_state=C.SEED)),
        ),
        {"clf__criterion": ["gini", "entropy"], "clf__max_depth": [10, 30, None], "clf__min_samples_leaf": [1, 5]},
    ),
    "rf": ModelSpec(
        "rf", "Random Forest", "Slide 6 – Random Forest",
        lambda: _pipe(
            text_features("word+char+hand", word_max=20_000, char_max=30_000),
            ("clf", RandomForestClassifier(n_estimators=300, max_features="sqrt", n_jobs=C.N_JOBS,
                                           random_state=C.SEED)),
        ),
        {"clf__min_samples_leaf": [1, 2], "clf__class_weight": [None, "balanced_subsample"]},
    ),
    "svm_linear": ModelSpec(
        "svm_linear", "Linear SVM", "Slide 7 – SVM",
        lambda: _pipe(text_features(), ("clf", LinearSVC(max_iter=5000, random_state=C.SEED))),
        {"clf__C": [0.1, 0.3, 1.0], "clf__class_weight": [None, "balanced"]},
    ),
    "svm_rbf": ModelSpec(
        "svm_rbf", "SVM kernel RBF", "Slide 7 – SVM (kernel)",
        lambda: _pipe(
            text_features("word+char"),
            ("svd", TruncatedSVD(n_components=300, random_state=C.SEED)),
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="rbf", random_state=C.SEED)),
        ),
        {"clf__C": [1.0, 10.0], "clf__gamma": ["scale", 0.0005]},
        max_train=6000,
    ),
    "lgbm": ModelSpec(
        "lgbm", "LightGBM (bonus)", "Ngoài chương trình – Gradient Boosting",
        lambda: _pipe(text_features("word+char+hand", word_max=20_000, char_max=30_000), ("clf", _lgbm())),
        {"clf__num_leaves": [15, 63]},
        bonus=True,
    ),
}

CORE_MODELS = [k for k, s in MODEL_SPECS.items() if not s.bonus]
ALL_MODELS = list(MODEL_SPECS)


def scores(model, X) -> np.ndarray:
    """Điểm số liên tục cho lớp dương (xác suất nếu có, ngược lại decision_function)."""
    clf = model.steps[-1][1] if isinstance(model, Pipeline) else model
    if hasattr(clf, "predict_proba") and not isinstance(clf, (LinearProbabilityModel, SVC)):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def is_probabilistic(model) -> bool:
    clf = model.steps[-1][1] if isinstance(model, Pipeline) else model
    return hasattr(clf, "predict_proba") and not isinstance(clf, (LinearProbabilityModel, SVC))
