"""Bước 7 (bonus) – Sentence embeddings + các mô hình đã học.

Thay TF-IDF bằng vector ngữ nghĩa từ mô hình transformer đã huấn luyện sẵn
(không fine-tune), rồi dùng Logistic / KNN / SVM-RBF / LightGBM như bình thường.
Câu hỏi: biểu diễn ngữ nghĩa có giúp *tổng quát hoá* sang dữ liệu ngoài miền tốt hơn TF-IDF?

Chạy:
  python scripts/07_bonus_embeddings.py                                  # all-MiniLM-L6-v2 (nhẹ, chạy được CPU)
  python scripts/07_bonus_embeddings.py --encoder intfloat/multilingual-e5-base --batch 128   # GPU
"""
import argparse
import time

import _common  # noqa: F401
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pidetect import bonus as B
from pidetect import config as C
from pidetect import data as D

ap = argparse.ArgumentParser()
ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--max-len", type=int, default=256)
ap.add_argument("--max-train", type=int, default=None, help="giới hạn số mẫu train (chạy thử)")
ap.add_argument("--max-eval", type=int, default=None, help="giới hạn số mẫu mỗi tập đánh giá")
args = ap.parse_args()

from sentence_transformers import SentenceTransformer  # noqa: E402

device = B.pick_device()
tr, va, evals = B.combined_frames()
if args.max_train:
    tr = D.maybe_subsample(tr, args.max_train)
    va = D.maybe_subsample(va, max(500, args.max_train // 4))
evals = B.subsample_evals(evals, args.max_eval)

enc = SentenceTransformer(args.encoder, device=str(device))
enc.max_seq_length = args.max_len
prefix = "query: " if "e5" in args.encoder.lower() else ""
short = args.encoder.split("/")[-1]
cache = C.DATA_DIR / "cache" / "emb"
cache.mkdir(parents=True, exist_ok=True)


def embed(frame, tag):
    f = cache / f"{short}__{tag}__{len(frame)}.npy"
    if f.exists():
        return np.load(f)
    t0 = time.time()
    X = enc.encode([prefix + t[: C.MAX_CHARS] for t in frame["text"]], batch_size=args.batch,
                   normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
    print(f"  embed {tag}: {len(frame)} câu, {time.time() - t0:.0f}s")
    np.save(f, X)
    return X


X_tr, X_va = embed(tr, "train"), embed(va, "val")
X_ev = {ev: embed(fr, ev) for ev, fr in evals.items()}

CLFS = {
    "logreg": (LogisticRegression(max_iter=3000), {"clf__C": [0.3, 1, 3, 10], "clf__class_weight": [None, "balanced"]}),
    "knn": (KNeighborsClassifier(), {"clf__n_neighbors": [5, 15, 31], "clf__weights": ["distance"]}),
    "svm_rbf": (SVC(kernel="rbf"), {"clf__C": [1, 10], "clf__gamma": ["scale"]}),
}
try:
    from lightgbm import LGBMClassifier

    CLFS["lgbm"] = (LGBMClassifier(n_estimators=400, learning_rate=0.05, verbose=-1, random_state=C.SEED, force_row_wise=True),
                    {"clf__num_leaves": [15, 63]})
except ImportError:
    pass

for key, (clf, grid) in CLFS.items():
    steps = [("scale", StandardScaler())] if key in {"svm_rbf", "logreg"} else []
    pipe = Pipeline(steps + [("clf", clf)])
    Xfit, yfit = X_tr, tr["label"].values
    if key == "svm_rbf" and len(Xfit) > 12000:  # SVM kernel O(n²) → lấy mẫu con
        idx = D.maybe_subsample(tr.reset_index().rename(columns={"index": "i"}), 12000)["i"].values
        Xfit, yfit = X_tr[idx], yfit[idx]
    gs = GridSearchCV(pipe, grid, scoring="average_precision", cv=StratifiedKFold(3, shuffle=True, random_state=C.SEED),
                      n_jobs=1 if key == "lgbm" else -1)
    gs.fit(Xfit, yfit)
    m = gs.best_estimator_
    sc = (lambda X: m.predict_proba(X)[:, 1]) if hasattr(m, "predict_proba") and key != "svm_rbf" else m.decision_function
    print(f"\n=== Embedding {short} + {key}  (CV PR-AUC={gs.best_score_:.3f}, {gs.best_params_})")
    B.evaluate_scores(f"emb_{short}_{key}", va["label"].values, sc(X_va), {ev: sc(X) for ev, X in X_ev.items()},
                      evals, extra={"encoder": args.encoder, "best_params": str(gs.best_params_),
                                    "cv_pr_auc": gs.best_score_})
