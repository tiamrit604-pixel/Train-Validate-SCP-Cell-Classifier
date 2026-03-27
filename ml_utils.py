"""
ml_utils.py
-----------
Exact sklearn pipelines + GridSearchCV configs from the notebook.
All four classifiers: KNN, SVM, Decision Tree, Logistic Regression.
All models saved individually. Robustness testing on unseen data supported.
Context: Classify 27 cells of a sandwich composite plate (SCP) as
         Good (bonded/healthy) or Bad (debonded/unhealthy) via acoustic percussion.
"""

import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)


# ---------------------------------------------------------------------------
# Grid definitions (exact from notebook)
# ---------------------------------------------------------------------------

CLASSIFIER_CONFIGS = {
    "KNN": {
        "pipeline": Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier())
        ]),
        "param_grid": {
            'knn__n_neighbors': [3, 5, 7, 9, 11, 13, 15],
            'knn__weights':     ['uniform', 'distance'],
            'knn__metric':      ['euclidean', 'manhattan']
        }
    },
    "SVM": {
        "pipeline": Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(probability=True))
        ]),
        "param_grid": [
            {'svm__kernel': ['linear'], 'svm__C': [0.01, 0.1, 1, 10, 100]},
            {'svm__kernel': ['rbf'],    'svm__C': [0.1, 1, 10, 100],
             'svm__gamma':  ['scale', 0.01, 0.1, 1]}
        ]
    },
    "Decision Tree": {
        "pipeline": Pipeline([
            ('dt', DecisionTreeClassifier(random_state=42))
        ]),
        "param_grid": {
            'dt__max_depth':         [None, 3, 5, 8, 12],
            'dt__min_samples_split': [2, 5, 10],
            'dt__min_samples_leaf':  [1, 2, 4],
            'dt__criterion':         ['gini', 'entropy']
        }
    },
    "Logistic Regression": {
        "pipeline": Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(max_iter=500))
        ]),
        "param_grid": {
            'lr__C':       [0.01, 0.1, 1, 10, 100],
            'lr__penalty': ['l2'],
            'lr__solver':  ['lbfgs', 'liblinear']
        }
    }
}

SCORING = {'accuracy': 'accuracy', 'f1': 'f1', 'precision': 'precision', 'recall': 'recall'}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODEL_DIR    = os.path.join(os.path.dirname(__file__), "models")
RESULTS_PATH = os.path.join(MODEL_DIR, "last_results.pkl")


def _model_path(clf_name: str) -> str:
    safe = clf_name.replace(" ", "_").lower()
    return os.path.join(MODEL_DIR, f"{safe}_model.pkl")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_all(
    X: np.ndarray,
    y: np.ndarray,
    X_unseen: np.ndarray,
    y_unseen: np.ndarray,
    selected_classifiers: list,
    cv_folds: int = 5,
    test_size: float = 0.30,
    random_state: int = 42,
    progress_callback=None
) -> dict:
    """
    Trains all selected classifiers with GridSearchCV.
    Evaluates each on: training set, validation (held-out), and unseen robustness set.
    All models saved individually to models/ directory.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    has_unseen = X_unseen is not None and len(X_unseen) > 0

    results = {
        "X_train": X_train, "X_val": X_val,
        "y_train": y_train, "y_val": y_val,
        "X_unseen": X_unseen, "y_unseen": y_unseen,
        "classifiers":    {},
        "best_model_name": None,
        "best_model":      None,
        "best_val_acc":   -1.0,
        "summary_df":     None,
        "has_unseen":     has_unseen,
        "trained_at":     datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    summary_rows = []
    total = len(selected_classifiers)
    os.makedirs(MODEL_DIR, exist_ok=True)

    for idx, clf_name in enumerate(selected_classifiers):
        if progress_callback:
            progress_callback(idx / total, f"Training {clf_name}  ({idx+1}/{total})...")

        cfg  = CLASSIFIER_CONFIGS[clf_name]
        grid = GridSearchCV(
            cfg["pipeline"], cfg["param_grid"],
            cv=cv_folds, scoring=SCORING, refit='accuracy',
            n_jobs=-1, error_score=0.0
        )
        grid.fit(X_train, y_train)

        est      = grid.best_estimator_
        cv_acc   = grid.best_score_

        y_tr_p   = est.predict(X_train)
        y_val_p  = est.predict(X_val)
        train_m  = _calc_metrics(y_train, y_tr_p)
        val_m    = _calc_metrics(y_val,   y_val_p)
        cm_train = confusion_matrix(y_train, y_tr_p)
        cm_val   = confusion_matrix(y_val,   y_val_p)

        if has_unseen:
            y_uns_p  = est.predict(X_unseen)
            unseen_m = _calc_metrics(y_unseen, y_uns_p)
            cm_unseen= confusion_matrix(y_unseen, y_uns_p)
        else:
            unseen_m  = {"accuracy": None, "precision": None, "recall": None, "f1": None}
            cm_unseen = None

        results["classifiers"][clf_name] = {
            "name":           clf_name,
            "grid":           grid,
            "estimator":      est,
            "best_params":    grid.best_params_,
            "cv_acc":         cv_acc,
            "train_metrics":  train_m,
            "val_metrics":    val_m,
            "unseen_metrics": unseen_m,
            "cm_train":       cm_train,
            "cm_val":         cm_val,
            "cm_unseen":      cm_unseen,
            "cv_results_df":  pd.DataFrame(grid.cv_results_)
        }

        # Save each model individually
        joblib.dump(est, _model_path(clf_name))

        if val_m["accuracy"] > results["best_val_acc"]:
            results["best_val_acc"]    = val_m["accuracy"]
            results["best_model_name"] = clf_name
            results["best_model"]      = est

        row = {
            "Model":          clf_name,
            "Best Params":    str(grid.best_params_),
            "CV Acc (%)":     _pct(cv_acc),
            "Train Acc (%)":  _pct(train_m["accuracy"]),
            "Val Acc (%)":    _pct(val_m["accuracy"]),
            "Unseen Acc (%)": _pct(unseen_m["accuracy"]) if has_unseen else "N/A",
            "Precision":      _r3(val_m["precision"]),
            "Recall":         _r3(val_m["recall"]),
            "F1":             _r3(val_m["f1"]),
            "Unseen F1":      _r3(unseen_m["f1"]) if has_unseen else "N/A",
            "Acc Drop (%)":   _pct(train_m["accuracy"] - val_m["accuracy"]),
        }
        summary_rows.append(row)

    if progress_callback:
        progress_callback(1.0, "Training complete!")

    results["summary_df"] = pd.DataFrame(summary_rows)

    # Persist slim results (no large arrays)
    slim = {k: v for k, v in results.items()
            if k not in ("X_train","X_val","y_train","y_val","X_unseen","y_unseen")}
    joblib.dump(slim, RESULTS_PATH)

    return results


def _calc_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy":  accuracy_score(y_true,  y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true,    y_pred, zero_division=0),
        "f1":        f1_score(y_true,        y_pred, zero_division=0),
    }

def _pct(v):   return round(v * 100, 2) if v is not None else "N/A"
def _r3(v):    return round(v, 3)       if v is not None else "N/A"


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_model(results: dict):
    os.makedirs(MODEL_DIR, exist_ok=True)
    slim = {k: v for k, v in results.items()
            if k not in ("X_train","X_val","y_train","y_val","X_unseen","y_unseen")}
    joblib.dump(slim, RESULTS_PATH)


def load_results():
    if not os.path.exists(RESULTS_PATH):
        return None
    return joblib.load(RESULTS_PATH)


def load_model_by_name(clf_name: str):
    path = _model_path(clf_name)
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def list_saved_models() -> list:
    if not os.path.exists(MODEL_DIR):
        return []
    return [n for n in CLASSIFIER_CONFIGS.keys() if os.path.exists(_model_path(n))]


def model_exists() -> bool:
    return len(list_saved_models()) > 0


# ---------------------------------------------------------------------------
# Prediction (used in Mode 2)
# ---------------------------------------------------------------------------

def predict_files(model, X: np.ndarray, meta: list) -> pd.DataFrame:
    """Predict and group by filename via majority vote."""
    if X.shape[0] == 0:
        return pd.DataFrame()

    preds = model.predict(X)
    try:
        probas = model.predict_proba(X)[:, 1]
    except Exception:
        probas = preds.astype(float)

    file_groups: dict = {}
    for i, m in enumerate(meta):
        if m.get("error"):
            continue
        fname = m["filename"]
        if fname not in file_groups:
            file_groups[fname] = {"filename": fname, "votes": [], "probas": [],
                                  "n_segments": m["n_segments"]}
        file_groups[fname]["votes"].append(int(preds[i]))
        file_groups[fname]["probas"].append(float(probas[i]))

    rows = []
    for fname, grp in file_groups.items():
        votes     = grp["votes"]
        good_frac = sum(votes) / len(votes)
        avg_proba = float(np.mean(grp["probas"]))
        final_lbl = 1 if good_frac >= 0.5 else 0
        rows.append({
            "Filename":   fname,
            "Prediction": "✅ Good (Bonded)" if final_lbl == 1 else "❌ Bad (Debonded)",
            "Label":      final_lbl,
            "Confidence": round(avg_proba * 100, 1),
            "Segments":   len(votes),
            "Good Votes": sum(votes),
            "Bad Votes":  len(votes) - sum(votes),
        })

    return pd.DataFrame(rows)
