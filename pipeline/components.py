"""
pipeline/imbalance.py   — IB-0 through IB-3
pipeline/selection.py   — FS-0 through FS-3
pipeline/classifiers.py — CL-1 through CL-7
pipeline/evaluation.py  — metrics

All four are defined in this single file to keep imports concise.
Each section is clearly delimited.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# ===========================================================================
# IMBALANCE  (IB)
# ===========================================================================

def apply_imbalance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    ib_key: str,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply imbalance correction to the training set only.

    IB-0 : none
    IB-1 : class_weight handled at classifier level — no-op here
    IB-2 : SMOTE
    IB-3 : SMOTE (class_weight handled at classifier level)
    """
    if ib_key in ("IB-0", "IB-1"):
        return X_train, y_train

    from imblearn.over_sampling import SMOTE

    try:
        sm = SMOTE(random_state=random_state, k_neighbors=min(5, int(y_train.sum()) - 1))
        X_res, y_res = sm.fit_resample(X_train, y_train)
        return X_res.astype(np.float32), y_res
    except Exception as exc:
        print(f"[imbalance] SMOTE failed ({exc}), returning original data.")
        return X_train, y_train


USES_CLASS_WEIGHT = {"IB-1", "IB-3"}


# ===========================================================================
# FEATURE SELECTION  (FS)
# ===========================================================================

class FeatureSelector:
    """Fits on X_train, transforms X_train / X_val / X_test."""

    def __init__(self, fs_key: str, fe_key: str, random_state: int = 42) -> None:
        self.fs_key = fs_key
        self.fe_key = fe_key
        self.random_state = random_state
        self._selector = None

    def fit_transform(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit on X_train, return (X_train_transformed, X_test_transformed)."""
        if self.fs_key == "FS-0":
            return X_train, X_test

        # Safety: PCA cannot be applied on top of an already-reduced FE
        if self.fs_key == "FS-3" and self.fe_key in ("FE-H", "FE-I", "FE-J", "FE-N"):
            return X_train, X_test

        if self.fs_key == "FS-1":
            from sklearn.feature_selection import SelectKBest, f_classif
            k = min(50, X_train.shape[1])
            sel = SelectKBest(f_classif, k=k)
            X_tr = sel.fit_transform(X_train, y_train).astype(np.float32)
            X_te = sel.transform(X_test).astype(np.float32)
            self._selector = sel
            return X_tr, X_te

        elif self.fs_key == "FS-2":
            from sklearn.feature_selection import RFE
            from sklearn.ensemble import RandomForestClassifier
            n_feat = max(10, X_train.shape[1] // 4)
            est = RandomForestClassifier(
                n_estimators=50, random_state=self.random_state, n_jobs=-1
            )
            sel = RFE(est, n_features_to_select=n_feat, step=0.2)
            X_tr = sel.fit_transform(X_train, y_train).astype(np.float32)
            X_te = sel.transform(X_test).astype(np.float32)
            self._selector = sel
            return X_tr, X_te

        elif self.fs_key == "FS-3":
            from sklearn.decomposition import PCA
            n_components = min(0.95, X_train.shape[1])
            pca = PCA(n_components=n_components, random_state=self.random_state)
            X_tr = pca.fit_transform(X_train).astype(np.float32)
            X_te = pca.transform(X_test).astype(np.float32)
            self._selector = pca
            return X_tr, X_te

        raise ValueError(f"Unknown FS key: {self.fs_key!r}")


# ===========================================================================
# CLASSIFIERS  (CL)
# ===========================================================================

def build_classifier(cl_key: str, ib_key: str, random_state: int = 42):
    """Build and return an sklearn-compatible classifier."""
    use_weight = ib_key in USES_CLASS_WEIGHT

    if cl_key == "CL-1":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced" if use_weight else None,
            random_state=random_state,
            n_jobs=-1,
        )

    elif cl_key == "CL-2":
        from sklearn.svm import SVC
        return SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced" if use_weight else None,
            random_state=random_state,
        )

    elif cl_key == "CL-3":
        from xgboost import XGBClassifier
        # XGBoost uses scale_pos_weight for imbalance
        scale_pos = None
        if use_weight:
            scale_pos = 1.5  # rough proxy; tuned in full hyperopt
        kwargs = dict(
            n_estimators=200,
            max_depth=6,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
        if scale_pos is not None:
            kwargs["scale_pos_weight"] = scale_pos
        return XGBClassifier(**kwargs)

    elif cl_key == "CL-4":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=200,
            random_state=random_state,
        )

    elif cl_key == "CL-5":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=300,
            random_state=random_state,
        )

    elif cl_key == "CL-6":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced" if use_weight else None,
            random_state=random_state,
        )

    elif cl_key == "CL-7":
        from sklearn.neighbors import KNeighborsClassifier
        # KNN does not support class_weight natively
        return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

    raise ValueError(f"Unknown CL key: {cl_key!r}")


# ===========================================================================
# EVALUATION  (EV)
# ===========================================================================

def evaluate(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Compute classification metrics.

    Returns a dict with:
        accuracy, f1_macro, f1_feasible, f1_infeasible,
        precision_infeasible, recall_infeasible,
        roc_auc, tn, fp, fn, tp
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = model.predict(X_test)

    # Probability for ROC-AUC
    try:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)
        roc_auc = float(roc_auc_score(y_test, y_prob))
    except Exception:
        roc_auc = float("nan")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "accuracy":             float(accuracy_score(y_test, y_pred)),
        "f1_macro":             float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_feasible":          float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "f1_infeasible":        float(f1_score(y_test, y_pred, pos_label=0, zero_division=0)),
        "precision_infeasible": float(precision_score(y_test, y_pred, pos_label=0, zero_division=0)),
        "recall_infeasible":    float(recall_score(y_test, y_pred, pos_label=0, zero_division=0)),
        "roc_auc":              roc_auc,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
