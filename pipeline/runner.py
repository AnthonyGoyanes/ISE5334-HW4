"""
pipeline/runner.py
Orchestrates preprocessing, feature extraction, and ML for all selected pipelines.

Key optimisations
-----------------
1. Preprocessing cache  — each unique PP module is run once across all clouds.
2. Feature cache        — each unique (PP, FE) combination is extracted once.
3. AE pre-training      — autoencoder is trained once and shared across all
                          AE-group pipelines.
4. FE-H PCA             — fitted on X_train only, applied to all splits, per pipeline.
"""
from __future__ import annotations

import time
import traceback
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pipeline.catalog import PIPELINE_CATALOG_DICT, PipelineConfig
from pipeline.components import (
    apply_imbalance,
    build_classifier,
    evaluate,
    FeatureSelector,
)
from pipeline.features import extract_all, PLANE_REQUIRED_FE
from pipeline.preprocessing import PREPROCESSORS


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------

DEFAULT_PARAMS: dict = {
    "ransac_distance_threshold": 0.02,
    "ransac_iterations":         1000,
    "voxel_size":                0.05,
    "dbscan_eps":                0.1,
    "dbscan_min_samples":        5,
    "ae_n_points":               256,
    "ae_latent_dim":             64,
    "ae_epochs":                 50,
    "random_seed":               42,
    "test_size":                 0.30,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _preprocess_all(
    clouds: List[np.ndarray],
    pp_key: str,
    params: dict,
    progress_callback: Optional[Callable] = None,
) -> Tuple[List[np.ndarray], list]:
    """Run pp_key on every cloud.  Returns (processed_clouds, plane_models)."""
    fn = PREPROCESSORS[pp_key]
    processed, plane_models = [], []
    n = len(clouds)
    for i, cloud in enumerate(clouds):
        try:
            proc, pm = fn(cloud, **params)
        except Exception as exc:
            print(f"[preprocess] PP={pp_key} cloud {i} failed: {exc}")
            proc, pm = cloud.astype(np.float32), None
        processed.append(proc)
        plane_models.append(pm)
        if progress_callback:
            progress_callback("preprocess", pp_key, i + 1, n)
    return processed, plane_models


def _scale(X_train: np.ndarray, X_test: np.ndarray):
    sc = StandardScaler()
    return (
        sc.fit_transform(X_train).astype(np.float32),
        sc.transform(X_test).astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipelines(
    clouds: List[np.ndarray],
    labels: np.ndarray,
    pipeline_ids: List[str],
    params: Optional[dict] = None,
    progress_callback: Optional[Callable] = None,
) -> pd.DataFrame:
    """
    Run all selected pipelines and return a results DataFrame.

    Parameters
    ----------
    clouds           : list of raw (N_i, 3) point clouds
    labels           : int32 array, 1=feasible / 0=infeasible
    pipeline_ids     : list of pipeline IDs to run (e.g. ['P-001', 'P-020'])
    params           : override DEFAULT_PARAMS
    progress_callback: callable(stage, detail, current, total)

    Returns
    -------
    pd.DataFrame sorted by f1_infeasible descending
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    seed = int(p["random_seed"])

    # Resolve configs
    configs: Dict[str, PipelineConfig] = {
        pid: PIPELINE_CATALOG_DICT[pid]
        for pid in pipeline_ids
        if pid in PIPELINE_CATALOG_DICT
    }
    if not configs:
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Stage 1: Identify unique PP modules and preprocess
    # ------------------------------------------------------------------
    pp_set = {cfg.pp for cfg in configs.values()}
    preprocess_cache: Dict[str, Tuple[List, list]] = {}

    for pp_key in pp_set:
        if progress_callback:
            progress_callback("preprocess_start", pp_key, 0, len(clouds))
        preprocess_cache[pp_key] = _preprocess_all(
            clouds, pp_key, p,
            progress_callback=progress_callback,
        )

    # ------------------------------------------------------------------
    # Stage 2: Autoencoder pre-training (once, shared)
    # ------------------------------------------------------------------
    ae_model = None
    ae_embeddings_cache: Dict[str, np.ndarray] = {}   # keyed by pp_key

    ae_pipelines = [pid for pid, cfg in configs.items() if cfg.requires_ae]
    if ae_pipelines:
        from pipeline.autoencoder import SimpleAutoencoder

        # Use the PP of the first AE pipeline for training
        ae_pp = configs[ae_pipelines[0]].pp
        ae_clouds, _ = preprocess_cache[ae_pp]

        if progress_callback:
            progress_callback("ae_train", ae_pp, 0, 1)

        ae_model = SimpleAutoencoder(
            n_points=int(p["ae_n_points"]),
            latent_dim=int(p["ae_latent_dim"]),
        )
        ae_model.fit(ae_clouds, epochs=int(p["ae_epochs"]), random_state=seed)

        # Pre-compute embeddings for every PP variant that needs AE
        ae_pp_set = {configs[pid].pp for pid in ae_pipelines}
        for pp_key in ae_pp_set:
            ae_clouds_k, _ = preprocess_cache[pp_key]
            ae_embeddings_cache[pp_key] = ae_model.transform(ae_clouds_k)

        if progress_callback:
            progress_callback("ae_train", ae_pp, 1, 1)

    # ------------------------------------------------------------------
    # Stage 3: Feature extraction — cache per (PP, FE)
    # ------------------------------------------------------------------
    feature_cache: Dict[Tuple[str, str], np.ndarray] = {}

    fe_pp_pairs = {(cfg.pp, cfg.fe) for cfg in configs.values()}
    for pp_key, fe_key in fe_pp_pairs:
        cache_key = (pp_key, fe_key)
        if cache_key in feature_cache:
            continue

        proc_clouds, plane_models = preprocess_cache[pp_key]

        # Validation: FE needs plane_model but PP doesn't produce one
        if fe_key in PLANE_REQUIRED_FE and plane_models[0] is None:
            print(f"[runner] Skipping FE={fe_key} with PP={pp_key}: no PlaneModel")
            feature_cache[cache_key] = None
            continue

        ae_emb = ae_embeddings_cache.get(pp_key)

        if progress_callback:
            progress_callback("features", f"{pp_key}/{fe_key}", 0, len(proc_clouds))

        X = extract_all(proc_clouds, plane_models, fe_key, ae_embeddings=ae_emb)
        feature_cache[cache_key] = X

        if progress_callback:
            progress_callback("features", f"{pp_key}/{fe_key}", len(proc_clouds), len(proc_clouds))

    # ------------------------------------------------------------------
    # Stage 4: Run each pipeline
    # ------------------------------------------------------------------
    results = []
    total = len(configs)

    for i, (pid, cfg) in enumerate(configs.items()):
        if progress_callback:
            progress_callback("pipeline", pid, i, total)

        start = time.time()
        row: dict = {
            "id":    cfg.id,
            "name":  cfg.name,
            "group": cfg.group,
            "pp":    cfg.pp,
            "fe":    cfg.fe,
            "ib":    cfg.ib,
            "fs":    cfg.fs,
            "cl":    cfg.cl,
            "status": "ok",
            "error":  "",
        }

        try:
            X = feature_cache.get((cfg.pp, cfg.fe))
            if X is None:
                raise RuntimeError(f"Feature extraction failed for ({cfg.pp}, {cfg.fe})")

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, labels, test_size=p["test_size"],
                stratify=labels, random_state=seed,
            )

            # Imbalance (SMOTE on train only)
            X_train, y_train = apply_imbalance(X_train, y_train, cfg.ib, seed)

            # PCA for FE-H (fitted on train only)
            if cfg.apply_pca_post:
                from sklearn.decomposition import PCA
                n_comp = min(int(X_train.shape[1] * 0.95), X_train.shape[0] - 1, X_train.shape[1])
                pca = PCA(n_components=n_comp, random_state=seed)
                X_train = pca.fit_transform(X_train).astype(np.float32)
                X_test  = pca.transform(X_test).astype(np.float32)

            # Feature selection (fit on train, transform both)
            fs = FeatureSelector(cfg.fs, cfg.fe, random_state=seed)
            X_train, X_test = fs.fit_transform(X_train, y_train, X_test)

            # Scale
            X_train, X_test = _scale(X_train, X_test)

            # Build and train classifier
            clf = build_classifier(cfg.cl, cfg.ib, random_state=seed)
            clf.fit(X_train, y_train)

            # Evaluate
            metrics = evaluate(clf, X_test, y_test)
            row.update(metrics)

        except Exception as exc:
            row["status"] = "failed"
            row["error"]  = str(exc)
            traceback.print_exc()

        row["runtime_s"] = round(time.time() - start, 2)
        results.append(row)

    if progress_callback:
        progress_callback("done", "", total, total)

    df = pd.DataFrame(results)
    if "f1_infeasible" in df.columns:
        df = df.sort_values("f1_infeasible", ascending=False).reset_index(drop=True)
    return df
