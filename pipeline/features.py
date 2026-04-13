"""
pipeline/features.py
FE-A through FE-N — feature extraction modules.

Every extractor has the signature:
    extract(cloud: np.ndarray, plane_model: Optional[PlaneModel],
            ae_embeddings: Optional[np.ndarray] = None) -> np.ndarray

ae_embeddings is a pre-computed (1, latent_dim) array for the current cloud,
passed in for FE-I / FE-J / FE-N.

FE-H is handled separately in the runner (PCA must be fit on training only).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from pipeline.preprocessing import PlaneModel


# ---------------------------------------------------------------------------
# FE-A: Geometric statistics  (~47 features)
# ---------------------------------------------------------------------------

def _fe_a(cloud: np.ndarray, **__) -> np.ndarray:
    from scipy.stats import skew, kurtosis
    from sklearn.neighbors import NearestNeighbors

    feats: list = []

    # Per-axis stats (4 × 3 = 12)
    for ax in range(3):
        x = cloud[:, ax]
        feats += [float(np.mean(x)), float(np.std(x)),
                  float(skew(x)), float(kurtosis(x))]

    # Bounding box dimensions and aspect ratios (6)
    mins, maxs = cloud.min(0), cloud.max(0)
    dims = maxs - mins + 1e-10
    feats += dims.tolist()
    feats += [dims[0] / dims[1], dims[0] / dims[2], dims[1] / dims[2]]

    # PCA eigenvalues + shape descriptors (9)
    centred = cloud - cloud.mean(0)
    cov = np.cov(centred.T) + np.eye(3) * 1e-10
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigvals = np.maximum(eigvals, 0)
    total_var = eigvals.sum() + 1e-10
    feats += eigvals.tolist()
    feats += (eigvals / total_var).tolist()  # relative eigenvalues
    feats += [
        (eigvals[1] - eigvals[2]) / (eigvals[0] + 1e-10),  # planarity
        eigvals[2] / (eigvals[0] + 1e-10),                  # sphericity
        (eigvals[0] - eigvals[1]) / (eigvals[0] + 1e-10),  # linearity
    ]

    # Covariance upper triangle (6)
    feats += [cov[0, 0], cov[0, 1], cov[0, 2], cov[1, 1], cov[1, 2], cov[2, 2]]

    # Nearest-neighbour density (4)
    k = min(6, len(cloud) - 1)
    if k > 0:
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(cloud)
        dists, _ = nn.kneighbors(cloud)
        nd = dists[:, 1:].ravel()
        feats += [float(np.mean(nd)), float(np.std(nd)),
                  float(np.max(nd)), float(np.percentile(nd, 90))]
    else:
        feats += [0.0, 0.0, 0.0, 0.0]

    # Radial distance distribution (8)
    radii = np.linalg.norm(centred, axis=1)
    feats += [float(np.mean(radii)), float(np.std(radii)), float(np.max(radii))]
    feats += np.percentile(radii, [10, 25, 50, 75, 90]).tolist()

    # Point count (log) (1)
    feats += [float(np.log1p(len(cloud)))]

    # Convex hull volume (1, may fail)
    try:
        from scipy.spatial import ConvexHull
        feats += [float(ConvexHull(cloud).volume)]
    except Exception:
        feats += [0.0]

    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# FE-B: Voxel occupancy  (8³ = 512 features)
# ---------------------------------------------------------------------------

def _fe_b(cloud: np.ndarray, **__) -> np.ndarray:
    GRID = 8
    mins = cloud.min(0)
    maxs = cloud.max(0)
    ranges = maxs - mins + 1e-10
    normed = (cloud - mins) / ranges

    idx = np.clip((normed * GRID).astype(np.int32), 0, GRID - 1)
    flat = idx[:, 0] * GRID * GRID + idx[:, 1] * GRID + idx[:, 2]
    occ = np.bincount(flat, minlength=GRID ** 3).astype(np.float32)
    occ /= len(cloud) + 1e-10
    return occ


# ---------------------------------------------------------------------------
# FE-C: 2-D projection histograms  (3 × 16² = 768 features)
# ---------------------------------------------------------------------------

def _fe_c(cloud: np.ndarray, **__) -> np.ndarray:
    BINS = 16
    feats: list = []
    for ax1, ax2 in [(0, 1), (0, 2), (1, 2)]:
        proj = cloud[:, [ax1, ax2]]
        lo, hi = proj.min(0), proj.max(0)
        rng = hi - lo + 1e-10
        normed = (proj - lo) / rng
        h, _, _ = np.histogram2d(
            normed[:, 0], normed[:, 1],
            bins=BINS, range=[[0, 1], [0, 1]]
        )
        h = h / (h.sum() + 1e-10)
        feats.extend(h.ravel().tolist())
    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# FE-K: Plane-relative statistics  (~14 features)
# ---------------------------------------------------------------------------

def _fe_k(cloud: np.ndarray, plane_model: Optional[PlaneModel], **__) -> np.ndarray:
    if plane_model is None:
        return np.zeros(14, dtype=np.float32)

    from scipy.stats import skew, kurtosis

    # Signed height above plane
    heights = (cloud @ plane_model.normal + plane_model.d).astype(np.float64)

    h_feats = [
        float(np.mean(heights)),
        float(np.std(heights)),
        float(skew(heights)),
        float(kurtosis(heights)),
        float(np.max(heights)),
        float(np.min(heights)),
        *np.percentile(heights, [10, 50, 90]).tolist(),
    ]  # 9

    # Lateral spread in the plane
    proj_pts = cloud - np.outer(heights, plane_model.normal)
    lat_radii = np.linalg.norm(proj_pts - plane_model.centroid, axis=1)
    l_feats = [
        float(np.mean(lat_radii)),
        float(np.std(lat_radii)),
        float(np.max(lat_radii)),
    ]  # 3

    # Fraction of points above plane
    above = float((heights > 0).mean())

    # Density gradient: top half vs bottom half
    med = float(np.median(heights))
    ratio = float((heights > med).sum() / ((heights <= med).sum() + 1e-10))

    return np.array(h_feats + l_feats + [above, ratio], dtype=np.float32)


# ---------------------------------------------------------------------------
# Combination helpers
# ---------------------------------------------------------------------------

def _combine(*arrays: np.ndarray) -> np.ndarray:
    return np.concatenate(arrays, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Public extractor dispatch
# ---------------------------------------------------------------------------

def extract_features(
    cloud: np.ndarray,
    plane_model: Optional[PlaneModel],
    fe_key: str,
    ae_embedding: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Extract a 1-D feature vector for one cloud.

    Parameters
    ----------
    cloud        : preprocessed (N, 3) array
    plane_model  : PlaneModel or None
    fe_key       : one of FE-A … FE-N  (FE-H handled in runner)
    ae_embedding : (latent_dim,) array pre-computed by autoencoder
    """
    if fe_key == "FE-A":
        return _fe_a(cloud)
    elif fe_key == "FE-B":
        return _fe_b(cloud)
    elif fe_key == "FE-C":
        return _fe_c(cloud)
    elif fe_key == "FE-K":
        return _fe_k(cloud, plane_model)
    elif fe_key == "FE-D":
        return _combine(_fe_a(cloud), _fe_b(cloud))
    elif fe_key == "FE-E":
        return _combine(_fe_a(cloud), _fe_c(cloud))
    elif fe_key == "FE-F":
        return _combine(_fe_b(cloud), _fe_c(cloud))
    elif fe_key == "FE-G":
        return _combine(_fe_a(cloud), _fe_b(cloud), _fe_c(cloud))
    elif fe_key == "FE-H":
        # FE-H = FE-G; PCA is applied separately in the runner
        return _combine(_fe_a(cloud), _fe_b(cloud), _fe_c(cloud))
    elif fe_key == "FE-L":
        return _combine(_fe_a(cloud), _fe_k(cloud, plane_model))
    elif fe_key == "FE-M":
        return _combine(_fe_a(cloud), _fe_b(cloud), _fe_c(cloud),
                        _fe_k(cloud, plane_model))
    elif fe_key == "FE-I":
        if ae_embedding is None:
            raise ValueError("ae_embedding required for FE-I")
        return ae_embedding.astype(np.float32)
    elif fe_key == "FE-J":
        if ae_embedding is None:
            raise ValueError("ae_embedding required for FE-J")
        return _combine(ae_embedding, _fe_a(cloud))
    elif fe_key == "FE-N":
        if ae_embedding is None:
            raise ValueError("ae_embedding required for FE-N")
        return _combine(ae_embedding, _fe_a(cloud), _fe_k(cloud, plane_model))
    else:
        raise ValueError(f"Unknown FE key: {fe_key!r}")


def extract_all(
    clouds: list,
    plane_models: list,
    fe_key: str,
    ae_embeddings: Optional[np.ndarray] = None,
    progress_callback=None,
) -> np.ndarray:
    """Extract features for an entire dataset."""
    rows = []
    for i, (cloud, pm) in enumerate(zip(clouds, plane_models)):
        emb = ae_embeddings[i] if ae_embeddings is not None else None
        rows.append(extract_features(cloud, pm, fe_key, emb))
        if progress_callback:
            progress_callback(i + 1, len(clouds))
    return np.array(rows, dtype=np.float32)


# Keys that require a valid PlaneModel
PLANE_REQUIRED_FE = {"FE-K", "FE-L", "FE-M", "FE-N"}

# Keys that require AE embeddings
AE_REQUIRED_FE = {"FE-I", "FE-J", "FE-N"}

# Keys for which the runner must apply PCA post-extraction
PCA_FE = {"FE-H"}
