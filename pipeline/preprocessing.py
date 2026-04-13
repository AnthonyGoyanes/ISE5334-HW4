"""
pipeline/preprocessing.py
PP-A through PP-E* — point cloud preprocessing modules.

Every public function signature:
    preprocess(cloud: np.ndarray, **params) -> Tuple[np.ndarray, Optional[PlaneModel]]

The returned PlaneModel is used downstream by FE-K / FE-L / FE-M / FE-N.
PP-A, PP-B, PP-D return plane_model=None.
PP-C, PP-E return a PlaneModel but the cloud is NOT canonically aligned.
PP-C*, PP-E* return a PlaneModel AND the cloud is rotated to the plane-aligned frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# PlaneModel dataclass
# ---------------------------------------------------------------------------

@dataclass
class PlaneModel:
    """Fitted plane: normal · x + d = 0"""
    normal: np.ndarray   # unit normal (3,)
    centroid: np.ndarray  # centroid of inlier points (3,)
    d: float              # scalar offset


# ---------------------------------------------------------------------------
# Primitive operations
# ---------------------------------------------------------------------------

def _fit_ransac_plane(
    points: np.ndarray,
    distance_threshold: float = 0.02,
    num_iterations: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[PlaneModel, np.ndarray]:
    """RANSAC plane fitting.  Returns (PlaneModel, inlier_indices)."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(points)
    best_inliers: np.ndarray = np.array([], dtype=int)
    best_model: Optional[PlaneModel] = None

    for _ in range(num_iterations):
        idx = rng.choice(n, 3, replace=False)
        s = points[idx]
        v1, v2 = s[1] - s[0], s[2] - s[0]
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal = normal / norm
        d = float(-normal @ s[0])
        dists = np.abs(points @ normal + d)
        inliers = np.where(dists < distance_threshold)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            centroid = points[inliers].mean(axis=0)
            best_model = PlaneModel(normal=normal, centroid=centroid, d=d)

    if best_model is None:
        # Degenerate: return a dummy horizontal plane
        best_model = PlaneModel(
            normal=np.array([0.0, 0.0, 1.0]),
            centroid=points.mean(axis=0),
            d=float(-points.mean(axis=0)[2]),
        )
        best_inliers = np.arange(n)

    return best_model, best_inliers


def _normalize_unit_sphere(points: np.ndarray) -> np.ndarray:
    """Translate centroid to origin, scale to unit sphere."""
    pts = points - points.mean(axis=0)
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 1e-10:
        pts = pts / scale
    return pts.astype(np.float32)


def _voxel_downsample(points: np.ndarray, voxel_size: float = 0.05) -> np.ndarray:
    """Simple voxel-grid downsampling: average points per cell."""
    mins = points.min(axis=0)
    voxel_idx = ((points - mins) / voxel_size).astype(np.int32)
    voxel_map: dict = {}
    for i, key in enumerate(map(tuple, voxel_idx)):
        voxel_map.setdefault(key, []).append(i)
    result = np.array(
        [points[v].mean(axis=0) for v in voxel_map.values()], dtype=np.float32
    )
    return result


def _largest_component(
    points: np.ndarray, eps: float = 0.1, min_samples: int = 5
) -> np.ndarray:
    """DBSCAN-based connected-component analysis; return largest cluster."""
    from sklearn.cluster import DBSCAN

    if len(points) < min_samples:
        return points
    lbl = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(points)
    valid = lbl[lbl >= 0]
    if len(valid) == 0:
        return points
    unique, counts = np.unique(valid, return_counts=True)
    biggest = unique[np.argmax(counts)]
    return points[lbl == biggest]


def _plane_to_z_rotation(normal: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix that maps `normal` onto +Z."""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(normal, z)
    c = float(normal @ z)
    if np.linalg.norm(v) < 1e-10:
        # Already aligned (or anti-aligned)
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([
        [0.0,  -v[2],  v[1]],
        [v[2],  0.0,  -v[0]],
        [-v[1],  v[0],  0.0],
    ])
    R = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
    return R


# ---------------------------------------------------------------------------
# PP module functions
# ---------------------------------------------------------------------------

def pp_a(cloud: np.ndarray, **_) -> Tuple[np.ndarray, None]:
    """PP-A: raw — no preprocessing."""
    return cloud.astype(np.float32), None


def pp_b(cloud: np.ndarray, **_) -> Tuple[np.ndarray, None]:
    """PP-B: normalize to unit sphere."""
    return _normalize_unit_sphere(cloud), None


def pp_c(
    cloud: np.ndarray,
    ransac_distance_threshold: float = 0.02,
    ransac_iterations: int = 1000,
    random_seed: int = 42,
    **_,
) -> Tuple[np.ndarray, PlaneModel]:
    """PP-C: RANSAC plane removal → normalize.  No coordinate alignment."""
    rng = np.random.default_rng(random_seed)
    model, inliers = _fit_ransac_plane(cloud, ransac_distance_threshold, ransac_iterations, rng)
    mask = np.ones(len(cloud), dtype=bool)
    mask[inliers] = False
    remaining = cloud[mask]
    if len(remaining) < 10:
        remaining = cloud  # fallback: keep all
    return _normalize_unit_sphere(remaining), model


def pp_c_star(
    cloud: np.ndarray,
    ransac_distance_threshold: float = 0.02,
    ransac_iterations: int = 1000,
    dbscan_eps: float = 0.1,
    dbscan_min_samples: int = 5,
    random_seed: int = 42,
    **_,
) -> Tuple[np.ndarray, PlaneModel]:
    """PP-C*: RANSAC → largest component → rotate to plane-canonical frame → normalize."""
    rng = np.random.default_rng(random_seed)
    model, inliers = _fit_ransac_plane(cloud, ransac_distance_threshold, ransac_iterations, rng)

    mask = np.ones(len(cloud), dtype=bool)
    mask[inliers] = False
    remaining = cloud[mask]
    if len(remaining) < 10:
        remaining = cloud

    remaining = _largest_component(remaining, eps=dbscan_eps, min_samples=dbscan_min_samples)
    if len(remaining) < 10:
        remaining = cloud[mask] if mask.sum() >= 10 else cloud

    # Rotate so plane normal → +Z
    R = _plane_to_z_rotation(model.normal)
    aligned = (R @ remaining.T).T

    # Update plane model to aligned coordinate frame
    aligned_normal = R @ model.normal
    aligned_centroid = R @ model.centroid
    aligned_model = PlaneModel(
        normal=aligned_normal,
        centroid=aligned_centroid,
        d=float(-aligned_normal @ aligned_centroid),
    )

    return _normalize_unit_sphere(aligned), aligned_model


def pp_d(
    cloud: np.ndarray,
    voxel_size: float = 0.05,
    **_,
) -> Tuple[np.ndarray, None]:
    """PP-D: normalize → voxel downsample."""
    normalized = _normalize_unit_sphere(cloud)
    downsampled = _voxel_downsample(normalized, voxel_size)
    return downsampled, None


def pp_e(
    cloud: np.ndarray,
    ransac_distance_threshold: float = 0.02,
    ransac_iterations: int = 1000,
    voxel_size: float = 0.05,
    random_seed: int = 42,
    **_,
) -> Tuple[np.ndarray, PlaneModel]:
    """PP-E: RANSAC plane removal → normalize → voxel downsample."""
    processed, model = pp_c(cloud, ransac_distance_threshold, ransac_iterations, random_seed)
    downsampled = _voxel_downsample(processed, voxel_size)
    return downsampled, model


def pp_e_star(
    cloud: np.ndarray,
    ransac_distance_threshold: float = 0.02,
    ransac_iterations: int = 1000,
    dbscan_eps: float = 0.1,
    dbscan_min_samples: int = 5,
    voxel_size: float = 0.05,
    random_seed: int = 42,
    **_,
) -> Tuple[np.ndarray, PlaneModel]:
    """PP-E*: RANSAC → largest component → plane-canonical frame → normalize → downsample."""
    processed, model = pp_c_star(
        cloud, ransac_distance_threshold, ransac_iterations,
        dbscan_eps, dbscan_min_samples, random_seed
    )
    downsampled = _voxel_downsample(processed, voxel_size)
    return downsampled, model


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PREPROCESSORS = {
    "PP-A":  pp_a,
    "PP-B":  pp_b,
    "PP-C":  pp_c,
    "PP-C*": pp_c_star,
    "PP-D":  pp_d,
    "PP-E":  pp_e,
    "PP-E*": pp_e_star,
}

# Which PP modules produce a valid PlaneModel (required for FE-K/L/M/N)
PLANE_MODEL_PP = {"PP-C", "PP-C*", "PP-E", "PP-E*"}

# Which PP modules produce a canonically aligned frame (best for FE-K/L/M/N)
CANONICAL_PP = {"PP-C*", "PP-E*"}
