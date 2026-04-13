"""
pipeline/ingestion.py
Load .ply point cloud files from feasible and infeasible directories.
"""
import os
import numpy as np
from typing import Callable, List, Optional, Tuple

SUPPORTED_EXT = (".ply",)


def _load_ply(filepath: str) -> np.ndarray:
    """Load a single .ply file and return (N, 3) float32 array."""
    try:
        import trimesh
        obj = trimesh.load(filepath, process=False)
        if isinstance(obj, trimesh.Scene):
            vertices = np.vstack([g.vertices for g in obj.geometry.values()])
        elif hasattr(obj, "vertices"):
            vertices = np.array(obj.vertices)
        else:
            raise ValueError("Loaded object has no vertices attribute.")
        if vertices.shape[0] == 0:
            raise ValueError("Empty point cloud.")
        return vertices.astype(np.float32)
    except Exception as exc:
        raise RuntimeError(f"Failed to load {filepath}: {exc}") from exc


def load_dataset(
    feasible_dir: str,
    infeasible_dir: str,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
    """
    Load all .ply files from two directories.

    Returns
    -------
    clouds     : list of (N_i, 3) float32 arrays
    labels     : int32 array, 1 = feasible, 0 = infeasible
    filenames  : list of filenames matching each cloud
    """
    clouds: List[np.ndarray] = []
    labels: List[int] = []
    filenames: List[str] = []
    errors: List[str] = []

    def _load_dir(directory: str, label: int) -> None:
        files = sorted(
            f for f in os.listdir(directory)
            if f.lower().endswith(SUPPORTED_EXT)
        )
        for fname in files:
            fpath = os.path.join(directory, fname)
            try:
                cloud = _load_ply(fpath)
                clouds.append(cloud)
                labels.append(label)
                filenames.append(fname)
            except RuntimeError as exc:
                errors.append(str(exc))
            if progress_callback:
                progress_callback(fname)

    _load_dir(feasible_dir, label=1)
    _load_dir(infeasible_dir, label=0)

    if errors:
        print(f"[ingestion] {len(errors)} file(s) failed to load:")
        for e in errors:
            print(f"  {e}")

    return clouds, np.array(labels, dtype=np.int32), filenames


def dataset_summary(
    clouds: List[np.ndarray], labels: np.ndarray, filenames: List[str]
) -> dict:
    """Return a dict of summary statistics for display in the UI."""
    point_counts = [len(c) for c in clouds]
    return {
        "n_total": len(clouds),
        "n_feasible": int((labels == 1).sum()),
        "n_infeasible": int((labels == 0).sum()),
        "avg_points": float(np.mean(point_counts)),
        "min_points": int(np.min(point_counts)),
        "max_points": int(np.max(point_counts)),
        "std_points": float(np.std(point_counts)),
    }
