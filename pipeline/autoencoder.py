"""
pipeline/autoencoder.py
Shallow autoencoder for unsupervised point-cloud feature learning.

Implementation uses sklearn's MLPRegressor trained in auto-association mode
(input = target).  The bottleneck activations are extracted via a manual
forward pass through the first two weight layers.

No PyTorch dependency required.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


class SimpleAutoencoder:
    """
    Architecture  (default):
        input (n_points × 3 = 768) → 256 → latent_dim (64) → 256 → 768

    Usage
    -----
    ae = SimpleAutoencoder(n_points=256, latent_dim=64)
    ae.fit(clouds)                    # unsupervised — no labels used
    embeddings = ae.transform(clouds) # shape: (N, latent_dim)
    """

    def __init__(self, n_points: int = 256, latent_dim: int = 64) -> None:
        self.n_points = n_points
        self.latent_dim = latent_dim
        self._ae = None
        self._X_mean: Optional[np.ndarray] = None
        self._X_std: Optional[np.ndarray] = None
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample(self, cloud: np.ndarray, seed: int = 0) -> np.ndarray:
        """Sample exactly n_points from a cloud and flatten to 1-D."""
        rng = np.random.default_rng(seed)
        n = len(cloud)
        idx = rng.choice(n, self.n_points, replace=(n < self.n_points))
        return cloud[idx].flatten().astype(np.float32)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        clouds: List[np.ndarray],
        epochs: int = 50,
        random_state: int = 42,
        progress_callback=None,
    ) -> "SimpleAutoencoder":
        """
        Train autoencoder on all clouds (unsupervised — labels ignored).

        Parameters
        ----------
        clouds    : list of (N_i, 3) arrays
        epochs    : MLPRegressor max_iter
        """
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler

        X = np.array([self._sample(c, i) for i, c in enumerate(clouds)])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self._X_mean = scaler.mean_
        self._X_std = scaler.scale_

        self._ae = MLPRegressor(
            hidden_layer_sizes=(256, self.latent_dim, 256),
            activation="relu",
            max_iter=epochs,
            random_state=random_state,
            verbose=False,
            early_stopping=False,
        )
        self._ae.fit(X_scaled, X_scaled)
        self.is_fitted = True

        if progress_callback:
            progress_callback(1.0)

        return self

    def transform(self, clouds: List[np.ndarray]) -> np.ndarray:
        """
        Extract latent embeddings for each cloud.

        Returns
        -------
        embeddings : (N, latent_dim) float32
        """
        if not self.is_fitted or self._ae is None:
            raise RuntimeError("Autoencoder has not been fitted yet.")

        X = np.array([self._sample(c, i) for i, c in enumerate(clouds)])
        X_scaled = (X - self._X_mean) / (self._X_std + 1e-10)

        # Manual forward pass to bottleneck (after 2nd hidden layer)
        h = X_scaled
        # Layer 0: input → 256
        h = self._relu(h @ self._ae.coefs_[0] + self._ae.intercepts_[0])
        # Layer 1: 256 → latent_dim
        h = self._relu(h @ self._ae.coefs_[1] + self._ae.intercepts_[1])

        return h.astype(np.float32)

    def fit_transform(
        self,
        clouds: List[np.ndarray],
        epochs: int = 50,
        random_state: int = 42,
    ) -> np.ndarray:
        self.fit(clouds, epochs=epochs, random_state=random_state)
        return self.transform(clouds)
