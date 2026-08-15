"""
synthesizer.py
--------------
A from-scratch Gaussian Copula synthesizer for tabular data.

Why Gaussian Copula instead of SDV/CTGAN?
This sandbox has no internet access, so `pip install sdv` / `ctgan` (which
also pulls in torch) isn't available. Gaussian Copula is the same modeling
family SDV's `GaussianCopulaSynthesizer` uses under the hood: it is fast,
CPU-only, and typically the recommended default for structured transaction
data (CTGAN/GAN-based synthesizers mainly earn their keep on data with
complex multi-modal numeric distributions or many rare categories).
The `main.py` docstring shows the one-line swap to plug in real SDV/CTGAN
on a machine with internet access, with no other pipeline changes needed.

How it works:
1. Every column (numeric or categorical) is converted to a uniform [0,1]
   value via its empirical CDF (categoricals are frequency-encoded first).
2. Uniform values are mapped to standard-normal space (inverse normal CDF)
   -> this is the "copula" space where correlation is well defined.
3. We fit a multivariate Gaussian correlation matrix in that space,
   capturing how columns move together (e.g. total_amount vs unit_price).
4. To generate synthetic rows: sample from that multivariate normal,
   map back through the normal CDF to uniforms, then back through each
   column's inverse empirical CDF to real values in the original space.

This preserves each column's original marginal distribution exactly
(by construction) and approximates the original correlation structure.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


class GaussianCopulaSynthesizer:
    def __init__(self, random_state: int = 7):
        self.rng = np.random.default_rng(random_state)
        self.columns: list[str] = []
        self.dtypes: dict[str, str] = {}
        self.category_maps: dict[str, list] = {}   # categorical -> sorted category list
        self.category_probs: dict[str, np.ndarray] = {}
        self.numeric_sorted: dict[str, np.ndarray] = {}  # sorted values, for empirical inverse-CDF
        self.corr_matrix: np.ndarray | None = None
        self.n_fit_rows: int = 0

    # ---------- fitting ----------
    def fit(self, df: pd.DataFrame):
        self.columns = list(df.columns)
        self.n_fit_rows = len(df)
        latent = np.zeros((len(df), len(self.columns)))

        for i, col in enumerate(self.columns):
            series = df[col]
            if pd.api.types.is_numeric_dtype(series) and series.nunique() > 15:
                self.dtypes[col] = "numeric"
                sorted_vals = np.sort(series.values.astype(float))
                self.numeric_sorted[col] = sorted_vals
                # empirical CDF rank -> uniform (with small jitter to avoid 0/1 clipping)
                ranks = stats.rankdata(series.values, method="average")
                u = (ranks - 0.5) / len(series)
            else:
                self.dtypes[col] = "categorical"
                vc = series.value_counts(normalize=True).sort_index()
                cats = list(vc.index)
                probs = vc.values
                self.category_maps[col] = cats
                self.category_probs[col] = probs
                # map each category to its cumulative-probability midpoint -> uniform
                cum = np.concatenate([[0], np.cumsum(probs)])
                cat_to_mid = {c: (cum[j] + cum[j + 1]) / 2 for j, c in enumerate(cats)}
                u = series.map(cat_to_mid).values.astype(float)

            u = np.clip(u, 1e-6, 1 - 1e-6)
            latent[:, i] = stats.norm.ppf(u)

        self.corr_matrix = np.corrcoef(latent, rowvar=False)
        # ensure positive semi-definite (numerical safety)
        eigvals, eigvecs = np.linalg.eigh(self.corr_matrix)
        eigvals = np.clip(eigvals, 1e-6, None)
        self.corr_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
        d = np.sqrt(np.diag(self.corr_matrix))
        self.corr_matrix = self.corr_matrix / np.outer(d, d)
        return self

    # ---------- sampling ----------
    def sample(self, n: int) -> pd.DataFrame:
        latent = self.rng.multivariate_normal(
            mean=np.zeros(len(self.columns)), cov=self.corr_matrix, size=n
        )
        u = stats.norm.cdf(latent)  # back to uniforms, correlation structure preserved

        out = {}
        for i, col in enumerate(self.columns):
            col_u = np.clip(u[:, i], 1e-6, 1 - 1e-6)
            if self.dtypes[col] == "numeric":
                sorted_vals = self.numeric_sorted[col]
                idx = (col_u * len(sorted_vals)).astype(int)
                idx = np.clip(idx, 0, len(sorted_vals) - 1)
                out[col] = sorted_vals[idx]
            else:
                cats = self.category_maps[col]
                probs = self.category_probs[col]
                cum = np.cumsum(probs)
                idx = np.searchsorted(cum, col_u)
                idx = np.clip(idx, 0, len(cats) - 1)
                out[col] = [cats[j] for j in idx]

        synth = pd.DataFrame(out, columns=self.columns)

        # restore sensible integer dtypes where the original column was integer-like
        for col in self.columns:
            if self.dtypes[col] == "numeric" and np.allclose(
                self.numeric_sorted[col], np.round(self.numeric_sorted[col])
            ):
                synth[col] = synth[col].round().astype(int)
        return synth
