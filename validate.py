"""
validate.py
-----------
Statistical fidelity validation between real and synthetic data:
  * Kolmogorov-Smirnov test per numeric column (distribution match)
  * Total variation distance per categorical column (distribution match)
  * Correlation matrix comparison (structure/relationship match)
  * A single blended "overall statistical similarity" score
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def numeric_ks_report(real: pd.DataFrame, synth: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in numeric_cols:
        ks_stat, p_value = stats.ks_2samp(real[col].astype(float), synth[col].astype(float))
        similarity = (1 - ks_stat) * 100  # KS=0 -> identical dist -> 100% similarity
        rows.append({
            "column": col,
            "ks_statistic": round(ks_stat, 4),
            "p_value": round(p_value, 4),
            "similarity_pct": round(similarity, 2),
        })
    return pd.DataFrame(rows)


def categorical_tvd_report(real: pd.DataFrame, synth: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cat_cols:
        p_real = real[col].value_counts(normalize=True)
        p_synth = synth[col].value_counts(normalize=True)
        idx = p_real.index.union(p_synth.index)
        p_real = p_real.reindex(idx, fill_value=0)
        p_synth = p_synth.reindex(idx, fill_value=0)
        tvd = 0.5 * np.abs(p_real - p_synth).sum()
        similarity = (1 - tvd) * 100
        rows.append({
            "column": col,
            "total_variation_distance": round(tvd, 4),
            "similarity_pct": round(similarity, 2),
        })
    return pd.DataFrame(rows)


def correlation_similarity(real: pd.DataFrame, synth: pd.DataFrame, numeric_cols: list[str]):
    real_corr = real[numeric_cols].corr().values
    synth_corr = synth[numeric_cols].corr().values
    diff = real_corr - synth_corr
    frob_norm = np.linalg.norm(diff, "fro")
    max_norm = np.linalg.norm(np.ones_like(diff) * 2, "fro")  # worst case corr diff of 2 everywhere
    similarity = max(0.0, (1 - frob_norm / max_norm)) * 100
    corr_diff_df = pd.DataFrame(diff, index=numeric_cols, columns=numeric_cols)
    return similarity, corr_diff_df, real_corr, synth_corr


def overall_score(numeric_report: pd.DataFrame, cat_report: pd.DataFrame, corr_similarity: float) -> float:
    parts = list(numeric_report["similarity_pct"]) + list(cat_report["similarity_pct"]) + [corr_similarity]
    return round(float(np.mean(parts)), 2)
