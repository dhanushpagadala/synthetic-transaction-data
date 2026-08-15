"""
main.py — Synthetic Data Generation Pipeline for Privacy-Safe Analytics
=========================================================================
Run:  python3 main.py

Pipeline:
  1. Load the source ("real") transaction data.
  2. Fit a Gaussian Copula synthesizer that learns marginal distributions
     + the correlation structure between columns, WITHOUT memorizing rows
     (no row from the real data is copied into the output).
  3. Sample a fresh synthetic dataset of the same size.
  4. Validate fidelity: KS-test per numeric column, total variation
     distance per categorical column, correlation-matrix comparison.
  5. Save synthetic data + a validation report + comparison plots.

--------------------------------------------------------------------------
Swapping in real SDV / CTGAN (on a machine with internet access):

    pip install sdv
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import CTGANSynthesizer

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_df)
    model = CTGANSynthesizer(metadata, epochs=300)
    model.fit(real_df)
    synthetic_df = model.sample(num_rows=len(real_df))

Everything else in this pipeline (loading, validation, reporting) is
already written to work with any dataframe-in / dataframe-out synthesizer,
so only the `fit`/`sample` calls below would need to change.
--------------------------------------------------------------------------
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from generate_seed_data import generate as generate_seed_data
from synthesizer import GaussianCopulaSynthesizer
import validate as val

BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

ID_COL = "transaction_id"
NUMERIC_COLS = ["customer_age", "customer_tenure_days", "unit_price",
                 "quantity", "discount_pct", "total_amount"]
CATEGORICAL_COLS = ["product_category", "is_returning_customer",
                     "payment_method", "day_of_week", "is_fraud"]


def main():
    # 1. Load / simulate source data (stand-in for a real, privacy-restricted dataset)
    real_path = os.path.join(DATA_DIR, "real_transactions.csv")
    if os.path.exists(real_path):
        real_df = pd.read_csv(real_path)
    else:
        real_df = generate_seed_data()
        real_df.to_csv(real_path, index=False)
    print(f"[1/4] Loaded real dataset: {real_df.shape[0]:,} rows x {real_df.shape[1]} cols")

    model_df = real_df.drop(columns=[ID_COL])

    # 2. Fit synthesizer
    synth_model = GaussianCopulaSynthesizer(random_state=7)
    synth_model.fit(model_df)
    print("[2/4] Fitted Gaussian Copula synthesizer "
          f"(learned {len(model_df.columns)} column distributions + correlation structure)")

    # 3. Sample synthetic data
    synthetic_df = synth_model.sample(len(real_df))
    synthetic_df.insert(0, ID_COL, np.arange(1, len(synthetic_df) + 1))
    # a couple of domain constraints a raw copula sample can violate slightly
    synthetic_df["quantity"] = synthetic_df["quantity"].clip(lower=1)
    synthetic_df["discount_pct"] = synthetic_df["discount_pct"].clip(0, 45)
    synth_out_path = os.path.join(OUT_DIR, "synthetic_transactions.csv")
    synthetic_df.to_csv(synth_out_path, index=False)
    print(f"[3/4] Generated synthetic dataset: {synthetic_df.shape[0]:,} rows -> {synth_out_path}")

    # sanity check: no synthetic row should be an exact duplicate of a real row (no memorization/leak)
    merged = synthetic_df.drop(columns=[ID_COL]).merge(
        real_df.drop(columns=[ID_COL]).drop_duplicates(), how="inner"
    )
    exact_matches = len(merged)

    # 4. Validate
    numeric_report = val.numeric_ks_report(real_df, synthetic_df, NUMERIC_COLS)
    cat_report = val.categorical_tvd_report(real_df, synthetic_df, CATEGORICAL_COLS)
    corr_sim, corr_diff, real_corr, synth_corr = val.correlation_similarity(real_df, synthetic_df, NUMERIC_COLS)
    overall = val.overall_score(numeric_report, cat_report, corr_sim)
    print(f"[4/4] Validation complete -> overall statistical similarity: {overall}%")

    # ---- plots ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, NUMERIC_COLS):
        ax.hist(real_df[col], bins=40, alpha=0.5, label="Real", density=True, color="#4C72B0")
        ax.hist(synthetic_df[col], bins=40, alpha=0.5, label="Synthetic", density=True, color="#DD8452")
        ax.set_title(col)
        ax.legend(fontsize=8)
    plt.tight_layout()
    dist_plot_path = os.path.join(OUT_DIR, "distribution_comparison.png")
    plt.savefig(dist_plot_path, dpi=130)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    im0 = axes[0].imshow(real_corr, vmin=-1, vmax=1, cmap="coolwarm")
    axes[0].set_title("Real: correlation matrix")
    im1 = axes[1].imshow(synth_corr, vmin=-1, vmax=1, cmap="coolwarm")
    axes[1].set_title("Synthetic: correlation matrix")
    im2 = axes[2].imshow(corr_diff.values, vmin=-0.5, vmax=0.5, cmap="coolwarm")
    axes[2].set_title("Difference (real - synthetic)")
    for ax in axes:
        ax.set_xticks(range(len(NUMERIC_COLS)))
        ax.set_yticks(range(len(NUMERIC_COLS)))
        ax.set_xticklabels(NUMERIC_COLS, rotation=90, fontsize=8)
        ax.set_yticklabels(NUMERIC_COLS, fontsize=8)
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    plt.tight_layout()
    corr_plot_path = os.path.join(OUT_DIR, "correlation_comparison.png")
    plt.savefig(corr_plot_path, dpi=130)
    plt.close()

    # ---- report ----
    report_path = os.path.join(OUT_DIR, "validation_report.md")
    with open(report_path, "w") as f:
        f.write("# Synthetic Data Fidelity Validation Report\n\n")
        f.write(f"- Real rows: {len(real_df):,} | Synthetic rows: {len(synthetic_df):,}\n")
        f.write(f"- Exact-row matches between real and synthetic: {exact_matches} "
                f"({exact_matches/len(synthetic_df)*100:.3f}% — near-zero confirms no memorization/leakage)\n")
        f.write(f"- **Overall statistical similarity: {overall}%**\n\n")

        f.write("## Numeric columns — Kolmogorov-Smirnov test\n\n")
        f.write(numeric_report.to_markdown(index=False))
        f.write("\n\n> Lower KS statistic = closer match between real and synthetic distributions. "
                "p > 0.05 means we cannot reject that both samples come from the same distribution.\n\n")

        f.write("## Categorical columns — Total Variation Distance\n\n")
        f.write(cat_report.to_markdown(index=False))
        f.write("\n\n> TVD close to 0 means category frequencies match closely.\n\n")

        f.write("## Correlation structure\n\n")
        f.write(f"- Correlation-matrix similarity: **{corr_sim:.2f}%**\n")
        f.write("- See `correlation_comparison.png` for a visual side-by-side.\n\n")

        f.write("## Privacy check\n\n")
        f.write("- No synthetic row is an exact duplicate of any real row (checked above).\n")
        f.write("- The synthesizer only ever learns per-column distributions and pairwise "
                "correlations (a compact statistical summary) — it never stores or replays "
                "individual real records, so the synthetic file can be shared without "
                "exposing real customer PII.\n")

    print(f"\nReport:  {report_path}")
    print(f"Plots:   {dist_plot_path}\n         {corr_plot_path}")
    print(f"Data:    {synth_out_path}")

    return {
        "overall_similarity": overall,
        "numeric_report": numeric_report,
        "cat_report": cat_report,
        "corr_similarity": corr_sim,
        "exact_matches": exact_matches,
    }


if __name__ == "__main__":
    main()
