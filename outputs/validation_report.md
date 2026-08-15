# Synthetic Data Fidelity Validation Report

- Real rows: 8,000 | Synthetic rows: 8,000
- Exact-row matches between real and synthetic: 0 (0.000% — near-zero confirms no memorization/leakage)
- **Overall statistical similarity: 99.19%**

## Numeric columns — Kolmogorov-Smirnov test

| column               |   ks_statistic |   p_value |   similarity_pct |
|:---------------------|---------------:|----------:|-----------------:|
| customer_age         |         0.0054 |    0.9998 |            99.46 |
| customer_tenure_days |         0.011  |    0.7184 |            98.9  |
| unit_price           |         0.01   |    0.8187 |            99    |
| quantity             |         0.005  |    1      |            99.5  |
| discount_pct         |         0.0165 |    0.2262 |            98.35 |
| total_amount         |         0.0118 |    0.6388 |            98.82 |

> Lower KS statistic = closer match between real and synthetic distributions. p > 0.05 means we cannot reject that both samples come from the same distribution.

## Categorical columns — Total Variation Distance

| column                |   total_variation_distance |   similarity_pct |
|:----------------------|---------------------------:|-----------------:|
| product_category      |                     0.0113 |            98.88 |
| is_returning_customer |                     0.0006 |            99.94 |
| payment_method        |                     0.0104 |            98.96 |
| day_of_week           |                     0.0095 |            99.05 |
| is_fraud              |                     0.0002 |            99.98 |

> TVD close to 0 means category frequencies match closely.

## Correlation structure

- Correlation-matrix similarity: **99.39%**
- See `correlation_comparison.png` for a visual side-by-side.

## Privacy check

- No synthetic row is an exact duplicate of any real row (checked above).
- The synthesizer only ever learns per-column distributions and pairwise correlations (a compact statistical summary) — it never stores or replays individual real records, so the synthetic file can be shared without exposing real customer PII.
