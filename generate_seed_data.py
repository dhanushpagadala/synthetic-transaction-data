"""
generate_seed_data.py
----------------------
Simulates a "real" e-commerce transactions dataset with realistic
statistical structure (marginal distributions + correlations between
columns). In a real deployment this would simply be replaced by
`pd.read_csv("real_transactions.csv")` pointing at the sensitive data
you are NOT allowed to share. Everything downstream treats this the
same way regardless of where it came from.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

CATEGORIES = ["Electronics", "Clothing", "Home", "Books", "Beauty", "Sports"]
CATEGORY_BASE_PRICE = {
    "Electronics": 220, "Clothing": 45, "Home": 90,
    "Books": 18, "Beauty": 30, "Sports": 60,
}
PAYMENT_METHODS = ["Credit Card", "Debit Card", "PayPal", "UPI", "COD"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def generate(n_rows: int = 8000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    customer_age = np.clip(rng.normal(38, 12, n_rows), 18, 80).round().astype(int)
    tenure_days = np.clip(rng.exponential(400, n_rows), 0, 3650).round().astype(int)

    category = rng.choice(CATEGORIES, size=n_rows,
                           p=[0.22, 0.20, 0.18, 0.14, 0.14, 0.12])

    # unit price depends on category + a bit of age effect (older -> home/books pricier baskets)
    base = np.array([CATEGORY_BASE_PRICE[c] for c in category])
    age_effect = np.where(np.isin(category, ["Home", "Books"]),
                           (customer_age - 38) * 0.6, 0)
    unit_price = np.clip(base * rng.lognormal(mean=0, sigma=0.35, size=n_rows) + age_effect, 3, None)
    unit_price = unit_price.round(2)

    quantity = np.clip(rng.poisson(2, n_rows) + 1, 1, 10)

    discount_pct = np.clip(rng.beta(1.5, 6, n_rows) * 100, 0, 45).round(1)

    total_amount = (unit_price * quantity * (1 - discount_pct / 100)).round(2)

    is_returning_customer = (rng.random(n_rows) < np.clip(tenure_days / 1500, 0.05, 0.9)).astype(int)

    payment_method = rng.choice(PAYMENT_METHODS, size=n_rows,
                                 p=[0.35, 0.25, 0.15, 0.18, 0.07])

    day_of_week = rng.choice(DAYS, size=n_rows,
                              p=[0.13, 0.13, 0.13, 0.13, 0.16, 0.18, 0.14])

    # fraud probability rises with transaction size and for COD/PayPal, correlated signal for validation
    fraud_logit = (
        -6.5
        + 0.004 * total_amount
        + 1.1 * (payment_method == "COD")
        + 0.6 * (payment_method == "PayPal")
        - 0.5 * is_returning_customer
    )
    fraud_prob = 1 / (1 + np.exp(-fraud_logit))
    is_fraud = (rng.random(n_rows) < fraud_prob).astype(int)

    df = pd.DataFrame({
        "transaction_id": np.arange(1, n_rows + 1),
        "customer_age": customer_age,
        "customer_tenure_days": tenure_days,
        "product_category": category,
        "unit_price": unit_price,
        "quantity": quantity,
        "discount_pct": discount_pct,
        "total_amount": total_amount,
        "is_returning_customer": is_returning_customer,
        "payment_method": payment_method,
        "day_of_week": day_of_week,
        "is_fraud": is_fraud,
    })
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("/home/claude/synthetic_data_project/data/real_transactions.csv", index=False)
    print(df.shape)
    print(df.head())
