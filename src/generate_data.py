import numpy as np
import pandas as pd

np.random.seed(42)

def generate_churn_data(n_samples=1000):
    age            = np.random.normal(40, 10, n_samples).clip(18, 75).astype(int)
    income         = np.random.normal(60000, 15000, n_samples).clip(20000, 150000).astype(int)
    tenure         = np.random.randint(1, 15, size=n_samples)
    monthly_charge = np.random.normal(70, 20, n_samples).clip(20, 150).round(2)
    num_products   = np.random.randint(1, 6, size=n_samples)
    region         = np.random.choice(["North", "South", "East", "West"], size=n_samples)
    support_calls  = np.random.poisson(lam=2, size=n_samples).clip(0, 10)

    # Clear, learnable churn signal
    churn_score = (
        0.8 * (support_calls / 10)
        + 0.5 * (1 - tenure / 15)
        + 0.3 * (monthly_charge / 150)
        - 0.2 * (income / 150000)
        - 0.1 * (num_products / 5)
    )
    churn_prob = (churn_score / churn_score.max() * 0.85).clip(0.05, 0.95)
    churn = (np.random.rand(n_samples) < churn_prob).astype(int)

    return pd.DataFrame({
        "age": age, "income": income, "tenure": tenure,
        "monthly_charge": monthly_charge, "num_products": num_products,
        "region": region, "support_calls": support_calls, "churn": churn,
    })

if __name__ == "__main__":
    df = generate_churn_data()
    df.to_csv("data/raw.csv", index=False)
    print(f"Saved! Shape: {df.shape}, Churn rate: {df['churn'].mean():.2%}")