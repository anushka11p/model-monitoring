import numpy as np
import pandas as pd

def simulate_drift():
    df = pd.read_csv("data/raw.csv")

    # 1. DATA DRIFT — input distributions change
    # Customers are getting older (population shift)
    df["age"] = df["age"] + 10

    # Income becomes more noisy (economic uncertainty)
    df["income"] = df["income"] * np.random.uniform(0.7, 1.3, size=len(df))

    # More people are calling support (service is getting worse)
    df["support_calls"] = (df["support_calls"] + np.random.randint(0, 4, size=len(df))).clip(0, 10)

    # 2. CONCEPT DRIFT — the relationship between features and churn changes
    # 20% of labels get flipped (model's learned rules no longer apply)
    mask = np.random.rand(len(df)) < 0.2
    df.loc[mask, "churn"] = 1 - df.loc[mask, "churn"]

    df.to_csv("data/drifted.csv", index=False)
    print("Drifted dataset saved to data/drifted.csv")
    print(f"Original churn rate : 35.50%")
    print(f"Drifted churn rate  : {df['churn'].mean():.2%}")

if __name__ == "__main__":
    simulate_drift()