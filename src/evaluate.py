import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

def evaluate(data_path, label="Dataset"):
    # Load model and encoder
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/encoders.pkl", "rb") as f:
        le = pickle.load(f)

    # Load and preprocess data
    df = pd.read_csv(data_path)
    df["region_enc"] = le.transform(df["region"])
    df = df.drop(columns=["region"])

    X = df.drop(columns=["churn"])
    y = df["churn"]

    # Predict
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1  = f1_score(y, y_pred, zero_division=0)

    print(f"\n--- {label} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")

    return acc, f1

if __name__ == "__main__":
    acc_orig,    f1_orig    = evaluate("data/raw.csv",     "Original Data")
    acc_drifted, f1_drifted = evaluate("data/drifted.csv", "Drifted Data")

    print("\n--- Performance Drop ---")
    print(f"Accuracy dropped by : {(acc_orig - acc_drifted):.4f}")
    print(f"F1 dropped by       : {(f1_orig  - f1_drifted):.4f}")