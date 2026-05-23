import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

def train():
    # Load data
    df = pd.read_csv("data/raw.csv")

    # Encode the region column (text → number)
    le = LabelEncoder()
    df["region_enc"] = le.fit_transform(df["region"])
    df = df.drop(columns=["region"])

    # Split features and target
    X = df.drop(columns=["churn"])
    y = df["churn"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, zero_division=0)

    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save model and encoder
    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/encoders.pkl", "wb") as f:
        pickle.dump(le, f)

    print("Model saved to models/model.pkl")

if __name__ == "__main__":
    train()