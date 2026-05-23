import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

ACCURACY_THRESHOLD = 0.80  # if accuracy drops below this, retrain

def get_accuracy(data_path, model, le):
    df = pd.read_csv(data_path)
    df["region_enc"] = le.transform(df["region"])
    df = df.drop(columns=["region"])
    X = df.drop(columns=["churn"])
    y = df["churn"]
    y_pred = model.predict(X)
    return accuracy_score(y, y_pred)

def retrain():
    # Load existing model
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/encoders.pkl", "rb") as f:
        le = pickle.load(f)

    # Check performance on drifted data
    acc = get_accuracy("data/drifted.csv", model, le)
    print(f"Current accuracy on drifted data: {acc:.4f}")

    if acc < ACCURACY_THRESHOLD:
        print(f"Accuracy {acc:.4f} is below threshold {ACCURACY_THRESHOLD}")
        print("Retraining triggered...")

        # Retrain on drifted data
        df_old = pd.read_csv("data/raw.csv")
        df_new = pd.read_csv("data/drifted.csv")
        df = pd.concat([df_old, df_new], ignore_index=True)
        le_new = LabelEncoder()
        df["region_enc"] = le_new.fit_transform(df["region"])
        df = df.drop(columns=["region"])

        X = df.drop(columns=["churn"])
        y = df["churn"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        new_model = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        )
        new_model.fit(X_train, y_train)

        y_pred = new_model.predict(X_test)
        new_acc = accuracy_score(y_test, y_pred)
        new_f1  = f1_score(y_test, y_pred, zero_division=0)

        print(f"\nAfter retraining:")
        print(f"Accuracy : {new_acc:.4f}")
        print(f"F1 Score : {new_f1:.4f}")

        # Save updated model
        with open("models/model.pkl", "wb") as f:
            pickle.dump(new_model, f)
        with open("models/encoders.pkl", "wb") as f:
            pickle.dump(le_new, f)

        print("\nModel updated and saved!")

    else:
        print(f"Accuracy is fine. No retraining needed.")

if __name__ == "__main__":
    retrain()