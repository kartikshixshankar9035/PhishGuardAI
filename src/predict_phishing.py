from pathlib import Path

import joblib
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "random_forest.joblib"
DATA_PATH = PROJECT_DIR / "data" / "phishing_prepared.csv"


def main():
    if not MODEL_PATH.exists():
        print("ERROR: Trained model not found.")
        return

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["result"])
    y = df["result"].map({-1: 1, 1: 0})

    # Select one existing dataset example for a safe demonstration.
    sample = X.iloc[[0]]
    actual_label = int(y.iloc[0])

    prediction = int(model.predict(sample)[0])
    probabilities = model.predict_proba(sample)[0]
    phishing_probability = float(probabilities[1])

    print("\n===== PhishGuard AI Prediction Demo =====")
    print("Input: One example from the prepared dataset")
    print("Actual class:", "Phishing" if actual_label == 1 else "Legitimate")
    print("Predicted class:", "Phishing" if prediction == 1 else "Legitimate")
    print(f"Model phishing score: {phishing_probability:.2%}")
    print("\nNote: This demo uses engineered dataset features, not a raw URL.")
    print("The score is a model estimate, not a guarantee of safety.")


if __name__ == "__main__":
    main()
