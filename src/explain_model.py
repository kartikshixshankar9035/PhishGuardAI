from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "phishing_prepared.csv"
REPORT_DIR = PROJECT_DIR / "reports"
MODEL_DIR = PROJECT_DIR / "models"


def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["result"])
    y = df["result"].map({-1: 1, 1: 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    print("Training Random Forest for explainability...")
    model.fit(X_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "random_forest.joblib"
    joblib.dump(model, model_path)

    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    })

    importance = importance.sort_values(
        "importance", ascending=False
    )

    csv_path = REPORT_DIR / "feature_importance.csv"
    importance.to_csv(csv_path, index=False)

    print("\nTop 10 important features:")
    print(importance.head(10).to_string(index=False))

    top = importance.head(10).sort_values("importance")

    plt.figure(figsize=(10, 6))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("PhishGuard AI - Random Forest Feature Importance")
    plt.tight_layout()

    chart_path = REPORT_DIR / "feature_importance.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()

    print("\nModel saved to:", model_path)
    print("Importance CSV saved to:", csv_path)
    print("Chart saved to:", chart_path)
    print("\nExplainability analysis complete.")


if __name__ == "__main__":
    main()
