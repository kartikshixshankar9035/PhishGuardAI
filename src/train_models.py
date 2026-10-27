from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "phishing_prepared.csv"
REPORT_DIR = PROJECT_DIR / "reports"
MODEL_DIR = PROJECT_DIR / "models"


def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["result"])
    y = df["result"].map({-1: 1, 1: 0})
    # Internal convention: 1 = phishing, 0 = legitimate.

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=42),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42,
        ),
    }

    results = []
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("PHISHGUARD AI - MODEL COMPARISON")
    print("=" * 50)
    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("Positive class: phishing")
    print()

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision_phishing": precision_score(
                y_test, predictions, pos_label=1, zero_division=0
            ),
            "recall_phishing": recall_score(
                y_test, predictions, pos_label=1, zero_division=0
            ),
            "f1_phishing": f1_score(
                y_test, predictions, pos_label=1, zero_division=0
            ),
        }
        results.append(metrics)

        print("\nMetrics:")
        for metric, value in metrics.items():
            if metric != "model":
                print(f"  {metric}: {value:.4f}")

        print("\nClassification report:")
        print(
            classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["Legitimate", "Phishing"],
                zero_division=0,
            )
        )

        print("Confusion matrix (rows=actual, columns=predicted):")
        print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    results_df = pd.DataFrame(results).sort_values(
        "f1_phishing", ascending=False
    )
    results_path = REPORT_DIR / "model_comparison.csv"
    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 50)
    print("FINAL MODEL COMPARISON")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nResults saved to:", results_path)
    print(
        "\nNote: metrics describe this test split and this prepared dataset; "
        "they do not guarantee performance on real-world URLs."
    )


if __name__ == "__main__":
    main()
