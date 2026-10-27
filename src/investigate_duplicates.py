from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "phishing_websites.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    features = df.drop(columns=["result"])
    labels = df["result"]

    print("=" * 50)
    print("PHISHGUARD AI - DUPLICATE INVESTIGATION")
    print("=" * 50)

    print("\n1. Original dataset rows:", len(df))

    print("\n2. Exact duplicate rows:", df.duplicated().sum())

    print("\n3. Unique complete records:", len(df.drop_duplicates()))

    feature_groups = df.groupby(
        list(features.columns),
        dropna=False
    )["result"].nunique()

    conflicting_groups = int((feature_groups > 1).sum())

    print("\n4. Unique feature combinations:", len(feature_groups))
    print("5. Feature combinations with conflicting labels:",
          conflicting_groups)

    cleaned = df.drop_duplicates()

    print("\n6. Rows after removing exact duplicates:", len(cleaned))

    print("\n7. Labels before removing duplicates:")
    print(labels.value_counts().to_string())

    print("\n8. Labels after removing exact duplicates:")
    print(cleaned["result"].value_counts().to_string())

    print("\nInvestigation complete.")


if __name__ == "__main__":
    main()
