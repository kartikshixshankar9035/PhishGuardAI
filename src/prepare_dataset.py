from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "phishing_websites.csv"
REPORT_PATH = PROJECT_DIR / "reports" / "duplicate_report.txt"


def main():
    df = pd.read_csv(DATA_PATH)
    features = df.drop(columns=["result"])

    print("PHISHGUARD AI - DATA PREPARATION")
    print("=" * 40)

    # Find feature combinations that have conflicting labels.
    grouped = df.groupby(
        list(features.columns),
        dropna=False
    )["result"]

    conflicting = grouped.nunique()
    conflicting = conflicting[conflicting > 1]

    print("Original rows:", len(df))
    print("Conflicting feature combinations:", len(conflicting))

    # Remove exact duplicate rows only for this preliminary analysis.
    exact_unique = df.drop_duplicates()

    print("Rows after exact deduplication:", len(exact_unique))

    # Keep one row per feature combination only when its label is
    # consistent. Exclude conflicting groups from this preliminary
    # modelling dataset, while preserving the original CSV.
    consistent_keys = conflicting.index

    feature_key_columns = list(features.columns)

    unique_features = exact_unique.drop_duplicates(
        subset=feature_key_columns
    )

    conflicting_key_set = set(consistent_keys)

    row_keys = unique_features[feature_key_columns].apply(
        lambda row: tuple(row), axis=1
    )

    keep_mask = ~row_keys.isin(conflicting_key_set)
    prepared = unique_features.loc[keep_mask].copy()

    output_path = PROJECT_DIR / "data" / "phishing_prepared.csv"
    prepared.to_csv(output_path, index=False)

    report = [
        "PHISHGUARD AI - DATA PREPARATION REPORT",
        f"Original rows: {len(df)}",
        f"Exact unique rows: {len(exact_unique)}",
        f"Conflicting feature combinations: {len(conflicting)}",
        f"Prepared rows: {len(prepared)}",
        "",
        "Prepared target distribution:",
        prepared["result"].value_counts().to_string(),
    ]

    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    print("\nPrepared rows:", len(prepared))
    print("\nPrepared target distribution:")
    print(prepared["result"].value_counts().to_string())
    print("\nSaved dataset:", output_path)
    print("Saved report:", REPORT_PATH)


if __name__ == "__main__":
    main()
