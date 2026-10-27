from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "phishing_websites.csv"

def main():
    print("PHISHGUARD AI - DATASET ANALYSIS")
    print("=" * 40)

    if not DATA_PATH.exists():
        print("Dataset not found:", DATA_PATH)
        return

    df = pd.read_csv(DATA_PATH)

    print("\n1. DATASET SHAPE")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\n2. MISSING VALUES")
    print("Total:", df.isnull().sum().sum())

    print("\n3. DUPLICATE RECORDS")
    print("Duplicate rows:", df.duplicated().sum())

    print("\n4. TARGET DISTRIBUTION")
    print(df["result"].value_counts().to_string())

    print("\n5. TARGET PERCENTAGES")
    print((df["result"].value_counts(normalize=True) * 100).round(2).to_string())

    print("\n6. TARGET VALUES")
    print(sorted(df["result"].dropna().unique().tolist()))

    print("\n7. DATA TYPES")
    print(df.dtypes.to_string())

    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()
