from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo


def main():
    project_dir = Path(__file__).resolve().parent.parent
    data_dir = project_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading the UCI Phishing Websites dataset...")

    try:
        dataset = fetch_ucirepo(id=327)
        features = dataset.data.features
        targets = dataset.data.targets
        data = pd.concat([features, targets], axis=1)

        output_path = data_dir / "phishing_websites.csv"
        data.to_csv(output_path, index=False)

        print("\nDownload successful!")
        print(f"Rows: {data.shape[0]}")
        print(f"Columns: {data.shape[1]}")
        print(f"Saved to: {output_path}")
        print("\nFirst five rows:")
        print(data.head().to_string())
        print("\nTarget distribution:")
        print(targets.value_counts().to_string())

    except Exception as error:
        print("\nDataset download failed.")
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
