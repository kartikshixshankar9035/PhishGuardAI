# PhishGuard AI

An educational phishing-analysis prototype combining URL heuristics, machine learning, and model explainability.

## Features
- URL structure analysis and rule-based risk scoring.
- Random Forest, Gradient Boosting, and Logistic Regression comparison.
- Interactive Streamlit dashboard.
- Feature-importance visualization.
- Automated URL-analyzer tests using pytest.

## Model Performance
| Model | Accuracy |
|---|---:|
| Random Forest | 96.33% |
| Gradient Boosting | 95.28% |
| Logistic Regression | 91.27% |

These results are from an experimental dataset split. They do not guarantee real-world phishing detection performance.

## Installation and Usage
1. Install Python and clone or download this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies using pip install -r requirements.txt.
4. Ensure the dataset and trained model files are present in the data and models folders.
5. Start the dashboard using python -m streamlit run app.py.




## Run Tests
Run the URL analyzer tests using python -m pytest -p no:cacheprovider tests/test_url_analyzer.py -v.

## Dataset
This project uses the UCI Phishing Websites dataset from the UCI Machine Learning Repository (dataset ID 327).
The prepared dataset contains engineered website features, not raw URLs. The URL structure analyzer uses separate rule-based heuristics.

## Limitations
- The URL risk score uses illustrative rules and is not a calibrated probability.
- The machine-learning model predicts from engineered dataset features; it does not directly classify an arbitrary URL entered into the dashboard.
- Evaluation results are experimental and do not guarantee real-world performance.
