from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from url_analyzer import analyze_url

MODEL_PATH = PROJECT_DIR / "models" / "random_forest.joblib"
DATA_PATH = PROJECT_DIR / "data" / "phishing_prepared.csv"
CHART_PATH = PROJECT_DIR / "reports" / "feature_importance.png"
IMPORTANCE_PATH = PROJECT_DIR / "reports" / "feature_importance.csv"

st.set_page_config(
    page_title="PhishGuard AI",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ PhishGuard AI")
st.subheader("Machine Learning-Based Phishing Detection")

st.write(
    "An educational phishing-analysis prototype combining URL heuristics, "
    "machine learning, and feature-importance reporting."
)

st.warning(
    "The URL risk score is based on simple rules, not a trained ML prediction. "
    "It does not establish whether a website is safe. The ML model does not "
    "yet accept arbitrary raw URLs."
)

# --------------------------------------------------
# 1. URL STRUCTURE ANALYZER
# --------------------------------------------------

st.divider()
st.header("1. URL Structure Analyzer")

url_input = st.text_input(
    "Enter a URL to inspect",
    placeholder="https://example.com/login",
)

if st.button("Analyze URL", type="primary"):
    try:
        result = analyze_url(url_input)

        st.write("**Hostname:**", result["hostname"])

        score_col, level_col = st.columns(2)

        score_col.metric(
            "Heuristic Risk Score",
            f'{result["risk_score"]}/100',
        )

        level_col.metric(
            "Heuristic Risk Level",
            result["risk_level"],
        )

        st.progress(result["risk_score"] / 100)

        if result["risk_level"] == "Elevated":
            st.error(
                "Elevated heuristic risk: inspect the URL carefully."
            )
        elif result["risk_level"] == "Moderate":
            st.warning(
                "Moderate heuristic risk: review the findings carefully."
            )
        else:
            st.success(
                "Low heuristic risk under these rules. "
                "This does not guarantee the URL is safe."
            )

        if result["is_ip_address"]:
            st.warning("The hostname is an IP address.")

        st.write("**Checks and observations:**")

        for level, message in result["findings"]:
            if level == "Caution":
                st.warning(message)
            else:
                st.info(message)

        st.caption(
            "The score is a rule-based indicator, not a phishing probability. "
            "No website content is fetched by this analyzer."
        )

    except ValueError as error:
        st.error(str(error))

# --------------------------------------------------
# 2. MACHINE-LEARNING DEMONSTRATION
# --------------------------------------------------

st.divider()
st.header("2. Machine-Learning Demonstration")

if not MODEL_PATH.exists() or not DATA_PATH.exists():
    st.error("The trained model or prepared dataset is missing.")
    st.stop()

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["result"])
y = df["result"].map({-1: 1, 1: 0})

col1, col2, col3 = st.columns(3)

col1.metric("Prepared Samples", f"{len(df):,}")
col2.metric("Input Features", str(X.shape[1]))
col3.metric("Model", "Random Forest")

sample_index = st.number_input(
    "Select an existing dataset example",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1,
)

if st.button("Predict Dataset Example"):
    sample = X.iloc[[int(sample_index)]]
    actual = int(y.iloc[int(sample_index)])
    prediction = int(model.predict(sample)[0])
    probabilities = model.predict_proba(sample)[0]
    phishing_score = float(probabilities[1])

    left, right = st.columns(2)

    with left:
        st.write("**Model prediction**")

        if prediction == 1:
            st.error("Phishing")
        else:
            st.success("Legitimate")

    with right:
        st.write("**Dataset label**")
        st.write("Phishing" if actual == 1 else "Legitimate")

    st.metric(
        "Estimated phishing probability",
        f"{phishing_score:.2%}",
    )

    st.caption(
        "This ML prediction uses an existing engineered-feature dataset "
        "example. It is not a prediction for the URL entered above. "
        "The displayed probability is the model's estimate for that sample."
    )

    with st.expander("View input features"):
        st.dataframe(sample.T, width="stretch")

# --------------------------------------------------
# 3. MODEL EXPLAINABILITY
# --------------------------------------------------

st.divider()
st.header("3. Model Explainability")

if CHART_PATH.exists():
    st.image(
        str(CHART_PATH),
        caption="Top 10 Random Forest feature-importance scores",
        width="stretch",
    )
else:
    st.info(
        "Feature-importance chart not found. "
        "Run src/explain_model.py."
    )

if IMPORTANCE_PATH.exists():
    importance = pd.read_csv(IMPORTANCE_PATH)
    st.dataframe(importance, width="stretch")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "PhishGuard AI | Educational prototype | "
    "Not a standalone security tool"
)
