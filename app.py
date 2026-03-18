import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Loan Approval Prediction", layout="wide")

MODEL_PATH = Path("model_artifacts/best_model.joblib")
METRICS_PATH = Path("model_artifacts/metrics.json")
DATA_PATH = Path("loan.csv")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def build_input_form(df: pd.DataFrame, feature_columns: list[str], categorical_features: list[str]) -> pd.DataFrame:
    input_data = {}

    left, right = st.columns(2)
    cols = [left, right]

    for i, col in enumerate(feature_columns):
        container = cols[i % 2]
        with container:
            if col in categorical_features:
                options = sorted(df[col].dropna().astype(str).unique().tolist())
                default = options[0] if options else ""
                input_data[col] = st.selectbox(col, options=options if options else [""], index=0)
                if input_data[col] == "":
                    input_data[col] = default
            else:
                series = pd.to_numeric(df[col], errors="coerce")
                default_value = float(series.median()) if not series.dropna().empty else 0.0
                min_value = float(series.min()) if not series.dropna().empty else 0.0
                max_value = float(series.max()) if not series.dropna().empty else 1000000.0
                step = 1.0 if abs(default_value) >= 1 else 0.01

                input_data[col] = st.number_input(
                    col,
                    min_value=min_value,
                    max_value=max_value,
                    value=default_value,
                    step=step,
                )

    return pd.DataFrame([input_data])


def main() -> None:
    st.title("Loan Approval Prediction")
    st.write("Use the form below to predict whether a loan application is likely to be approved.")

    model = load_model()
    metadata = load_metadata()

    if model is None:
        st.error(
            "Model file not found. Run 'python train_model.py' first to train and save the best model."
        )
        st.stop()

    if not DATA_PATH.exists():
        st.error("loan.csv not found in this folder.")
        st.stop()

    df = load_data()

    feature_columns = metadata.get("feature_columns")
    categorical_features = metadata.get("categorical_features", [])

    if not feature_columns:
        fallback_cols = [c for c in df.columns if c not in {"Loan_Approved", "Applicant_ID"}]
        feature_columns = fallback_cols
        categorical_features = df[feature_columns].select_dtypes(include=["object", "category"]).columns.tolist()

    with st.expander("Model details", expanded=True):
        st.write(f"Best model: {metadata.get('best_model_name', 'Unknown')}")
        scores = metadata.get("metrics", {}).get(metadata.get("best_model_name", ""), {})
        if scores:
            st.json(scores)

    with st.form("prediction_form"):
        input_df = build_input_form(df, feature_columns, categorical_features)
        submitted = st.form_submit_button("Predict")

    if submitted:
        prediction = model.predict(input_df)[0]

        st.subheader("Prediction result")
        st.write(f"Loan approval prediction: {prediction}")

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]
            classes = [str(c) for c in model.classes_]
            prob_df = pd.DataFrame({"Class": classes, "Probability": probs}).sort_values(
                by="Probability", ascending=False
            )
            st.write("Prediction probabilities")
            st.dataframe(prob_df, use_container_width=True)


if __name__ == "__main__":
    main()
