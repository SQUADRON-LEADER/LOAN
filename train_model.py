import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TARGET_COLUMN = "Loan_Approved"
DROP_COLUMNS = ["Applicant_ID"]
OUTPUT_DIR = Path("model_artifacts")
MODEL_PATH = OUTPUT_DIR / "best_model.joblib"
METRICS_PATH = OUTPUT_DIR / "metrics.json"


def main() -> None:
    df = pd.read_csv("loan.csv")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in loan.csv")

    # Training rows must have target labels.
    df = df.dropna(subset=[TARGET_COLUMN]).copy()

    for col in DROP_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = [c for c in X.columns if c not in categorical_features]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "naive_bayes": GaussianNB(),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    all_results = {}
    best_name = None
    best_score = -1.0
    best_pipeline = None

    for name, model in candidates.items():
        pipeline = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        f1 = f1_score(y_test, y_pred, average="weighted")
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

        all_results[name] = {
            "f1_weighted": float(f1),
            "accuracy": float(acc),
            "precision_weighted": float(precision),
            "recall_weighted": float(recall),
        }

        # Select the best model by weighted F1, then accuracy as tie-breaker.
        rank_score = (f1, acc)
        if rank_score > (best_score, all_results.get(best_name, {}).get("accuracy", -1.0)):
            best_score = f1
            best_name = name
            best_pipeline = pipeline

    if best_pipeline is None or best_name is None:
        raise RuntimeError("No model was trained successfully.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)

    summary = {
        "target_column": TARGET_COLUMN,
        "dropped_columns": DROP_COLUMNS,
        "feature_columns": X.columns.tolist(),
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "best_model_name": best_name,
        "metrics": all_results,
    }

    METRICS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Best model: {best_name}")
    print(json.dumps(all_results[best_name], indent=2))
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
