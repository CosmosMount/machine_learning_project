import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Integer-coded categorical columns in this dataset.
SEMANTIC_CATEGORICAL_COLUMNS = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]


def load_dataset() -> pd.DataFrame:
    """Load dataset from common project locations."""
    candidates = [
        Path("data/data.csv"),
        Path("data.csv"),
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path, sep=';')
            df.columns = df.columns.str.strip()
            return df
    raise FileNotFoundError("Could not find dataset at data/data.csv or data.csv")


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    semantic_categorical_cols = [c for c in SEMANTIC_CATEGORICAL_COLUMNS if c in X.columns]
    semantic_numeric_cols = [c for c in X.columns if c not in semantic_categorical_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, semantic_numeric_cols),
            ("cat", categorical_transformer, semantic_categorical_cols),
        ],
        remainder="drop",
    )


def summarize_dataset(df: pd.DataFrame, target_col: str = "Target") -> dict:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    target_counts = df[target_col].value_counts(dropna=False)
    target_ratios = (target_counts / len(df) * 100).round(2)

    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values_total": int(df.isna().sum().sum()),
        "target_distribution": {
            cls: {
                "count": int(target_counts[cls]),
                "ratio_percent": float(target_ratios[cls]),
            }
            for cls in target_counts.index
        },
    }
    return summary


def main() -> None:
    df = load_dataset()
    target_col = "Target"

    X = df.drop(columns=[target_col])
    preprocessor = build_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)

    parsed_numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    parsed_categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    semantic_categorical_cols = [c for c in SEMANTIC_CATEGORICAL_COLUMNS if c in X.columns]
    semantic_numeric_cols = [c for c in X.columns if c not in semantic_categorical_cols]

    summary = summarize_dataset(df, target_col=target_col)
    summary["parsed_numeric_feature_count"] = len(parsed_numeric_cols)
    summary["parsed_categorical_feature_count"] = len(parsed_categorical_cols)
    summary["semantic_numeric_feature_count"] = len(semantic_numeric_cols)
    summary["semantic_categorical_feature_count"] = len(semantic_categorical_cols)
    summary["processed_shape"] = list(X_processed.shape)

    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
