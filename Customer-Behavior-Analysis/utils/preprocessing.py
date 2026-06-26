"""
Data Preprocessing Utilities
Handles data cleaning, encoding, scaling, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import joblib
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def load_data(uploaded_file) -> pd.DataFrame:
    """Load CSV or Excel data from uploaded file object."""
    try:
        filename = uploaded_file.name
        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file type. Please upload CSV or Excel.")
        return df
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")


def get_dataset_info(df: pd.DataFrame) -> dict:
    """Return a dictionary of basic dataset information."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicates": int(df.duplicated().sum()),
        "missing_total": int(df.isnull().sum().sum()),
        "missing_by_col": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 ** 2, 3),
    }


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows."""
    return df.drop_duplicates().reset_index(drop=True)


def handle_missing_values(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """
    Impute missing values.
    Numeric columns: mean / median / most_frequent / constant(0)
    Categorical columns: most_frequent
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if num_cols:
        num_strategy = strategy if strategy in ("mean", "median", "most_frequent") else "median"
        num_imputer = SimpleImputer(strategy=num_strategy)
        df[num_cols] = num_imputer.fit_transform(df[num_cols])

    if cat_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    return df


def label_encode_columns(df: pd.DataFrame, columns: list) -> tuple:
    """
    Apply LabelEncoding to specified columns.
    Returns (encoded_df, encoders_dict).
    """
    df = df.copy()
    encoders = {}
    for col in columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders


def one_hot_encode_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Apply One-Hot Encoding to specified columns."""
    df = df.copy()
    df = pd.get_dummies(df, columns=columns, drop_first=False)
    return df


def scale_features(df: pd.DataFrame, columns: list, method: str = "standard") -> tuple:
    """
    Scale numeric features.
    method: 'standard' (z-score) or 'minmax'.
    Returns (scaled_df, scaler).
    """
    df = df.copy()
    if method == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    valid_cols = [c for c in columns if c in df.columns]
    df[valid_cols] = scaler.fit_transform(df[valid_cols])
    return df, scaler


def auto_clean_pipeline(df: pd.DataFrame, missing_strategy: str = "median") -> pd.DataFrame:
    """
    Full automatic cleaning pipeline:
    1. Remove duplicates
    2. Handle missing values
    3. Strip whitespace from string columns
    4. Convert obvious date columns
    """
    df = remove_duplicates(df)
    df = handle_missing_values(df, strategy=missing_strategy)

    # Strip whitespace
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Attempt to parse obvious datetime columns
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass

    return df


def prepare_churn_features(df: pd.DataFrame) -> tuple:
    """
    Prepare feature matrix X and target y for churn prediction.
    Looks for a 'Churn' column (binary 0/1 or Yes/No).
    Returns (X, y, feature_names) or raises ValueError if no churn column found.
    """
    df = df.copy()

    # Detect churn column
    churn_col = None
    for col in df.columns:
        if col.lower() in ("churn", "churned", "is_churn", "customer_churn"):
            churn_col = col
            break

    if churn_col is None:
        raise ValueError("No 'Churn' column found in dataset.")

    # Encode target
    if df[churn_col].dtype == object:
        df[churn_col] = df[churn_col].map({"Yes": 1, "No": 0, "yes": 1, "no": 0,
                                            "True": 1, "False": 0, "1": 1, "0": 0})

    y = df[churn_col].astype(int)
    X = df.drop(columns=[churn_col])

    # Drop non-numeric / id columns
    drop_cols = [c for c in X.columns if X[c].dtype == object or "id" in c.lower()]
    X = X.drop(columns=drop_cols, errors="ignore")
    X = X.select_dtypes(include=[np.number])

    # Fill any remaining NaN
    X = X.fillna(X.median(numeric_only=True))

    return X, y, X.columns.tolist()


def save_artifacts(scaler, encoder, path: str = MODELS_DIR):
    """Persist scaler and encoder to disk."""
    os.makedirs(path, exist_ok=True)
    joblib.dump(scaler, os.path.join(path, "scaler.pkl"))
    joblib.dump(encoder, os.path.join(path, "encoder.pkl"))


def load_artifacts(path: str = MODELS_DIR):
    """Load scaler and encoder from disk."""
    scaler_path = os.path.join(path, "scaler.pkl")
    encoder_path = os.path.join(path, "encoder.pkl")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    encoder = joblib.load(encoder_path) if os.path.exists(encoder_path) else None
    return scaler, encoder
