"""
Prediction Utilities
Train, evaluate, and persist multiple churn models.
Select the best automatically.
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report,
)
from sklearn.preprocessing import StandardScaler

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def build_models() -> dict:
    """Return a dict of candidate classifiers."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8,
                                                 random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150,
                                                         learning_rate=0.1,
                                                         max_depth=4,
                                                         random_state=42),
    }


def train_all_models(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Train all classifiers on train split.
    Returns dict with results per model:
      {name: {model, scaler, accuracy, precision, recall, f1, cm, fpr, tpr, auc_score, X_test, y_test}}
    """
    X = X.fillna(X.median(numeric_only=True))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = build_models()
    results = {}

    for name, clf in models.items():
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        y_prob = clf.predict_proba(X_test_s)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = auc(fpr, tpr)

        results[name] = {
            "model": clf,
            "scaler": scaler,
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "cm": confusion_matrix(y_test, y_pred),
            "fpr": fpr,
            "tpr": tpr,
            "auc_score": round(auc_score, 4),
            "X_test": X_test,
            "y_test": y_test,
            "feature_names": X.columns.tolist(),
            "report": classification_report(y_test, y_pred, output_dict=True),
        }

    return results


def select_best_model(results: dict) -> tuple:
    """Select the model with the highest F1 score. Returns (name, result_dict)."""
    best_name = max(results, key=lambda n: results[n]["f1"])
    return best_name, results[best_name]


def save_model(model, scaler, path: str = MODELS_DIR, name: str = "churn_model"):
    """Persist model and scaler to disk."""
    os.makedirs(path, exist_ok=True)
    joblib.dump(model, os.path.join(path, f"{name}.pkl"))
    joblib.dump(scaler, os.path.join(path, "scaler.pkl"))


def load_model(path: str = MODELS_DIR, name: str = "churn_model"):
    """Load persisted model and scaler. Returns (model, scaler) or (None, None)."""
    model_path = os.path.join(path, f"{name}.pkl")
    scaler_path = os.path.join(path, "scaler.pkl")
    model = joblib.load(model_path) if os.path.exists(model_path) else None
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return model, scaler


def predict_single(model, scaler, feature_values: list) -> tuple:
    """
    Predict churn for a single customer.
    Returns (label_str, probability_float).
    """
    X = np.array(feature_values).reshape(1, -1)
    X_s = scaler.transform(X)
    pred = model.predict(X_s)[0]
    prob = model.predict_proba(X_s)[0][1]
    label = "🔴 Will Churn" if pred == 1 else "🟢 Will Stay"
    return label, round(float(prob), 4)


def get_feature_importances(model, feature_names: list) -> pd.DataFrame:
    """Extract feature importances if available."""
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
    elif hasattr(model, "coef_"):
        imps = np.abs(model.coef_[0])
    else:
        return pd.DataFrame()

    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": imps,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    return df
