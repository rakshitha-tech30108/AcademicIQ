"""
train.py
--------
AcademicIQ - Trains and compares:
  Regression models  -> predict Final_Score (Linear Regression, Random Forest Regressor)
  Classification model -> predict Risk_Level (Random Forest Classifier)

Saves the best regression model and the risk classifier to models/ as
Pickle bundles, ready to be loaded by src/predict.py and app.py.

Run:
    python src/train.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import (  # noqa: E402
    load_data, clean_data, prepare_regression_data, prepare_classification_data,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "academiciq_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REGRESSION_MODEL_PATH = os.path.join(MODELS_DIR, "score_model.pkl")
CLASSIFICATION_MODEL_PATH = os.path.join(MODELS_DIR, "risk_model.pkl")


def regression_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
    }


def classification_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def train_regression_models(reg_data):
    print("\n[Regression] Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(reg_data["X_train_scaled"], reg_data["y_train"])
    lr_preds = lr.predict(reg_data["X_test_scaled"])
    lr_metrics = regression_metrics(reg_data["y_test"], lr_preds)
    print(f"    R2={lr_metrics['r2']:.4f}  MAE={lr_metrics['mae']:.3f}  "
          f"MSE={lr_metrics['mse']:.3f}  RMSE={lr_metrics['rmse']:.3f}")

    print("[Regression] Training Random Forest Regressor...")
    rf = RandomForestRegressor(
        n_estimators=350, max_depth=12, min_samples_split=4,
        min_samples_leaf=2, random_state=42, n_jobs=-1,
    )
    rf.fit(reg_data["X_train"], reg_data["y_train"])
    rf_preds = rf.predict(reg_data["X_test"])
    rf_metrics = regression_metrics(reg_data["y_test"], rf_preds)
    print(f"    R2={rf_metrics['r2']:.4f}  MAE={rf_metrics['mae']:.3f}  "
          f"MSE={rf_metrics['mse']:.3f}  RMSE={rf_metrics['rmse']:.3f}")

    comparison = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest Regressor"],
        "R2": [lr_metrics["r2"], rf_metrics["r2"]],
        "MAE": [lr_metrics["mae"], rf_metrics["mae"]],
        "MSE": [lr_metrics["mse"], rf_metrics["mse"]],
        "RMSE": [lr_metrics["rmse"], rf_metrics["rmse"]],
    })

    if rf_metrics["r2"] >= lr_metrics["r2"]:
        return {
            "model": rf, "model_name": "Random Forest Regressor",
            "uses_scaler": False, "metrics": rf_metrics, "comparison": comparison,
        }
    else:
        return {
            "model": lr, "model_name": "Linear Regression",
            "uses_scaler": True, "metrics": lr_metrics, "comparison": comparison,
        }


def train_classification_model(clf_data):
    print("\n[Classification] Training Random Forest Classifier (Risk_Level)...")
    clf = RandomForestClassifier(
        n_estimators=350, max_depth=12, min_samples_split=4,
        min_samples_leaf=2, random_state=42, n_jobs=-1, class_weight="balanced",
    )
    clf.fit(clf_data["X_train"], clf_data["y_train"])
    preds = clf.predict(clf_data["X_test"])
    metrics = classification_metrics(clf_data["y_test"], preds)
    cm = confusion_matrix(clf_data["y_test"], preds)

    print(f"    Accuracy={metrics['accuracy']:.4f}  Precision={metrics['precision']:.4f}  "
          f"Recall={metrics['recall']:.4f}  F1={metrics['f1']:.4f}")

    return {"model": clf, "metrics": metrics, "confusion_matrix": cm.tolist()}


def main():
    print("=" * 65)
    print("ACADEMICIQ - MODEL TRAINING (Regression + Classification)")
    print("=" * 65)

    if not os.path.exists(DATA_PATH):
        print("Dataset not found. Generating synthetic dataset first...")
        sys.path.append(BASE_DIR)
        from generate_dataset import main as generate_main
        generate_main()

    print("\n[1/3] Loading & cleaning data...")
    df = clean_data(load_data(DATA_PATH))
    print(f"    Records: {len(df)}")

    print("\n[2/3] Preparing regression & classification splits...")
    reg_data = prepare_regression_data(df)
    clf_data = prepare_classification_data(df)

    print("\n[3/3] Training models...")
    reg_result = train_regression_models(reg_data)
    clf_result = train_classification_model(clf_data)

    print("\nRegression model comparison:")
    print(reg_result["comparison"].to_string(index=False))
    print(f"\nBest regression model: {reg_result['model_name']} "
          f"(R2 = {reg_result['metrics']['r2']:.4f})")

    os.makedirs(MODELS_DIR, exist_ok=True)

    regression_bundle = {
        "model": reg_result["model"],
        "model_name": reg_result["model_name"],
        "scaler": reg_data["scaler"],
        "uses_scaler": reg_result["uses_scaler"],
        "feature_columns": reg_data["feature_columns"],
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "metrics": reg_result["metrics"],
        "comparison": reg_result["comparison"].to_dict(orient="records"),
    }
    with open(REGRESSION_MODEL_PATH, "wb") as f:
        pickle.dump(regression_bundle, f)
    print(f"\nRegression model bundle saved to: {REGRESSION_MODEL_PATH}")

    classification_bundle = {
        "model": clf_result["model"],
        "model_name": "Random Forest Classifier",
        "label_encoder": clf_data["label_encoder"],
        "feature_columns": clf_data["feature_columns"],
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "metrics": clf_result["metrics"],
        "confusion_matrix": clf_result["confusion_matrix"],
    }
    with open(CLASSIFICATION_MODEL_PATH, "wb") as f:
        pickle.dump(classification_bundle, f)
    print(f"Classification model bundle saved to: {CLASSIFICATION_MODEL_PATH}")

    print("\n" + "=" * 65)
    print("TRAINING COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
