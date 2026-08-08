"""
preprocessing.py
-----------------
AcademicIQ - Data loading, cleaning, encoding, and train/test preparation
for both the regression pipeline (Final_Score / GPA) and the classification
pipeline (Risk_Level).
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Numeric behavioral/academic features common to every student record
NUMERIC_FEATURES = [
    "Age",
    "Attendance",
    "Study_Hours",
    "Assignment_Completion",
    "Previous_GPA",
    "Participation_Score",
    "Internet_Usage",
    "Sleep_Hours",
    "Subject_1_Score",
    "Subject_2_Score",
    "Subject_3_Score",
    "Subject_4_Score",
    "Subject_5_Score",
]

# Categorical features that provide institutional context
CATEGORICAL_FEATURES = ["Gender", "Institution_Type", "Board", "Department"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

REGRESSION_TARGET = "Final_Score"
CLASSIFICATION_TARGET = "Risk_Level"

RISK_CLASSES = ["Low_Risk", "Medium_Risk", "High_Risk"]


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the AcademicIQ dataset from a CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'. Run generate_dataset.py first."
        )
    return pd.read_csv(csv_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, fill missing numeric values, ensure required columns exist."""
    df = df.copy()
    df = df.drop_duplicates(subset=["Student_ID"])

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna("Unknown")

    required = set(ALL_FEATURES + [REGRESSION_TARGET, CLASSIFICATION_TARGET])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    return df


def encode_categoricals(df: pd.DataFrame, encoders: dict = None):
    """
    One-hot encode categorical features. If `encoders` (a dict of expected
    dummy column names) is provided, the resulting frame is aligned to match
    those columns exactly (for consistent inference-time transformation).

    Returns
    -------
    tuple(pd.DataFrame, list)
        Encoded numeric-only dataframe and the final list of dummy column names.
    """
    df = df.copy()
    encoded = pd.get_dummies(df[CATEGORICAL_FEATURES], prefix=CATEGORICAL_FEATURES)
    numeric_part = df[NUMERIC_FEATURES].reset_index(drop=True)
    combined = pd.concat([numeric_part, encoded.reset_index(drop=True)], axis=1)

    if encoders is not None:
        # Align columns: add any missing dummy columns as 0, drop unexpected ones,
        # and enforce the original training-time column order.
        for col in encoders:
            if col not in combined.columns:
                combined[col] = 0
        combined = combined[encoders]

    return combined, list(combined.columns)


def prepare_regression_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Prepare train/test splits for the regression task (predicting Final_Score).
    """
    X_encoded, feature_columns = encode_categoricals(df)
    y = df[REGRESSION_TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
        "feature_columns": feature_columns,
    }


def prepare_classification_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Prepare train/test splits for the classification task (predicting Risk_Level).
    """
    X_encoded, feature_columns = encode_categoricals(df)

    label_encoder = LabelEncoder()
    label_encoder.fit(RISK_CLASSES)
    y = label_encoder.transform(df[CLASSIFICATION_TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_columns": feature_columns,
        "label_encoder": label_encoder,
    }


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "data", "academiciq_dataset.csv")
    df = clean_data(load_data(csv_path))
    reg_data = prepare_regression_data(df)
    clf_data = prepare_classification_data(df)
    print("Preprocessing smoke test complete.")
    print(f"Regression train/test: {reg_data['X_train'].shape} / {reg_data['X_test'].shape}")
    print(f"Classification train/test: {clf_data['X_train'].shape} / {clf_data['X_test'].shape}")
