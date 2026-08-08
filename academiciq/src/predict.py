"""
predict.py
----------
AcademicIQ - Unified prediction interface.

Loads the trained regression (Final_Score) and classification (Risk_Level)
model bundles, and exposes a single AcademicIQPredictor class that:
  - Validates and transforms raw student input
  - Predicts Final_Score, derives GPA and Grade
  - Predicts Risk_Level via the ML classifier
  - Runs the analytics engine and recommendation engine
  - Returns one consolidated result dictionary ready for the dashboard/report
"""

import os
import pickle
import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES  # noqa: E402
from analytics import full_analytics_report  # noqa: E402
from recommendations import generate_recommendations  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGRESSION_MODEL_PATH = os.path.join(BASE_DIR, "models", "score_model.pkl")
CLASSIFICATION_MODEL_PATH = os.path.join(BASE_DIR, "models", "risk_model.pkl")


def score_to_gpa(score: float) -> float:
    return round(float(np.clip(score / 10.0, 0, 10)), 2)


def score_to_grade(score: float) -> str:
    if score >= 90:
        return "O"
    elif score >= 80:
        return "A+"
    elif score >= 70:
        return "A"
    elif score >= 60:
        return "B"
    elif score >= 50:
        return "C"
    elif score >= 40:
        return "D"
    else:
        return "F"


REQUIRED_FIELDS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
RANGE_CHECKS = {
    "Age": (5, 60),
    "Attendance": (0, 100),
    "Study_Hours": (0, 24),
    "Assignment_Completion": (0, 100),
    "Previous_GPA": (0, 10),
    "Participation_Score": (0, 10),
    "Internet_Usage": (0, 24),
    "Sleep_Hours": (0, 24),
    "Subject_1_Score": (0, 100),
    "Subject_2_Score": (0, 100),
    "Subject_3_Score": (0, 100),
    "Subject_4_Score": (0, 100),
    "Subject_5_Score": (0, 100),
}


class AcademicIQPredictor:
    """
    Loads both trained model bundles once and provides a single `predict()`
    entry point returning a consolidated analytics + prediction payload.
    """

    def __init__(self, regression_model_path: str = REGRESSION_MODEL_PATH,
                 classification_model_path: str = CLASSIFICATION_MODEL_PATH):
        self._load_regression_model(regression_model_path)
        self._load_classification_model(classification_model_path)

    def _load_regression_model(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Regression model not found at '{path}'. Run 'python src/train.py' first."
            )
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        self.reg_model = bundle["model"]
        self.reg_model_name = bundle["model_name"]
        self.reg_scaler = bundle["scaler"]
        self.reg_uses_scaler = bundle["uses_scaler"]
        self.reg_feature_columns = bundle["feature_columns"]
        self.reg_metrics = bundle.get("metrics", {})
        self.reg_comparison = bundle.get("comparison", [])

    def _load_classification_model(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Classification model not found at '{path}'. Run 'python src/train.py' first."
            )
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        self.clf_model = bundle["model"]
        self.clf_model_name = bundle["model_name"]
        self.label_encoder = bundle["label_encoder"]
        self.clf_feature_columns = bundle["feature_columns"]
        self.clf_metrics = bundle.get("metrics", {})
        self.clf_confusion_matrix = bundle.get("confusion_matrix", [])

    # ------------------------------------------------------------------
    def _validate_input(self, student: dict):
        missing = set(REQUIRED_FIELDS) - set(student.keys())
        if missing:
            raise ValueError(f"Missing required input fields: {missing}")

        for field, (low, high) in RANGE_CHECKS.items():
            value = student.get(field)
            if value is None or not isinstance(value, (int, float)):
                raise ValueError(f"Field '{field}' must be numeric.")
            if not (low <= value <= high):
                raise ValueError(f"Field '{field}' value {value} is out of expected range [{low}, {high}].")

    def _encode_row(self, student: dict, target_columns: list) -> pd.DataFrame:
        """Build a single-row, one-hot-encoded, column-aligned DataFrame from raw input."""
        numeric_part = {col: student[col] for col in NUMERIC_FEATURES}
        row_df = pd.DataFrame([numeric_part])

        available_cols = [
        col for col in CATEGORICAL_FEATURES
        if col in student and student[col] is not None
        ]

        
        cat_df = pd.DataFrame([{col: student[col] for col in available_cols}])

        cat_encoded = pd.get_dummies(
        cat_df
        )

        combined = pd.concat([row_df, cat_encoded], axis=1)

        for col in target_columns:
            if col not in combined.columns:
                combined[col] = 0
        combined = combined[target_columns]

        return combined

    # ------------------------------------------------------------------
    def predict(self, student: dict, class_avg_scores: list = None) -> dict:
        """
        Run the full AcademicIQ pipeline for a single student.

        Parameters
        ----------
        student : dict
            Must contain all NUMERIC_FEATURES and CATEGORICAL_FEATURES, plus
            'subject_names' (list[str], length 5) for display/analytics purposes.
        class_avg_scores : list[float], optional
            Peer/class average per subject slot, for subject intelligence comparison.

        Returns
        -------
        dict
            Consolidated prediction + analytics + recommendations payload.
        """
        self._validate_input(student)

        # --- Regression: Final Score ---
        reg_row = self._encode_row(student, self.reg_feature_columns)
        reg_input = self.reg_scaler.transform(reg_row) if self.reg_uses_scaler else reg_row
        raw_score = self.reg_model.predict(reg_input)[0]
        final_score = float(np.clip(raw_score, 0, 100))
        gpa = score_to_gpa(final_score)
        grade = score_to_grade(final_score)

        # --- Classification: Risk Level ---
        clf_row = self._encode_row(student, self.clf_feature_columns)
        risk_encoded = self.clf_model.predict(clf_row)[0]
        risk_level = self.label_encoder.inverse_transform([risk_encoded])[0]
        risk_probabilities = self.clf_model.predict_proba(clf_row)[0]
        risk_prob_map = {
            str(cls): round(float(prob), 3)
            for cls, prob in zip(self.label_encoder.classes_, risk_probabilities)
        }

        # --- Analytics engine ---
        subject_names = student.get("subject_names", [
            "Subject_1", "Subject_2", "Subject_3", "Subject_4", "Subject_5"
        ])
        subject_scores = [
            student["Subject_1_Score"], student["Subject_2_Score"], student["Subject_3_Score"],
            student["Subject_4_Score"], student["Subject_5_Score"],
        ]

        analytics_input = {
            "Attendance": student["Attendance"],
            "Study_Hours": student["Study_Hours"],
            "Assignment_Completion": student["Assignment_Completion"],
            "Participation_Score": student["Participation_Score"],
            "Previous_GPA": student["Previous_GPA"],
            "Final_Score": final_score,
            "subject_names": subject_names,
            "subject_scores": subject_scores,
            "class_avg_scores": class_avg_scores,
        }
        analytics = full_analytics_report(analytics_input)

        # --- Recommendations engine ---
        rec_input = {
            "Attendance": student["Attendance"],
            "Study_Hours": student["Study_Hours"],
            "Assignment_Completion": student["Assignment_Completion"],
            "Participation_Score": student["Participation_Score"],
            "Internet_Usage": student["Internet_Usage"],
            "Sleep_Hours": student["Sleep_Hours"],
        }
        recommendations = generate_recommendations(rec_input, analytics)

        return {
            "final_score": round(final_score, 2),
            "gpa": gpa,
            "grade": grade,
            "risk_level": risk_level,
            "risk_probabilities": risk_prob_map,
            "regression_model_used": self.reg_model_name,
            "classification_model_used": self.clf_model_name,
            "analytics": analytics,
            "recommendations": recommendations,
            "subject_names": subject_names,
            "subject_scores": subject_scores,
        }


def main():
    """CLI demo."""
    predictor = AcademicIQPredictor()

    sample_student = {
        "Age": 20,
        "Attendance": 72.0,
        "Study_Hours": 3.5,
        "Assignment_Completion": 68.0,
        "Previous_GPA": 6.2,
        "Participation_Score": 5.0,
        "Internet_Usage": 4.5,
        "Sleep_Hours": 6.5,
        "Subject_1_Score": 65.0,
        "Subject_2_Score": 58.0,
        "Subject_3_Score": 72.0,
        "Subject_4_Score": 45.0,
        "Subject_5_Score": 70.0,
        "Gender": "Female",
        "Institution_Type": "Engineering",
        "Board": "N/A",
        "Department": "Computer Science",
        "subject_names": ["Engineering_Mathematics", "Programming", "Data_Structures", "DBMS", "Operating_Systems"],
    }

    result = predictor.predict(sample_student)
    print(f"Final Score: {result['final_score']} | GPA: {result['gpa']} | Grade: {result['grade']}")
    print(f"Risk Level: {result['risk_level']} | Probabilities: {result['risk_probabilities']}")
    print(f"Academic Health Score: {result['analytics']['academic_health_score']}")
    print(f"Weakest Subject: {result['analytics']['subject_intelligence']['weakest_subject']}")
    print("\nRecommendations:")
    for r in result["recommendations"]:
        print(f"  [{r['priority']}] ({r['category']}) {r['message']}")


if __name__ == "__main__":
    main()
