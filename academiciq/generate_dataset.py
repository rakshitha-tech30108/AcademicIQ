"""
generate_dataset.py
--------------------
AcademicIQ - Academic Performance Intelligence System
Synthetic dataset generator.

Generates a realistic dataset of 5000+ student records spanning multiple
education levels (School: CBSE/ICSE/State Board, and College: Engineering,
Arts & Science, Commerce, Diploma, Postgraduate), including subject-wise
scores, behavioral/academic factors, and derived target variables
(Final Score, GPA, Grade, Risk Level).

Run:
    python generate_dataset.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from config import (  # noqa: E402
    INSTITUTION_TYPES, SCHOOL_BOARDS, SCHOOL_CLASSES, COLLEGE_DEPARTMENTS,
    COLLEGE_SEMESTERS, PG_SEMESTERS, SUBJECT_NAME_MAPS, GENDERS,
)

FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Krishna", "Ishaan", "Rohan",
    "Ananya", "Diya", "Priya", "Aadhya", "Sara", "Myra", "Aanya", "Navya", "Riya", "Kavya",
    "Karthik", "Rahul", "Amit", "Sanjay", "Vikram", "Neha", "Pooja", "Shreya", "Meera", "Divya",
    "Aditi", "Nikhil", "Varun", "Siddharth", "Tanvi", "Ishita", "Om", "Yash", "Kiara", "Zara",
]
LAST_NAMES = [
    "Sharma", "Verma", "Gupta", "Kumar", "Singh", "Patel", "Reddy", "Nair", "Iyer", "Rao",
    "Mehta", "Joshi", "Chopra", "Malhotra", "Kapoor", "Bansal", "Agarwal", "Desai", "Pillai", "Menon",
]


def _generate_name(rng: np.random.Generator) -> str:
    first = FIRST_NAMES[rng.integers(0, len(FIRST_NAMES))]
    last = LAST_NAMES[rng.integers(0, len(LAST_NAMES))]
    return f"{first} {last}"


def _score_to_gpa(score: float) -> float:
    """Convert a 0-100 percentage score into a 0-10 GPA scale."""
    return round(np.clip(score / 10.0, 0, 10), 2)


def _score_to_grade(score: float) -> str:
    """Convert a 0-100 percentage score into a letter grade (O/A+/A/B/C/D/F)."""
    if score >= 90:
        return "O"       # Outstanding
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


def _compute_health_score(attendance, assignment_completion, study_hours, avg_subject_score) -> float:
    """
    Rule-based academic health score (0-100) used both for ground-truth
    risk labeling and for the analytics engine at inference time.
    """
    return (
        0.35 * attendance
        + 0.20 * assignment_completion
        + 0.20 * min(study_hours / 8.0 * 100, 100)
        + 0.25 * avg_subject_score
    )


def generate_academiciq_dataset(n_samples: int = 5200, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate the full AcademicIQ synthetic dataset.

    Returns
    -------
    pd.DataFrame
    """
    rng = np.random.default_rng(random_seed)
    records = []

    for i in range(n_samples):
        student_id = f"STU{i + 1:05d}"
        name = _generate_name(rng)
        age = int(rng.integers(11, 30))
        gender = GENDERS[rng.integers(0, len(GENDERS))]

        institution_type = INSTITUTION_TYPES[rng.integers(0, len(INSTITUTION_TYPES))]

        if institution_type == "School":
            board = SCHOOL_BOARDS[rng.integers(0, len(SCHOOL_BOARDS))]
            class_semester = SCHOOL_CLASSES[rng.integers(0, len(SCHOOL_CLASSES))]
            department = "General"
            age = int(rng.integers(11, 18))
        else:
            board = "N/A"
            department = COLLEGE_DEPARTMENTS[institution_type][
                rng.integers(0, len(COLLEGE_DEPARTMENTS[institution_type]))
            ]
            if institution_type == "Postgraduate":
                class_semester = PG_SEMESTERS[rng.integers(0, len(PG_SEMESTERS))]
                age = int(rng.integers(21, 30))
            else:
                class_semester = COLLEGE_SEMESTERS[rng.integers(0, len(COLLEGE_SEMESTERS))]
                age = int(rng.integers(17, 24))

        # --- Academic / behavioral factors ---
        attendance = float(np.clip(rng.normal(80, 13), 35, 100))
        study_hours = float(np.clip(rng.normal(4.8, 2.1), 0, 12))
        assignment_completion = float(np.clip(rng.normal(74, 19), 0, 100))
        previous_gpa = float(np.clip(rng.normal(6.5, 1.6), 0, 10))
        participation_score = float(np.clip(rng.normal(6, 2.1), 0, 10))
        internet_usage = float(np.clip(rng.normal(3.4, 2.0), 0, 12))
        sleep_hours = float(np.clip(rng.normal(6.9, 1.3), 3, 10))

        # --- Subject-wise scores (5 subject slots), influenced by the
        #     academic factors above plus per-subject noise ---
        base_ability = (
            0.25 * attendance
            + 3.0 * study_hours
            + 0.20 * assignment_completion
            + 4.0 * previous_gpa
            + 1.1 * participation_score
            - 0.9 * internet_usage
        )
        subject_scores = []
        for _ in range(5):
            subj_noise = rng.normal(0, 9)
            subj_score = np.clip(base_ability * 0.55 + subj_noise + rng.normal(20, 5), 0, 100)
            subject_scores.append(round(float(subj_score), 2))

        avg_subject_score = float(np.mean(subject_scores))

        # --- Final Score (percentage) ---
        noise = rng.normal(0, 5.5)
        final_score = (
            0.30 * avg_subject_score
            + 0.20 * attendance
            + 0.15 * assignment_completion
            + 0.15 * (previous_gpa * 10)
            + 0.10 * study_hours * 5
            + 0.10 * participation_score * 5
            - 0.05 * internet_usage * 5
            + noise
        )
        final_score = float(np.clip(final_score, 0, 100))

        gpa = _score_to_gpa(final_score)
        grade = _score_to_grade(final_score)
        health_score = _compute_health_score(attendance, assignment_completion, study_hours, avg_subject_score)

        subj_names = SUBJECT_NAME_MAPS[institution_type]

        record = {
            "Student_ID": student_id,
            "Name": name,
            "Age": age,
            "Gender": gender,
            "Institution_Type": institution_type,
            "Board": board,
            "Department": department,
            "Class_Semester": class_semester,
            "Attendance": round(attendance, 2),
            "Study_Hours": round(study_hours, 2),
            "Assignment_Completion": round(assignment_completion, 2),
            "Previous_GPA": round(previous_gpa, 2),
            "Participation_Score": round(participation_score, 2),
            "Internet_Usage": round(internet_usage, 2),
            "Sleep_Hours": round(sleep_hours, 2),
            "Subject_1_Name": subj_names[0],
            "Subject_1_Score": subject_scores[0],
            "Subject_2_Name": subj_names[1],
            "Subject_2_Score": subject_scores[1],
            "Subject_3_Name": subj_names[2],
            "Subject_3_Score": subject_scores[2],
            "Subject_4_Name": subj_names[3],
            "Subject_4_Score": subject_scores[3],
            "Subject_5_Name": subj_names[4],
            "Subject_5_Score": subject_scores[4],
            "Final_Score": round(final_score, 2),
            "GPA": gpa,
            "Grade": grade,
            "Academic_Health_Score": round(health_score, 2),
        }
        records.append(record)

    df = pd.DataFrame(records)

    # Assign Risk_Level using percentile-based thresholds on the health score
    # so the classification target has a realistic yet learnable class balance:
    # bottom 15% -> High_Risk, next 25% -> Medium_Risk, remaining 60% -> Low_Risk.
    high_cut = df["Academic_Health_Score"].quantile(0.15)
    medium_cut = df["Academic_Health_Score"].quantile(0.40)

    def _assign_risk(h):
        if h <= high_cut:
            return "High_Risk"
        elif h <= medium_cut:
            return "Medium_Risk"
        else:
            return "Low_Risk"

    df["Risk_Level"] = df["Academic_Health_Score"].apply(_assign_risk)

    return df


def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "academiciq_dataset.csv")

    df = generate_academiciq_dataset(n_samples=5200, random_seed=42)
    df.to_csv(output_path, index=False)

    print(f"AcademicIQ synthetic dataset generated with {len(df)} records.")
    print(f"Saved to: {output_path}")
    print("\nInstitution type distribution:")
    print(df["Institution_Type"].value_counts())
    print("\nRisk level distribution:")
    print(df["Risk_Level"].value_counts())
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()
