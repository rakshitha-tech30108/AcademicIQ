"""
analytics.py
------------
AcademicIQ - AI Analytics Engine.

Computes derived academic intelligence from a student's raw profile and
subject scores:
  - Academic Health Score (0-100)
  - Rule-based Risk category (used as a sanity cross-check against the ML classifier)
  - Performance Index (Overall Academic Strength, Improvement Potential, Consistency Score)
  - Subject Intelligence (strongest/weakest/most difficult relative-to-class subject, insights)
"""

import numpy as np


def compute_academic_health_score(attendance: float, assignment_completion: float,
                                    study_hours: float, avg_subject_score: float,
                                    participation_score: float = None) -> float:
    """
    Compute a 0-100 Academic Health Score from key indicators.

    Weighting:
        35% Attendance
        20% Assignment Completion
        20% Study Hours (normalized against an 8-hour healthy benchmark)
        25% Average Subject Performance
    If participation_score (0-10) is supplied, it nudges the score slightly
    (+/- up to 3 points) without changing the primary weighting contract.
    """
    study_component = min(study_hours / 8.0 * 100, 100)
    health = (
        0.35 * attendance
        + 0.20 * assignment_completion
        + 0.20 * study_component
        + 0.25 * avg_subject_score
    )
    if participation_score is not None:
        health += (participation_score - 5) * 0.3  # small nudge, +/-1.5 max around baseline 5

    return round(float(np.clip(health, 0, 100)), 2)


def rule_based_risk(health_score: float) -> str:
    """
    Rule-based risk classification (0-100 health score -> category).
    Serves as an explainable cross-check against the ML classifier's prediction.
    """
    if health_score >= 70:
        return "Low_Risk"
    elif health_score >= 50:
        return "Medium_Risk"
    else:
        return "High_Risk"


def compute_performance_index(subject_scores: list, previous_gpa: float, final_score: float) -> dict:
    """
    Compute a Performance Index with three sub-metrics:
      - Overall Academic Strength: average subject performance (0-100)
      - Improvement Potential: headroom between current performance and 100,
        weighted by consistency (more room + more consistent = more realistic potential)
      - Consistency Score: inverse of the standard deviation across subjects (0-100,
        higher = more consistent performance across subjects)

    Returns
    -------
    dict
    """
    subject_scores = np.array(subject_scores, dtype=float)
    overall_strength = float(np.mean(subject_scores))

    std_dev = float(np.std(subject_scores))
    # Convert std dev (typically 0-30 in this domain) into a 0-100 consistency score
    consistency_score = float(np.clip(100 - (std_dev * 3.0), 0, 100))

    headroom = 100 - overall_strength
    improvement_potential = float(np.clip(headroom * (consistency_score / 100) * 0.8 + headroom * 0.2, 0, 100))

    return {
        "overall_academic_strength": round(overall_strength, 2),
        "improvement_potential": round(improvement_potential, 2),
        "consistency_score": round(consistency_score, 2),
    }


def subject_intelligence(subject_names: list, subject_scores: list, class_avg_scores: list = None) -> dict:
    """
    Identify the strongest, weakest, and most-difficult-relative-to-class subjects,
    and generate natural-language insight strings.

    Parameters
    ----------
    subject_names : list[str]
    subject_scores : list[float]
    class_avg_scores : list[float], optional
        Average score for each subject across the peer group/class, used to
        detect subjects that are difficult relative to the overall class average.

    Returns
    -------
    dict
    """
    scores = np.array(subject_scores, dtype=float)
    names = list(subject_names)

    strongest_idx = int(np.argmax(scores))
    weakest_idx = int(np.argmin(scores))

    strongest_subject = names[strongest_idx]
    weakest_subject = names[weakest_idx]

    insights = []
    overall_avg = float(np.mean(scores))

    if scores[strongest_idx] - overall_avg >= 10:
        insights.append(
            f"{strongest_subject.replace('_', ' ')} performance is significantly higher than the overall average."
        )
    if overall_avg - scores[weakest_idx] >= 10:
        insights.append(
            f"{weakest_subject.replace('_', ' ')} requires additional attention and focused practice."
        )

    most_difficult_subject = None
    if class_avg_scores is not None and len(class_avg_scores) == len(scores):
        class_avg = np.array(class_avg_scores, dtype=float)
        relative_gap = class_avg - scores  # positive = student is below class average
        difficult_idx = int(np.argmax(relative_gap))
        if relative_gap[difficult_idx] > 5:
            most_difficult_subject = names[difficult_idx]
            insights.append(
                f"{most_difficult_subject.replace('_', ' ')} is being outperformed by peers on average — "
                "consider prioritizing this subject."
            )

    if not insights:
        insights.append("Performance is fairly consistent across all subjects with no major outliers.")

    return {
        "strongest_subject": strongest_subject,
        "strongest_score": round(float(scores[strongest_idx]), 2),
        "weakest_subject": weakest_subject,
        "weakest_score": round(float(scores[weakest_idx]), 2),
        "most_difficult_subject": most_difficult_subject,
        "insights": insights,
    }


def most_improved_subject(current_scores: dict, historical_avg: float) -> str:
    """
    Identify the subject with the greatest positive deviation from the
    student's historical average performance (proxy for 'most improved'
    since we do not track true longitudinal per-subject history here).

    Parameters
    ----------
    current_scores : dict[str, float]
        Mapping of subject name -> current score.
    historical_avg : float
        The student's historical average performance (e.g., Previous_GPA * 10).

    Returns
    -------
    str
        Name of the subject showing the largest improvement over the historical baseline.
    """
    deltas = {name: score - historical_avg for name, score in current_scores.items()}
    return max(deltas, key=deltas.get)


def full_analytics_report(student: dict) -> dict:
    """
    Convenience function: run the full analytics suite for a single student
    profile dictionary (as produced by predict.py / the Streamlit app).

    Expected keys in `student`:
        Attendance, Study_Hours, Assignment_Completion, Participation_Score,
        Previous_GPA, Final_Score (predicted or actual),
        subject_names (list), subject_scores (list)
    """
    avg_subject_score = float(np.mean(student["subject_scores"]))

    health_score = compute_academic_health_score(
        attendance=student["Attendance"],
        assignment_completion=student["Assignment_Completion"],
        study_hours=student["Study_Hours"],
        avg_subject_score=avg_subject_score,
        participation_score=student.get("Participation_Score"),
    )

    risk = rule_based_risk(health_score)

    perf_index = compute_performance_index(
        subject_scores=student["subject_scores"],
        previous_gpa=student.get("Previous_GPA", 6.0),
        final_score=student.get("Final_Score", avg_subject_score),
    )

    subj_intel = subject_intelligence(
        subject_names=student["subject_names"],
        subject_scores=student["subject_scores"],
        class_avg_scores=student.get("class_avg_scores"),
    )

    current_scores_map = dict(zip(student["subject_names"], student["subject_scores"]))
    historical_avg = student.get("Previous_GPA", 6.0) * 10
    improved_subject = most_improved_subject(current_scores_map, historical_avg)

    return {
        "academic_health_score": health_score,
        "rule_based_risk": risk,
        "performance_index": perf_index,
        "subject_intelligence": subj_intel,
        "most_improved_subject": improved_subject,
        "average_subject_score": round(avg_subject_score, 2),
    }
