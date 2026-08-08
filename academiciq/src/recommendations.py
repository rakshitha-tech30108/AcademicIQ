"""
recommendations.py
-------------------
AcademicIQ - Rule-based AI Recommendation Engine.

Generates personalized, prioritized recommendations for a student based on
their academic profile, analytics results, and subject intelligence.
"""


def _priority_rank(priority: str) -> int:
    order = {"High": 0, "Medium": 1, "Low": 2}
    return order.get(priority, 3)


def generate_recommendations(student: dict, analytics: dict) -> list:
    """
    Generate a prioritized list of recommendations.

    Parameters
    ----------
    student : dict
        Raw student profile (Attendance, Study_Hours, Assignment_Completion,
        Participation_Score, Internet_Usage, Sleep_Hours, etc.)
    analytics : dict
        Output of analytics.full_analytics_report(student).

    Returns
    -------
    list[dict]
        Each item: {"category": str, "message": str, "priority": "High"/"Medium"/"Low"}
    """
    recs = []

    # --- Attendance ---
    attendance = student.get("Attendance", 100)
    if attendance < 65:
        recs.append({
            "category": "Attendance",
            "message": f"Attendance is critically low at {attendance:.1f}%. Immediate improvement is "
                       "required to avoid academic penalties and knowledge gaps.",
            "priority": "High",
        })
    elif attendance < 75:
        recs.append({
            "category": "Attendance",
            "message": f"Attendance is below the recommended 75% threshold ({attendance:.1f}%). "
                       "Aim to attend classes more consistently to stay aligned with the syllabus pace.",
            "priority": "Medium",
        })

    # --- Study Habits ---
    study_hours = student.get("Study_Hours", 0)
    if study_hours < 2:
        recs.append({
            "category": "Study_Habits",
            "message": f"Weekly study time is very low ({study_hours:.1f} hrs/day). Increase daily study "
                       "hours gradually to at least 3-4 hours for measurable improvement.",
            "priority": "High",
        })
    elif study_hours < 4:
        recs.append({
            "category": "Study_Habits",
            "message": f"Study hours ({study_hours:.1f} hrs/day) are moderate. Increasing focused study time "
                       "by even 1 extra hour a day can noticeably raise performance.",
            "priority": "Medium",
        })

    # --- Assignment Completion ---
    assignment_completion = student.get("Assignment_Completion", 100)
    if assignment_completion < 60:
        recs.append({
            "category": "Assignments",
            "message": f"Assignment completion is low ({assignment_completion:.1f}%). Prioritize completing "
                       "pending assignments — they directly reinforce classroom learning and grades.",
            "priority": "High",
        })
    elif assignment_completion < 80:
        recs.append({
            "category": "Assignments",
            "message": f"Assignment completion ({assignment_completion:.1f}%) has room to improve. "
                       "Try setting a fixed daily slot for assignment work.",
            "priority": "Medium",
        })

    # --- Participation ---
    participation = student.get("Participation_Score", 10)
    if participation < 4:
        recs.append({
            "category": "Classroom_Engagement",
            "message": "Classroom participation is low. Actively engaging in discussions and Q&A "
                       "improves retention and often reflects positively in internal assessments.",
            "priority": "Medium",
        })

    # --- Sleep ---
    sleep_hours = student.get("Sleep_Hours", 8)
    if sleep_hours < 6:
        recs.append({
            "category": "Wellbeing",
            "message": f"Sleep duration ({sleep_hours:.1f} hrs) is below the recommended range. "
                       "Insufficient sleep can reduce concentration and long-term retention.",
            "priority": "Medium",
        })
    elif sleep_hours > 9.5:
        recs.append({
            "category": "Wellbeing",
            "message": f"Sleep duration ({sleep_hours:.1f} hrs) is unusually high. Consider evaluating "
                       "daily routine balance between rest and productive study time.",
            "priority": "Low",
        })

    # --- Internet Usage ---
    internet_usage = student.get("Internet_Usage", 0)
    if internet_usage > 6:
        recs.append({
            "category": "Digital_Wellbeing",
            "message": f"Recreational internet usage is high ({internet_usage:.1f} hrs/day). Reducing "
                       "this by even 1-2 hours daily could be redirected toward focused study.",
            "priority": "Medium",
        })

    # --- Subject-specific recommendations from analytics ---
    subj_intel = analytics.get("subject_intelligence", {})
    weakest = subj_intel.get("weakest_subject")
    weakest_score = subj_intel.get("weakest_score")
    if weakest and weakest_score is not None and weakest_score < 60:
        recs.append({
            "category": "Subject_Improvement",
            "message": f"Focus additional effort on {weakest.replace('_', ' ')} "
                       f"(current score: {weakest_score}). Consider extra practice sessions or tutoring support.",
            "priority": "High" if weakest_score < 45 else "Medium",
        })

    difficult = subj_intel.get("most_difficult_subject")
    if difficult and difficult != weakest:
        recs.append({
            "category": "Subject_Improvement",
            "message": f"{difficult.replace('_', ' ')} is an area where performance trails peers. "
                       "Targeted revision or peer study groups may help close the gap.",
            "priority": "Medium",
        })

    # --- Overall academic growth ---
    health_score = analytics.get("academic_health_score", 100)
    if health_score < 50:
        recs.append({
            "category": "Academic_Growth",
            "message": "Overall academic health is in the concerning range. A structured improvement "
                       "plan covering attendance, assignments, and study consistency is strongly recommended.",
            "priority": "High",
        })
    elif health_score < 70:
        recs.append({
            "category": "Academic_Growth",
            "message": "There is meaningful room to strengthen overall academic performance. "
                       "Small, consistent improvements across attendance and study habits will compound over time.",
            "priority": "Medium",
        })

    consistency = analytics.get("performance_index", {}).get("consistency_score", 100)
    if consistency < 50:
        recs.append({
            "category": "Consistency",
            "message": "Performance varies significantly across subjects. Balancing effort more evenly "
                       "across all subjects may improve overall consistency and final scores.",
            "priority": "Medium",
        })

    if not recs:
        recs.append({
            "category": "General",
            "message": "Performance across all tracked indicators is strong. Maintain current habits "
                       "and continue monitoring progress regularly.",
            "priority": "Low",
        })

    # Sort by priority: High -> Medium -> Low
    recs.sort(key=lambda r: _priority_rank(r["priority"]))

    return recs
