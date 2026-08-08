"""
reporting.py
------------
AcademicIQ - PDF Report Generator.

Builds a professional, downloadable PDF report for a student containing:
  - Student profile
  - GPA / Final Score prediction
  - Subject-wise analysis
  - Risk analysis
  - Personalized recommendations

Uses ReportLab (no external binary dependencies), so it works in any
standard Python environment.
"""

import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

BRAND_COLOR = colors.HexColor("#4F46E5")
LIGHT_BG = colors.HexColor("#EEF2FF")
RISK_COLORS = {
    "Low_Risk": colors.HexColor("#16A34A"),
    "Medium_Risk": colors.HexColor("#D97706"),
    "High_Risk": colors.HexColor("#DC2626"),
}


def _build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="AIQTitle", fontSize=22, leading=26, textColor=BRAND_COLOR,
        alignment=TA_CENTER, spaceAfter=6, fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        name="AIQSubtitle", fontSize=11, leading=14, textColor=colors.grey,
        alignment=TA_CENTER, spaceAfter=16,
    ))
    styles.add(ParagraphStyle(
        name="AIQSection", fontSize=14, leading=18, textColor=BRAND_COLOR,
        spaceBefore=14, spaceAfter=8, fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        name="AIQBody", fontSize=10, leading=14, alignment=TA_LEFT,
    ))
    return styles


def generate_student_report(student: dict, result: dict, output_path: str = None) -> str:
    """
    Generate a PDF report for a student and save it to disk.

    Parameters
    ----------
    student : dict
        Raw student profile (must include Name, Student_ID/identifier fields
        used for display, and the same fields consumed by predict.py).
    result : dict
        Output of AcademicIQPredictor.predict(student).
    output_path : str, optional
        Full path to save the PDF. If not given, a path is auto-generated
        under reports/.

    Returns
    -------
    str
        The path to the generated PDF file.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)

    if output_path is None:
        safe_name = str(student.get("Name", "student")).replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(REPORTS_DIR, f"AcademicIQ_Report_{safe_name}_{timestamp}.pdf")

    styles = _build_styles()
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
    )
    story = []

    # --- Header ---
    story.append(Paragraph("AcademicIQ", styles["AIQTitle"]))
    story.append(Paragraph("Academic Performance Intelligence Report", styles["AIQSubtitle"]))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%d %B %Y, %I:%M %p')}", styles["AIQBody"]))
    story.append(Spacer(1, 12))

    # --- Student Profile ---
    story.append(Paragraph("Student Profile", styles["AIQSection"]))
    profile_data = [
        ["Name", student.get("Name", "N/A"), "Student ID", student.get("Student_ID", "N/A")],
        ["Age", str(student.get("Age", "N/A")), "Gender", student.get("Gender", "N/A")],
        ["Institution Type", student.get("Institution_Type", "N/A"), "Department", student.get("Department", "N/A")],
        ["Board", student.get("Board", "N/A"), "Class / Semester", student.get("Class_Semester", "N/A")],
    ]
    profile_table = Table(profile_data, colWidths=[3.5 * cm, 4.5 * cm, 3.5 * cm, 4.5 * cm])
    profile_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT_BG),
        ("BACKGROUND", (2, 0), (2, -1), LIGHT_BG),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(profile_table)
    story.append(Spacer(1, 10))

    # --- Prediction Summary ---
    story.append(Paragraph("Performance Prediction", styles["AIQSection"]))
    risk_color = RISK_COLORS.get(result["risk_level"], colors.black)
    pred_data = [
        ["Final Score", f"{result['final_score']} / 100"],
        ["GPA", f"{result['gpa']} / 10"],
        ["Grade", result["grade"]],
        ["Risk Level", result["risk_level"].replace("_", " ")],
        ["Academic Health Score", f"{result['analytics']['academic_health_score']} / 100"],
    ]
    pred_table = Table(pred_data, colWidths=[6 * cm, 10 * cm])
    pred_style = TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT_BG),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TEXTCOLOR", (1, 3), (1, 3), risk_color),
        ("FONTNAME", (1, 3), (1, 3), "Helvetica-Bold"),
    ])
    pred_table.setStyle(pred_style)
    story.append(pred_table)
    story.append(Spacer(1, 10))

    # --- Subject-wise Analysis ---
    story.append(Paragraph("Subject-wise Analysis", styles["AIQSection"]))
    subj_header = ["Subject", "Score", "Performance"]
    subj_rows = [subj_header]
    for name, score in zip(result["subject_names"], result["subject_scores"]):
        level = "Excellent" if score >= 80 else "Good" if score >= 60 else "Needs Improvement"
        subj_rows.append([name.replace("_", " "), f"{score:.1f}", level])
    subj_table = Table(subj_rows, colWidths=[7 * cm, 3 * cm, 6 * cm])
    subj_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_COLOR),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(subj_table)
    story.append(Spacer(1, 10))

    subj_intel = result["analytics"]["subject_intelligence"]
    story.append(Paragraph(
        f"<b>Strongest Subject:</b> {subj_intel['strongest_subject'].replace('_', ' ')} "
        f"({subj_intel['strongest_score']}) &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<b>Weakest Subject:</b> {subj_intel['weakest_subject'].replace('_', ' ')} "
        f"({subj_intel['weakest_score']})",
        styles["AIQBody"]
    ))
    story.append(Spacer(1, 6))
    for insight in subj_intel["insights"]:
        story.append(Paragraph(f"&#8226; {insight}", styles["AIQBody"]))
    story.append(Spacer(1, 10))

    # --- Performance Index ---
    story.append(Paragraph("Performance Index", styles["AIQSection"]))
    perf = result["analytics"]["performance_index"]
    perf_data = [
        ["Overall Academic Strength", f"{perf['overall_academic_strength']} / 100"],
        ["Improvement Potential", f"{perf['improvement_potential']} / 100"],
        ["Consistency Score", f"{perf['consistency_score']} / 100"],
    ]
    perf_table = Table(perf_data, colWidths=[8 * cm, 8 * cm])
    perf_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT_BG),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(perf_table)
    story.append(Spacer(1, 10))

    # --- Risk Analysis ---
    story.append(Paragraph("Risk Analysis", styles["AIQSection"]))
    risk_probs = result["risk_probabilities"]
    risk_rows = [["Risk Category", "Probability"]]
    for cls, prob in risk_probs.items():
        risk_rows.append([cls.replace("_", " "), f"{prob * 100:.1f}%"])
    risk_table = Table(risk_rows, colWidths=[8 * cm, 8 * cm])
    risk_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_COLOR),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 12))

    # --- Recommendations ---
    story.append(PageBreak())
    story.append(Paragraph("Personalized Recommendations", styles["AIQSection"]))
    for rec in result["recommendations"]:
        priority_color = {"High": "#DC2626", "Medium": "#D97706", "Low": "#16A34A"}.get(rec["priority"], "#000000")
        story.append(Paragraph(
            f'<font color="{priority_color}"><b>[{rec["priority"]}]</b></font> '
            f'<b>{rec["category"].replace("_", " ")}:</b> {rec["message"]}',
            styles["AIQBody"]
        ))
        story.append(Spacer(1, 6))

    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "This report was generated automatically by the AcademicIQ Academic Performance "
        "Intelligence System based on a machine learning model trained on academic and "
        "behavioral indicators. It is intended to support — not replace — guidance from "
        "teachers, mentors, and academic counselors.",
        styles["AIQBody"]
    ))

    doc.build(story)
    return output_path
