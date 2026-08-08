"""
app.py
------
AcademicIQ - Academic Performance Intelligence System
Main Streamlit Dashboard.

Run with:
    streamlit run app.py
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src"))

from src.predict import AcademicIQPredictor  # noqa: E402
from src.reporting import generate_student_report  # noqa: E402
from src.config import (  # noqa: E402
    INSTITUTION_TYPES, GENDERS, SUBJECT_NAME_MAPS,
    get_board_options, get_department_options, get_class_semester_options,
)

DATA_PATH = os.path.join(BASE_DIR, "data", "academiciq_dataset.csv")
CSS_PATH = os.path.join(BASE_DIR, "assets", "css", "style.css")

st.set_page_config(
    page_title="AcademicIQ | Academic Performance Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------
# Styling & caching helpers
# --------------------------------------------------------------------------
def load_css():
    if os.path.exists(CSS_PATH):
        with open(CSS_PATH, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
st.markdown("""
<style>
button[data-baseweb="tab"] {
    color: red !important;
    font-size: 24px !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_predictor():
    return AcademicIQPredictor()


@st.cache_data(show_spinner=False)
def load_dataset():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None


def risk_badge_class(risk_level: str) -> str:
    return {
        "Low_Risk": "aiq-badge-low",
        "Medium_Risk": "aiq-badge-medium",
        "High_Risk": "aiq-badge-high",
    }.get(risk_level, "aiq-badge-medium")


def metric_card(label: str, value: str, sub: str = "") -> str:
    return f"""
    <div class="aiq-metric-card">
        <div class="aiq-metric-label">{label}</div>
        <div class="aiq-metric-value">{value}</div>
        <div class="aiq-metric-sub">{sub}</div>
    </div>
    """


def get_class_avg_scores(df: pd.DataFrame, institution_type: str) -> list:
    """Compute peer average score per subject slot for a given institution type."""
    subset = df[df["Institution_Type"] == institution_type]
    if subset.empty:
        return [70.0] * 5
    cols = [f"Subject_{i}_Score" for i in range(1, 6)]
    return [float(subset[c].mean()) for c in cols]


# --------------------------------------------------------------------------
# Sidebar: data source selection
# --------------------------------------------------------------------------
def render_sidebar(df: pd.DataFrame):
    st.sidebar.markdown("## 🧠 AcademicIQ")
    st.sidebar.caption("Academic Performance Intelligence System")
    st.sidebar.divider()

    mode = st.sidebar.radio(
        "Data Source",
        ["Select Existing Student", "Enter New Student Profile"],
        index=0,
    )

    student = None

    if mode == "Select Existing Student" and df is not None:
        institution_filter = st.sidebar.selectbox("Filter by Institution Type", ["All"] + INSTITUTION_TYPES)
        filtered = df if institution_filter == "All" else df[df["Institution_Type"] == institution_filter]

        student_options = filtered["Student_ID"] + " — " + filtered["Name"]
        selected = st.sidebar.selectbox("Select Student", student_options.tolist())
        selected_id = selected.split(" — ")[0]
        row = df[df["Student_ID"] == selected_id].iloc[0]

        student = {
            "Name": row["Name"],
            "Student_ID": row["Student_ID"],
            "Age": int(row["Age"]),
            "Gender": row["Gender"],
            "Institution_Type": row["Institution_Type"],
            "Board": row["Board"],
            "Department": row["Department"],
            "Class_Semester": row["Class_Semester"],
            "Attendance": float(row["Attendance"]),
            "Study_Hours": float(row["Study_Hours"]),
            "Assignment_Completion": float(row["Assignment_Completion"]),
            "Previous_GPA": float(row["Previous_GPA"]),
            "Participation_Score": float(row["Participation_Score"]),
            "Internet_Usage": float(row["Internet_Usage"]),
            "Sleep_Hours": float(row["Sleep_Hours"]),
            "Subject_1_Score": float(row["Subject_1_Score"]),
            "Subject_2_Score": float(row["Subject_2_Score"]),
            "Subject_3_Score": float(row["Subject_3_Score"]),
            "Subject_4_Score": float(row["Subject_4_Score"]),
            "Subject_5_Score": float(row["Subject_5_Score"]),
            "subject_names": [row[f"Subject_{i}_Name"] for i in range(1, 6)],
        }

    else:
        st.sidebar.markdown("### Profile")
        name = st.sidebar.text_input("Student Name", value="New Student")
        age = st.sidebar.slider("Age", 10, 30, 18)
        gender = st.sidebar.selectbox("Gender", GENDERS)
        institution_type = st.sidebar.selectbox("Institution Type", INSTITUTION_TYPES)
        board = st.sidebar.selectbox("Board", get_board_options(institution_type))
        department = st.sidebar.selectbox("Department", get_department_options(institution_type))
        class_semester = st.sidebar.selectbox("Class / Semester", get_class_semester_options(institution_type))

        st.sidebar.markdown("### Academic Factors")
        attendance = st.sidebar.slider("Attendance (%)", 0.0, 100.0, 80.0)
        study_hours = st.sidebar.slider("Study Hours (per day)", 0.0, 12.0, 5.0, 0.5)
        assignment_completion = st.sidebar.slider("Assignment Completion (%)", 0.0, 100.0, 75.0)
        previous_gpa = st.sidebar.slider("Previous GPA (0-10)", 0.0, 10.0, 6.5, 0.1)
        participation_score = st.sidebar.slider("Participation Score (0-10)", 0.0, 10.0, 6.0, 0.5)
        internet_usage = st.sidebar.slider("Internet Usage (hrs/day)", 0.0, 12.0, 3.0, 0.5)
        sleep_hours = st.sidebar.slider("Sleep Hours (per night)", 0.0, 12.0, 7.0, 0.5)

        st.sidebar.markdown("### Subject Scores")
        subj_names = SUBJECT_NAME_MAPS[institution_type]
        subject_scores = []
        for s_name in subj_names:
            score = st.sidebar.slider(f"{s_name.replace('_', ' ')}", 0.0, 100.0, 70.0, key=f"subj_{s_name}")
            subject_scores.append(score)

        student = {
            "Name": name,
            "Student_ID": "NEW_STUDENT",
            "Age": age,
            "Gender": gender,
            "Institution_Type": institution_type,
            "Board": board,
            "Department": department,
            "Class_Semester": class_semester,
            "Attendance": attendance,
            "Study_Hours": study_hours,
            "Assignment_Completion": assignment_completion,
            "Previous_GPA": previous_gpa,
            "Participation_Score": participation_score,
            "Internet_Usage": internet_usage,
            "Sleep_Hours": sleep_hours,
            "Subject_1_Score": subject_scores[0],
            "Subject_2_Score": subject_scores[1],
            "Subject_3_Score": subject_scores[2],
            "Subject_4_Score": subject_scores[3],
            "Subject_5_Score": subject_scores[4],
            "subject_names": subj_names,
        }

    return student


# --------------------------------------------------------------------------
# Tab renderers
# --------------------------------------------------------------------------
def render_overview_tab(student, result):
    st.markdown("#### Key Performance Indicators")
    c1, c2, c3, c4 = st.columns(4)
    analytics = result["analytics"]

    with c1:
        st.markdown(metric_card("Academic Health Score", f"{analytics['academic_health_score']}",
                                 "out of 100"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("GPA Prediction", f"{result['gpa']}",
                                 f"Grade: {result['grade']}"), unsafe_allow_html=True)
    with c3:
        badge = risk_badge_class(result["risk_level"])
        st.markdown(
            f"""<div class="aiq-metric-card">
                    <div class="aiq-metric-label">Risk Level</div>
                    <div style="margin-top:8px;"><span class="aiq-badge {badge}">
                        {result['risk_level'].replace('_',' ')}</span></div>
                    <div class="aiq-metric-sub">ML-based classification</div>
                </div>""",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(metric_card("Attendance", f"{student['Attendance']:.1f}%",
                                 "Current attendance rate"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown('<div class="aiq-card"><h4>Predicted Final Score</h4>', unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result["final_score"],
            number={"suffix": " / 100"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#4F46E5"},
                "steps": [
                    {"range": [0, 40], "color": "#FEE2E2"},
                    {"range": [40, 60], "color": "#FEF3C7"},
                    {"range": [60, 80], "color": "#DBEAFE"},
                    {"range": [80, 100], "color": "#DCFCE7"},
                ],
            },
        ))
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="aiq-card"><h4>Performance Index</h4>', unsafe_allow_html=True)
        perf = analytics["performance_index"]
        idx_fig = go.Figure(go.Bar(
            x=[perf["overall_academic_strength"], perf["improvement_potential"], perf["consistency_score"]],
            y=["Academic Strength", "Improvement Potential", "Consistency"],
            orientation="h",
            marker_color=["#4F46E5", "#7C3AED", "#06B6D4"],
        ))
        idx_fig.update_layout(height=280, xaxis_range=[0, 100], margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(idx_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


def render_subject_analytics_tab(student, result, df):
    st.markdown("#### Subject-wise Performance")
    subj_names = [s.replace("_", " ") for s in result["subject_names"]]
    subj_scores = result["subject_scores"]

    class_avg = get_class_avg_scores(df, student["Institution_Type"]) if df is not None else [70] * 5

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="aiq-card"><h4>Subject Scores vs Class Average</h4>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Student", x=subj_names, y=subj_scores, marker_color="#4F46E5"))
        fig.add_trace(go.Bar(name="Class Average", x=subj_names, y=class_avg, marker_color="#C7D2FE"))
        fig.update_layout(barmode="group", height=380, yaxis_range=[0, 100], margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="aiq-card"><h4>Subject Performance Heatmap</h4>', unsafe_allow_html=True)
        heat_df = pd.DataFrame({"Subject": subj_names, "Score": subj_scores}).set_index("Subject").T
        fig2 = px.imshow(
            heat_df, text_auto=".1f", color_continuous_scale="Blues", aspect="auto",
            labels=dict(color="Score"),
        )
        fig2.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="aiq-card"><h4>Subject Intelligence</h4>', unsafe_allow_html=True)
    subj_intel = result["analytics"]["subject_intelligence"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Strongest Subject", subj_intel["strongest_subject"].replace("_", " "),
               f"{subj_intel['strongest_score']}")
    c2.metric("Weakest Subject", subj_intel["weakest_subject"].replace("_", " "),
               f"{subj_intel['weakest_score']}")
    c3.metric("Most Improved (est.)", result["analytics"]["most_improved_subject"].replace("_", " "))

    for insight in subj_intel["insights"]:
        st.info(insight)
    st.markdown('</div>', unsafe_allow_html=True)


def render_student_analytics_tab(student, result, df):
    st.markdown("#### Student Profile Analysis")
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown('<div class="aiq-card"><h4>Multi-Factor Radar</h4>', unsafe_allow_html=True)
        categories = ["Attendance", "Study Hours", "Assignments", "Participation", "Sleep", "Avg Subject Score"]
        values = [
            student["Attendance"],
            min(student["Study_Hours"] / 12 * 100, 100),
            student["Assignment_Completion"],
            student["Participation_Score"] * 10,
            min(student["Sleep_Hours"] / 10 * 100, 100),
            result["analytics"]["average_subject_score"],
        ]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]],
                                       fill="toself", line_color="#4F46E5", name="Student"))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                           height=420, margin=dict(l=30, r=30, t=30, b=30))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="aiq-card"><h4>Performance Distribution (Peer Comparison)</h4>', unsafe_allow_html=True)
        if df is not None:
            peer_df = df[df["Institution_Type"] == student["Institution_Type"]]
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(x=peer_df["Final_Score"], nbinsx=30, marker_color="#C7D2FE",
                                         name="Peer Distribution"))
            fig2.add_vline(x=result["final_score"], line_width=3, line_dash="dash", line_color="#DC2626",
                            annotation_text="This Student")
            fig2.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                                xaxis_title="Final Score", yaxis_title="Number of Peers")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Dataset not available for peer comparison.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="aiq-card"><h4>Strength Analysis</h4>', unsafe_allow_html=True)
    perf = result["analytics"]["performance_index"]
    st.write(
        f"This student shows an **Overall Academic Strength** of **{perf['overall_academic_strength']}/100**, "
        f"an **Improvement Potential** of **{perf['improvement_potential']}/100**, and a "
        f"**Consistency Score** of **{perf['consistency_score']}/100** across subjects."
    )
    if perf["consistency_score"] < 50:
        st.warning("Performance varies notably across subjects — focus on balancing effort more evenly.")
    else:
        st.success("Performance is fairly consistent across subjects.")
    st.markdown('</div>', unsafe_allow_html=True)


def render_risk_analysis_tab(student, result):
    st.markdown("#### Risk Analysis & Early Warning System")
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown('<div class="aiq-card"><h4>Risk Indicator</h4>', unsafe_allow_html=True)
        badge = risk_badge_class(result["risk_level"])
        st.markdown(
            f'<div style="text-align:center; padding: 1rem 0;">'
            f'<span class="aiq-badge {badge}" style="font-size:1.1rem; padding:0.6rem 1.4rem;">'
            f'{result["risk_level"].replace("_"," ")}</span></div>',
            unsafe_allow_html=True,
        )
        risk_color_map = {"Low_Risk": "#16A34A", "Medium_Risk": "#D97706", "High_Risk": "#DC2626"}
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result["analytics"]["academic_health_score"],
            title={"text": "Academic Health Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": risk_color_map.get(result["risk_level"], "#4F46E5")},
                "steps": [
                    {"range": [0, 50], "color": "#FEE2E2"},
                    {"range": [50, 70], "color": "#FEF3C7"},
                    {"range": [70, 100], "color": "#DCFCE7"},
                ],
            },
        ))
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="aiq-card"><h4>Risk Probability Breakdown</h4>', unsafe_allow_html=True)
        probs = result["risk_probabilities"]
        fig2 = go.Figure(go.Bar(
            x=list(probs.values()), y=[k.replace("_", " ") for k in probs.keys()],
            orientation="h",
            marker_color=["#DC2626" if "High" in k else "#D97706" if "Medium" in k else "#16A34A"
                          for k in probs.keys()],
        ))
        fig2.update_layout(height=280, xaxis_range=[0, 1], xaxis_title="Probability",
                            margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="aiq-card"><h4>Early Warning Signals</h4>', unsafe_allow_html=True)
    high_priority = [r for r in result["recommendations"] if r["priority"] == "High"]
    if high_priority:
        for r in high_priority:
            st.markdown(
                f'<div class="aiq-rec aiq-rec-high"><b>{r["category"].replace("_"," ")}:</b> {r["message"]}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.success("No high-priority risk signals detected for this student.")
    st.markdown('</div>', unsafe_allow_html=True)


def render_recommendations_tab(result):
    st.markdown("#### Personalized Recommendations")
    priority_class = {"High": "aiq-rec-high", "Medium": "aiq-rec-medium", "Low": "aiq-rec-low"}
    for rec in result["recommendations"]:
        cls = priority_class.get(rec["priority"], "aiq-rec-medium")
        st.markdown(
            f'<div class="aiq-card {cls}" style="padding:1rem;">'
            f'<b>[{rec["priority"]}] {rec["category"].replace("_", " ")}</b><br>{rec["message"]}</div>',
            unsafe_allow_html=True,
        )


def render_reports_tab(student, result):
    st.markdown("#### Generate Downloadable Report")
    st.write(
        f"Generate a complete PDF report for **{student['Name']}** including profile, "
        "predictions, subject analysis, risk analysis, and recommendations."
    )
    if st.button("📄 Generate PDF Report", type="primary"):
        try:
            with st.spinner("Generating report..."):
                pdf_path = generate_student_report(student, result)
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            st.success("Report generated successfully.")
            st.download_button(
                "⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name=os.path.basename(pdf_path),
                mime="application/pdf",
            )
        except Exception as e:
            st.error(f"Failed to generate report: {e}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    load_css()

    st.markdown(
        """
        <div class="aiq-header">
            <h1>🧠 AcademicIQ</h1>
            <p>Academic Performance Intelligence System — predictive analytics, risk detection,
            and personalized recommendations for schools, colleges, and coaching institutes.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_dataset()

    try:
        predictor = load_predictor()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info(
            "Run the following from the project root before launching the app:\n\n"
            "```bash\npython generate_dataset.py\npython src/train.py\n```"
        )
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error while loading models: {e}")
        st.stop()

    student = render_sidebar(df)

    if student is None:
        st.warning("Please provide student information in the sidebar.")
        st.stop()

    class_avg_scores = get_class_avg_scores(df, student["Institution_Type"]) if df is not None else None

    try:
        result = predictor.predict(
            {k: v for k, v in student.items()},
            class_avg_scores=class_avg_scores,
        )
    except ValueError as ve:
        st.error(f"Input validation error: {ve}")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while generating predictions: {e}")
        st.stop()

    st.markdown(f"### Analyzing: **{student['Name']}** &nbsp;·&nbsp; "
                f"{student['Institution_Type'].replace('_',' ')} &nbsp;·&nbsp; {student['Class_Semester']}")

    tabs = st.tabs(["📊 Overview", "📚 Subject Analytics", "👤 Student Analytics",
                     "⚠️ Risk Analysis", "💡 Recommendations", "📄 Reports"])

    with tabs[0]:
        render_overview_tab(student, result)
    with tabs[1]:
        render_subject_analytics_tab(student, result, df)
    with tabs[2]:
        render_student_analytics_tab(student, result, df)
    with tabs[3]:
        render_risk_analysis_tab(student, result)
    with tabs[4]:
        render_recommendations_tab(result)
    with tabs[5]:
        render_reports_tab(student, result)

    st.divider()
    st.caption(
        f"Models in use — Regression: {result['regression_model_used']} · "
        f"Classification: {result['classification_model_used']} · "
        f"Report generated {datetime.now().strftime('%d %b %Y, %I:%M %p')}"
    )


if __name__ == "__main__":
    main()
