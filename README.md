# 🧠 AcademicIQ — Academic Performance Intelligence System

A production-ready, AI-powered educational analytics platform that analyzes student performance, predicts future outcomes, detects academic risk, identifies weak subjects, and generates personalized recommendations and downloadable reports — built for schools, colleges, universities, and coaching institutes.

---

## 📌 Project Abstract

AcademicIQ is an end-to-end academic intelligence platform that goes beyond simple score prediction. It combines two machine learning pipelines — a **regression model** that predicts a student's final academic score and a **classification model** that predicts academic risk level — with a **rule-based analytics engine** that computes an Academic Health Score, a Performance Index (strength, improvement potential, consistency), and subject-level intelligence (strongest/weakest/most-difficult subjects). These outputs feed a **recommendation engine** that generates prioritized, personalized guidance, and a **PDF reporting module** that produces shareable academic reports.

The platform supports multiple education levels — school students (CBSE/ICSE/State Board, Classes 6–12) and college students (Engineering, Arts & Science, Commerce, Diploma, Postgraduate) — on a single unified data schema, using a synthetic dataset of 5,200 realistic student records. The result is delivered through a premium, SaaS-style Streamlit dashboard with interactive Plotly visualizations, replacing what was previously a single-purpose exam-score predictor.

---

## 🧩 Problem Statement

Educational institutions collect large amounts of student data (attendance, assignments, test scores, participation) but rarely convert it into **actionable, individualized insight**. Teachers and academic counselors typically identify at-risk students reactively — after a poor exam result — rather than proactively. There is no simple, unified system that:

1. Works across school and college education levels with different subjects and grading conventions,
2. Predicts both **numeric performance** (score/GPA) and **categorical risk** (Low/Medium/High) from the same underlying data,
3. Explains *why* a student is at risk by identifying specific weak subjects and behavioral factors, and
4. Converts that explanation into **specific, prioritized, actionable recommendations** — automatically, at scale, for every student.

AcademicIQ addresses this gap with a single platform covering prediction, explanation, and recommendation.

---

## 💡 Innovation Highlights

- **Dual-model architecture**: a regression model (Final Score) and a classification model (Risk Level) trained on the same feature set, cross-validated against a transparent rule-based Academic Health Score for explainability.
- **Institution-agnostic schema**: a single 5-subject-slot data model supports School, Engineering, Arts & Science, Commerce, Diploma, and Postgraduate levels without schema duplication, using a shared subject-name mapping layer (`src/config.py`).
- **Explainable AI analytics**: rather than treating the ML models as black boxes, the analytics engine (`src/analytics.py`) recomputes an interpretable Academic Health Score and Performance Index from the same inputs, letting the dashboard show *why* a prediction was made.
- **Dynamic recommendation engine**: recommendations are generated live from the specific combination of a student's attendance, study habits, subject weaknesses, and risk indicators — not a static lookup table.
- **One-click PDF reporting**: every analysis can be exported as a professional, brand-styled PDF report suitable for sharing with students, parents, or academic counselors.
- **Peer benchmarking**: subject scores and final-score predictions are shown against real peer-group averages computed live from the dataset, not hardcoded thresholds.

---

## 🤖 AI Components

| Component | Type | Purpose |
|---|---|---|
| Score Predictor | Regression (Linear Regression / Random Forest Regressor — best selected automatically) | Predicts `Final_Score` (0–100), from which GPA and Grade are deterministically derived |
| Risk Classifier | Classification (Random Forest Classifier) | Predicts `Risk_Level` (Low / Medium / High Risk) with class probabilities |
| Analytics Engine | Rule-based / statistical | Academic Health Score, Performance Index, Subject Intelligence |
| Recommendation Engine | Rule-based expert system | Prioritized, personalized recommendations from analytics output |
| Report Generator | Templated PDF generation (ReportLab) | Converts prediction + analytics + recommendations into a shareable report |

---

## ⚙️ Machine Learning Pipeline

```
1. Data Generation     → generate_dataset.py creates 5,200 synthetic records across
                          6 institution types with subject-wise scores
2. Preprocessing        → src/preprocessing.py cleans data, one-hot encodes categorical
                          fields (Gender, Institution_Type, Board, Department),
                          scales numeric features, and produces separate train/test
                          splits for the regression and classification tasks
3. Regression Training   → Linear Regression + Random Forest Regressor trained to
                          predict Final_Score; best model selected by R²
4. Classification Training → Random Forest Classifier (class-balanced) trained to
                          predict Risk_Level from the same feature set
5. Model Persistence     → Both best models saved as Pickle bundles
                          (models/score_model.pkl, models/risk_model.pkl) with
                          their scalers/encoders and evaluation metrics
6. Analytics Layer       → src/analytics.py derives Academic Health Score,
                          Performance Index, and Subject Intelligence from raw +
                          predicted values
7. Recommendation Layer  → src/recommendations.py converts analytics into
                          prioritized, human-readable guidance
8. Reporting Layer       → src/reporting.py renders a PDF using ReportLab
9. Dashboard             → app.py (Streamlit + Plotly + custom CSS) ties every
                          layer together into an interactive SaaS-style UI
```

---

## 🗂️ Project Architecture

```
academiciq/
├── app.py                       # Streamlit dashboard (Overview, Subjects, Student,
│                                 # Risk, Recommendations, Reports tabs)
├── generate_dataset.py          # Synthetic dataset generator (5,200 records)
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── academiciq_dataset.csv   # Generated dataset
│
├── models/
│   ├── score_model.pkl          # Best regression model bundle (model+scaler+metrics)
│   └── risk_model.pkl           # Risk classifier bundle (model+encoder+metrics)
│
├── reports/                     # Generated PDF reports land here
│
├── src/
│   ├── __init__.py
│   ├── config.py                # Shared institution/subject configuration
│   ├── preprocessing.py         # Cleaning, encoding, train/test split, scaling
│   ├── train.py                 # Trains & compares regression + classification models
│   ├── predict.py                # AcademicIQPredictor — unified prediction API
│   ├── analytics.py             # Academic Health Score, Performance Index, Subject Intel
│   ├── recommendations.py       # Rule-based recommendation engine
│   └── reporting.py             # PDF report generation (ReportLab)
│
└── assets/
    ├── css/
    │   └── style.css            # Custom SaaS-style dashboard styling
    └── images/
```

---

## 🏗️ Architecture Diagram (ASCII)

```
                    ┌───────────────────────────┐
                    │     generate_dataset.py     │
                    │  5,200 synthetic records     │
                    │  (School + 5 college types)  │
                    └──────────────┬────────────────┘
                                   ▼
                    ┌───────────────────────────┐
                    │ data/academiciq_dataset.csv │
                    └──────────────┬────────────────┘
                                   ▼
                    ┌───────────────────────────┐
                    │   src/preprocessing.py       │
                    │ Clean → Encode → Split → Scale│
                    └──────────────┬────────────────┘
                    ┌──────────────┴──────────────┐
                    ▼                              ▼
      ┌──────────────────────────┐   ┌──────────────────────────┐
      │  Regression Models          │   │  Classification Model      │
      │  Linear Regression           │   │  Random Forest Classifier  │
      │  Random Forest Regressor     │   │  (Risk_Level)               │
      │  (src/train.py)              │   │  (src/train.py)             │
      └──────────────┬───────────────┘   └──────────────┬───────────────┘
                    ▼                                  ▼
      ┌──────────────────────────┐   ┌──────────────────────────┐
      │ models/score_model.pkl     │   │ models/risk_model.pkl       │
      └──────────────┬───────────────┘   └──────────────┬───────────────┘
                    └──────────────┬──────────────────┘
                                   ▼
                    ┌───────────────────────────┐
                    │      src/predict.py          │
                    │ AcademicIQPredictor.predict() │
                    └──────────────┬────────────────┘
                                   ▼
              ┌────────────────────┴────────────────────┐
              ▼                                          ▼
┌──────────────────────────┐              ┌──────────────────────────┐
│   src/analytics.py          │              │  src/recommendations.py    │
│ Health Score, Performance    │─────────────▶│ Prioritized recommendations │
│ Index, Subject Intelligence  │              └──────────────┬───────────────┘
└──────────────┬───────────────┘                             │
              ▼                                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                              app.py                                  │
│  Streamlit + Plotly SaaS Dashboard                                   │
│  Overview │ Subject Analytics │ Student Analytics │ Risk │ Recs │ PDF │
└──────────────┬─────────────────────────────────────────────────────┘
              ▼
     ┌───────────────────────────┐
     │   src/reporting.py           │
     │ PDF report (ReportLab)       │
     │ → reports/*.pdf              │
     └───────────────────────────┘
```

---

## 🚀 Installation Guide

### 1. Clone and set up the environment

```bash
git clone <your-repo-url>
cd academiciq
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate the dataset

```bash
python generate_dataset.py
```

### 3. Train the models

```bash
python src/train.py
```

This trains and compares the regression models (Linear Regression, Random Forest) and the Random Forest risk classifier, then saves the best regression model to `models/score_model.pkl` and the classifier to `models/risk_model.pkl`.

### 4. Launch the dashboard

```bash
streamlit run app.py
```

Open the local URL shown in the terminal (typically `http://localhost:8501`).

### 5. (Optional) CLI prediction demo

```bash
python src/predict.py
```

---

## 🎯 Grading & Risk Scales

| Grade | Score Range | | Risk Level | Meaning |
|-------|-------------|-|------------|---------|
| O (Outstanding) | 90–100 | | Low Risk | Student is on track academically |
| A+ | 80–89 | | Medium Risk | Some indicators need attention |
| A | 70–79 | | High Risk | Multiple indicators signal academic difficulty |
| B | 60–69 | | | |
| C | 50–59 | | | |
| D | 40–49 | | | |
| F | Below 40 | | | |

---

## 📊 Model Performance (from the included trained bundles)

| Task | Model | Key Metric |
|---|---|---|
| Final Score Regression | Linear Regression (auto-selected over Random Forest) | R² ≈ 0.54, RMSE ≈ 5.6 |
| Risk Classification | Random Forest Classifier (class-balanced) | Accuracy ≈ 0.88, F1 ≈ 0.88 |

*(Exact values vary slightly by random seed/run; re-run `python src/train.py` to regenerate.)*

---

## 🔭 Future Scope

- **Longitudinal tracking**: store per-student history across terms to compute *true* most-improved-subject and trend lines (currently estimated via `Previous_GPA` as a single historical baseline).
- **Deep learning sequence models** (LSTM/Transformer) for institutions with genuine multi-semester time-series data.
- **Teacher/Institution dashboard**: aggregate views across a whole class or department, not just per-student.
- **Explainability (SHAP/LIME)** integration so the dashboard can show which features drove each individual prediction.
- **Authentication & multi-tenant support** so schools/colleges can host isolated, secured instances.
- **Automated email/SMS alerts** to parents/mentors when a student crosses into High Risk.
- **Real dataset integration** via LMS/ERP connectors (Google Classroom, Moodle, custom SIS APIs) to replace synthetic data with real academic records.
- **Model monitoring & retraining pipeline** (e.g., MLflow + scheduled retraining) to keep predictions accurate as new data arrives.

---

## ❓ 20 AI/ML Interview Questions & Answers (Project-Specific)

**1. Why did AcademicIQ use two separate models instead of one multi-output model?**
Regression (Final_Score) and classification (Risk_Level) are fundamentally different learning tasks with different loss functions and evaluation metrics. Separating them keeps each model simpler, easier to evaluate independently (R²/MAE vs. accuracy/F1), and easier to retrain or swap without affecting the other.

**2. How does the platform support both school and college students with a single schema?**
Every student record uses five generic "subject slots" (Subject_1_Score … Subject_5_Score) with a separate name-mapping layer (`src/config.py`) that maps those slots to real subject names per institution type (e.g., Slot 2 = "Science" for School, "Programming" for Engineering). This keeps the ML feature schema uniform while the UI displays contextually correct subject names.

**3. Why use one-hot encoding for categorical features instead of label encoding?**
Institution_Type, Gender, Board, and Department are nominal categories with no inherent order. Label encoding would impose a false ordinal relationship (e.g., implying "Engineering" > "Commerce"), which could mislead both Linear Regression and tree-based splits. One-hot encoding avoids that assumption.

**4. Why did the Risk Classifier use `class_weight="balanced"`?**
Real-world (and our synthetic) risk distributions are imbalanced — far fewer students are High Risk than Low Risk. Without balancing, the model could achieve high accuracy by simply predicting "Low Risk" for everyone. `class_weight="balanced"` penalizes misclassifying minority classes more heavily during training, improving recall on High/Medium Risk students, who matter most operationally.

**5. How did you avoid a trivially imbalanced synthetic Risk_Level label?**
Risk_Level is derived from a continuous Academic Health Score using **percentile-based thresholds** (bottom 15% → High Risk, next 25% → Medium Risk, remaining 60% → Low Risk) computed across the generated dataset, rather than fixed absolute cutoffs. This guarantees a workable class balance for training regardless of how the underlying feature distributions shift.

**6. What is the Academic Health Score, and how is it different from the ML risk prediction?**
The Academic Health Score is a transparent, rule-based weighted average (35% attendance, 20% assignments, 20% study hours, 25% subject performance) computed independently of the ML classifier. It serves as an explainable cross-check: if the ML model's Risk_Level prediction and the rule-based score strongly disagree, that's a signal worth investigating rather than blindly trusting either one.

**7. How is "Improvement Potential" calculated, and why isn't it just `100 - current_score`?**
Improvement Potential blends the raw headroom (100 minus current average subject score) with the student's Consistency Score, because a student with high variance across subjects has less *realistic* near-term potential than one whose scores are already fairly uniform (consistency suggests stable study habits that can be redirected, not fixed). The formula is `headroom * (consistency/100) * 0.8 + headroom * 0.2`.

**8. How do you identify a student's "weakest subject" vs. "most difficult subject"?**
Weakest subject is simply the subject with the lowest absolute score for that student. Most difficult subject is *relative* — it's the subject where the gap between the peer/class average and the student's score is largest, meaning the student underperforms their peers most severely in that subject specifically, even if it isn't their lowest absolute score.

**9. Why is GPA derived deterministically from Final_Score instead of being predicted by its own model?**
GPA and Final_Score are mathematically related on a fixed 0–100 to 0–10 scale in most institutional grading systems. Training a second regression model to predict GPA independently would be redundant and could introduce inconsistency (e.g., a 72% final score mapping to a GPA that doesn't correspond to 7.2). A deterministic conversion guarantees consistency.

**10. What features would you add if this were deployed with real institutional data?**
Historical multi-semester performance trends, teacher-assigned qualitative remarks, socioeconomic/demographic context (with appropriate fairness safeguards), extracurricular engagement, disciplinary records, and possibly parent engagement indicators — all of which are known correlates of academic outcomes in educational research.

**11. How would you evaluate whether the Recommendation Engine is actually useful, not just plausible-sounding?**
In production, I'd A/B test recommendation delivery against a control group, tracking whether students who received specific recommendations (e.g., "increase study hours") showed measurable improvement in the targeted metric over the following term, and survey counselors/teachers on recommendation relevance.

**12. Why is the recommendation engine rule-based rather than a generative model?**
Rule-based logic is fully deterministic, auditable, and safe for an education context — every recommendation can be traced to a specific, explainable threshold (e.g., "attendance < 75%"). A generative model risks producing plausible-sounding but factually ungrounded or inconsistent advice, which is a poor tradeoff for a system advising real students.

**13. How does the platform handle a brand-new student with no historical data?**
Every prediction only requires the current-term inputs (attendance, study hours, subject scores, etc.) — there's no dependency on prior-term history except `Previous_GPA`, which can be reasonably estimated or set to a class-average default if unavailable, since it's just one of many weighted inputs.

**14. Why use ReportLab instead of a headless browser/HTML-to-PDF tool for reports?**
ReportLab generates PDFs programmatically without needing a browser engine or system-level dependencies, which makes deployment simpler and more portable (works identically in any Python environment, including constrained/serverless deployments) while still allowing full control over layout, tables, and styling.

**15. How do you prevent data leakage between the regression and classification pipelines?**
Each pipeline (`prepare_regression_data`, `prepare_classification_data`) performs its own independent train/test split and its own scaler/encoder fitting strictly on that split's training data. The classification split additionally uses `stratify=y` to preserve class proportions, but no information from either test set is used during either pipeline's training.

**16. Why did Linear Regression outperform Random Forest for Final_Score prediction in this dataset?**
The synthetic Final_Score target was generated primarily as a linear weighted combination of the input features plus Gaussian noise, so the true underlying relationship is close to linear. Random Forest's added flexibility to model non-linear interactions doesn't help — and can even slightly overfit — when the true relationship is already close to linear; this is a good illustration of why model comparison, not assumption, should drive model selection.

**17. How would you detect model drift after deployment?**
Track the distribution of incoming feature values and prediction outputs over time (e.g., via population stability index or simple summary statistics), and periodically compare model predictions against actual end-of-term outcomes once they're available, retraining when performance degrades beyond a defined threshold.

**18. How does the dashboard support peer benchmarking, and why does that matter?**
`get_class_avg_scores()` computes live per-subject averages filtered by the student's Institution_Type directly from the dataset, and the Subject Analytics tab plots the student's scores against that peer average. This contextualizes an absolute score (e.g., "65 in Mathematics") as either strong or weak *relative to peers facing the same curriculum*, which is more actionable than an absolute number alone.

**19. What are the fairness/ethical considerations in an academic risk-prediction system?**
Risk predictions could stigmatize students or bias teacher attention if used punitively rather than supportively; institutions should treat "High Risk" as a trigger for *additional support*, not reduced opportunity. The model should also be periodically audited for performance disparities across gender, institution type, or department to ensure no systematic bias in error rates, and any deployment should keep a human (teacher/counselor) in the loop rather than automating consequential decisions.

**20. How would you scale this system for a college with 50,000 students?**
Move from per-request Pickle model loading to a persistent model-serving layer (e.g., a FastAPI service holding models in memory), batch-score all students on a nightly schedule rather than on-demand, store results in a proper database instead of recomputing on each dashboard load, and paginate/cache the Streamlit dashboard's dataset queries — the current architecture is intentionally simple for a portfolio-scale deployment but each of these swaps is a natural next step.

---

## 🧪 Tech Stack

- **Python 3.10+**
- **Pandas** & **NumPy** — data manipulation
- **Scikit-learn** — regression, classification, preprocessing
- **Matplotlib** & **Seaborn** — static visualizations (available for notebook/EDA use)
- **Plotly** — interactive dashboard visualizations
- **Streamlit** — web application framework
- **ReportLab** — PDF report generation
- **Pickle** — model persistence

---

## 📄 License

This project is provided for educational and portfolio purposes. Feel free to fork, modify, and use it as a base for your own educational analytics projects.
