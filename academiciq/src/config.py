"""
config.py
---------
AcademicIQ - Shared configuration constants used across dataset generation,
preprocessing, prediction, and the Streamlit dashboard, so institution/subject
definitions stay in exactly one place.
"""

INSTITUTION_TYPES = ["School", "Engineering", "Arts_Science", "Commerce", "Diploma", "Postgraduate"]

SCHOOL_BOARDS = ["CBSE", "ICSE", "State_Board"]
SCHOOL_CLASSES = [f"Class {i}" for i in range(6, 13)]

COLLEGE_DEPARTMENTS = {
    "Engineering": ["Computer Science", "Electronics", "Mechanical", "Civil", "Electrical"],
    "Arts_Science": ["Physics", "Chemistry", "Mathematics", "English Literature", "Psychology"],
    "Commerce": ["B.Com General", "Accounting & Finance", "Banking", "Economics"],
    "Diploma": ["Polytechnic - Mechanical", "Polytechnic - Civil", "Polytechnic - Computer"],
    "Postgraduate": ["MBA", "M.Sc Computer Science", "M.Com", "M.A Economics"],
}

COLLEGE_SEMESTERS = [f"Semester {i}" for i in range(1, 9)]
PG_SEMESTERS = [f"Semester {i}" for i in range(1, 5)]

# Each institution type maps its 5 fixed "subject slots" to real display names.
SUBJECT_NAME_MAPS = {
    "School": ["Mathematics", "Science", "Social_Science", "English", "Second_Language"],
    "Engineering": ["Engineering_Mathematics", "Programming", "Data_Structures", "DBMS", "Operating_Systems"],
    "Arts_Science": ["Core_Subject_1", "Core_Subject_2", "Core_Subject_3", "Language", "Elective"],
    "Commerce": ["Accountancy", "Business_Studies", "Economics", "Statistics", "Elective"],
    "Diploma": ["Core_Subject_1", "Core_Subject_2", "Laboratory", "Workshop", "Elective"],
    "Postgraduate": ["Core_Subject_1", "Core_Subject_2", "Research_Methodology", "Elective", "Seminar"],
}

GENDERS = ["Male", "Female", "Other"]


def get_class_semester_options(institution_type: str) -> list:
    """Return the valid Class/Semester options for a given institution type."""
    if institution_type == "School":
        return SCHOOL_CLASSES
    elif institution_type == "Postgraduate":
        return PG_SEMESTERS
    else:
        return COLLEGE_SEMESTERS


def get_board_options(institution_type: str) -> list:
    """Return valid board options (School only; N/A for college types)."""
    if institution_type == "School":
        return SCHOOL_BOARDS
    return ["N/A"]


def get_department_options(institution_type: str) -> list:
    """Return valid department options for a given institution type."""
    if institution_type == "School":
        return ["General"]
    return COLLEGE_DEPARTMENTS.get(institution_type, ["General"])
