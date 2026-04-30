# Dataset: Predict Students' Dropout and Academic Success

## Overview
This dataset contains demographic, academic, and socio-economic information about students, with the goal of predicting academic outcomes.

**Target variable:**
- Dropout
- Enrolled
- Graduate

---

## Features

| Feature | Type | Description |
|--------|------|-------------|
| Marital Status | Categorical (int) | Student's marital status at enrollment (e.g., single, married). |
| Application mode | Categorical (int) | Method used to apply to the program (e.g., general admission, special regime). |
| Application order | Numerical (int) | Preference ranking of the selected course during application (1 = first choice). |
| Course | Categorical (int) | Academic program/course the student enrolled in. |
| Daytime/evening attendance | Categorical (binary) | Indicates whether the student attends classes during daytime or evening. |
| Previous qualification | Categorical (int) | Type of qualification obtained before enrolling (e.g., high school, degree). |
| Previous qualification (grade) | Numerical | Final grade of the previous qualification. |
| Nacionality | Categorical (int) | Student’s nationality. |
| Mother's qualification | Categorical (int) | Educational level of the student's mother. |
| Father's qualification | Categorical (int) | Educational level of the student's father. |
| Mother's occupation | Categorical (int) | Occupation category of the student's mother. |
| Father's occupation | Categorical (int) | Occupation category of the student's father. |
| Admission grade | Numerical | Admission score used for entry into the program. |
| Displaced | Binary | Indicates whether the student is displaced (e.g., relocated from home region). |
| Educational special needs | Binary | Indicates if the student has special educational needs. |
| Debtor | Binary | Indicates whether the student has outstanding financial debt. |
| Tuition fees up to date | Binary | Whether tuition payments are current. |
| Gender | Binary | Student's gender. |
| Scholarship holder | Binary | Indicates if the student receives a scholarship. |
| Age at enrollment | Numerical (int) | Age of the student at the time of enrollment. |
| International | Binary | Indicates whether the student is an international student. |

---

## Academic Performance (1st Semester)

| Feature | Type | Description |
|--------|------|-------------|
| Curricular units 1st sem (credited) | Numerical | Number of credited units transferred into the first semester. |
| Curricular units 1st sem (enrolled) | Numerical | Number of curricular units the student enrolled in. |
| Curricular units 1st sem (evaluations) | Numerical | Number of evaluation instances (exams, assessments). |
| Curricular units 1st sem (approved) | Numerical | Number of units successfully passed. |
| Curricular units 1st sem (grade) | Numerical | Average grade across completed units. |
| Curricular units 1st sem (without evaluations) | Numerical | Number of units without evaluation (e.g., absent or incomplete). |

---

## Academic Performance (2nd Semester)

| Feature | Type | Description |
|--------|------|-------------|
| Curricular units 2nd sem (credited) | Numerical | Number of credited units transferred into the second semester. |
| Curricular units 2nd sem (enrolled) | Numerical | Number of curricular units enrolled. |
| Curricular units 2nd sem (evaluations) | Numerical | Number of evaluations taken. |
| Curricular units 2nd sem (approved) | Numerical | Number of units passed. |
| Curricular units 2nd sem (grade) | Numerical | Average grade for the semester. |
| Curricular units 2nd sem (without evaluations) | Numerical | Units without evaluation. |

---

## Socio-economic Indicators

| Feature | Type | Description |
|--------|------|-------------|
| Unemployment rate | Numerical | National unemployment rate at the time of enrollment. |
| Inflation rate | Numerical | Inflation rate corresponding to the enrollment period. |
| GDP | Numerical | Gross Domestic Product indicator (macroeconomic context). |

---

## Target

| Feature | Type | Description |
|--------|------|-------------|
| Target | Categorical | Final academic outcome: Dropout, Enrolled, or Graduate. |

---