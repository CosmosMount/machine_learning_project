# DSAA2011 Project Planning

## 1. Project Objective
Build a reproducible and interpretable multiclass classification pipeline to predict student outcomes (`Dropout`, `Graduate`, `Enrolled`).

Primary direction:
- Supervised multiclass prediction as the core deliverable.
- Clustering and low-dimensional visualization as supporting analysis.

## 2. Dataset Detailed Description
Source:
- `data/data.csv`
- Delimiter: `;`

Scale:
- 4424 rows
- 37 columns (36 features + 1 target)

Target (`Target`) distribution:
- `Graduate`: 2209 (49.9%)
- `Dropout`: 1421 (32.1%)
- `Enrolled`: 794 (18.0%)

Data quality:
- Missing values: 0
- Duplicate rows: 0

Feature composition:
- Raw dtype parse: 36 numeric, 0 string categorical (because categories are integer-encoded)
- Semantic modeling split (recommended): 18 categorical-encoded features + 18 numeric-continuous features

Key numeric ranges:
- `Age at enrollment`: 17 to 70
- `Admission grade`: 95.0 to 190.0
- Semester grade-related columns: 0.0 to 18.57
- `Unemployment rate`: 7.6 to 16.2
- `Inflation rate`: -0.8 to 3.7
- `GDP`: -4.06 to 3.51

Suggested feature taxonomy:
- Demographic: marital status, nationality, gender, age, displaced/international flags
- Socioeconomic: parent background, scholarship, debtor, tuition-up-to-date
- Academic progression: application and qualification history, semester performance bundles
- Macroeconomic: unemployment, inflation, GDP

Caveats:
- Column `Nacionality` appears to be a typo of "Nationality".
- Several integer fields are categorical and should not be treated as continuous values.
- Class imbalance exists for `Enrolled` (minority class), so macro-averaged metrics are preferred.

## 3. Implementation Plan
1. Baseline audit and task definition
- Verify class balance, schema, and potential leakage risks.
- Fix the primary objective metric as `F1-macro`.

2. Preprocessing pipeline
- Numeric: median imputation + standardization
- Categorical: most-frequent imputation + one-hot encoding
- Use one shared `Pipeline`/`ColumnTransformer` across all models

3. Visualization and representation
- Class distribution plots
- Numeric distribution checks
- t-SNE 2D embedding for overlap/separation evidence

4. Clustering analysis (supporting)
- K-Means and Agglomerative clustering
- Metrics: Silhouette, Calinski-Harabasz, Davies-Bouldin, ARI, NMI

5. Supervised baselines
- 70/30 stratified split
- Logistic Regression and Decision Tree
- Evaluate on Train/Test/All splits with macro metrics

6. Diagnostic evaluation
- Confusion matrices
- Classification report
- Multiclass ROC/AUC (OVR, macro)

7. Hyperparameter tuning
- `GridSearchCV` with `StratifiedKFold`
- Optimize by `f1_macro`

8. Open-ended extension
- Add Random Forest
- Compare cross-validation mean/std and inspect feature importance

9. Final model decision
- Choose model based on performance + stability + interpretability

## 4. Verification Checklist
- Stratified splits include all 3 classes.
- No data leakage from preprocessing.
- Metrics include Accuracy, Precision-macro, Recall-macro, F1-macro, AUC-macro.
- Overfitting assessed via Train/Test/CV comparison.
- Final recommendation is evidence-backed.

## 5. Deliverables
- Notebook implementation: `project_tasks.ipynb`
- Planning document: `planning.md`
- Supporting analysis notes: `analysis.md`
