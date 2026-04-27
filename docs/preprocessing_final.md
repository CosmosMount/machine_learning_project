# Final Data Preprocessing Plan

## 1. Purpose

This document records the final preprocessing strategy used for the Student Dropout dataset. The goal is to build a preprocessing pipeline that is:

- statistically valid,
- reproducible,
- free from target leakage,
- aligned with the DSAA2011 project requirements,
- sufficiently sophisticated for a high-scoring project report.

The final technical implementation is reflected in `project_tasks.ipynb`.

## 2. Dataset Status

Dataset:

- Source file: `data/data.csv`
- Separator: semicolon (`;`)
- Shape: 4424 rows, 37 columns
- Features: 36
- Target column: `Target`
- Classes: `Graduate`, `Dropout`, `Enrolled`

Target distribution:

| Class | Count | Percentage |
|---|---:|---:|
| Graduate | 2209 | 49.9% |
| Dropout | 1421 | 32.1% |
| Enrolled | 794 | 17.9% |

Data quality:

| Check | Result |
|---|---:|
| Missing values | 0 |
| Duplicate rows | 0 |
| Majority/minority class ratio | 2.78 |

The dataset is clean in terms of missingness and duplication, but it has moderate class imbalance. Therefore, preprocessing does not need aggressive cleaning, but downstream evaluation should emphasize macro-averaged metrics.

## 3. Final Preprocessing Pipeline

The final preprocessing pipeline has three branches:

| Feature Type | Treatment | Reason |
|---|---|---|
| Numeric/count features | `SimpleImputer(strategy='median')` + `StandardScaler()` | Keeps features on comparable scale and satisfies the project requirement to standardize features. |
| Nominal categorical features | `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore', drop='first')` | Avoids imposing false ordinal meaning on integer-coded categories and does not use the target label. |
| Binary features | `passthrough` | 0/1 values are already meaningful indicators. |

Final transformed feature matrix:

| Item | Value |
|---|---:|
| Rows | 4424 |
| Raw features after filtering and feature engineering | 33 |
| Numeric features | 20 |
| Nominal categorical raw features | 7 |
| One-hot encoded categorical columns | 146 |
| Binary features | 6 |
| Final transformed columns | 172 |
| NaN / Inf after preprocessing | 0 / 0 |

## 4. Feature Typing

Although all raw columns are numeric in pandas, many are integer-coded categories. Treating all columns as continuous numeric values would be semantically wrong.

### Binary Features

These are kept unchanged:

- `Daytime/evening attendance`
- `Displaced`
- `Educational special needs`
- `Debtor`
- `Tuition fees up to date`
- `Gender`
- `Scholarship holder`
- `International`

After near-zero-variance filtering, the final binary branch keeps 6 binary features.

### Nominal Categorical Features

These are encoded by one-hot encoding:

- `Marital status`
- `Application mode`
- `Course`
- `Previous qualification`
- `Nacionality`
- `Mother's qualification`
- `Father's qualification`
- `Mother's occupation`
- `Father's occupation`

After filtering, the final categorical branch keeps 7 raw nominal features and expands them into 146 encoded columns.

### Numeric / Count Features

The remaining continuous, ordinal-count, grade, and macroeconomic columns are treated as numeric and standardized after imputation.

## 5. Feature Filtering

Two kinds of filtering are applied before model training:

### Near-Zero-Variance Filtering

Features where one value dominates more than 95% of rows are removed:

- `Nacionality`
- `Educational special needs`
- `International`

Reason:

These variables provide very limited predictive variation in this dataset and can introduce unstable sparse one-hot columns. Removing them reduces noise while keeping the pipeline interpretable.

### High-Correlation and Complexity Pruning

The numeric correlation audit flags highly correlated semester-count features. We remove clearly redundant administrative count fields:

- `Curricular units 1st sem (credited)`
- `Curricular units 1st sem (enrolled)`

Reason:

Highly correlated administrative semester-count features can make linear-model coefficients less stable and add redundant information. Removing one feature from these pairs simplifies interpretation without removing the underlying signal.

Domain-aware exception:

- `Curricular units 1st sem (approved)` and `Curricular units 2nd sem (approved)` are highly correlated but retained because they represent academic success at two different time points. This temporal progression is central to dropout risk.

In addition, `Father's occupation` is removed to control high-cardinality sparse one-hot expansion while retaining related family-background information from other parental background variables.

## 6. Feature Engineering

Three academic trajectory features are added:

| New Feature | Formula | Interpretation |
|---|---|---|
| `approval_rate_1st` | `1st sem approved / (1st sem evaluations + 1)` | First-semester academic success rate. |
| `approval_rate_2nd` | `2nd sem approved / (2nd sem evaluations + 1)` | Second-semester academic success rate. |
| `grade_trend` | `2nd sem grade - 1st sem grade` | Whether academic performance improves or declines. |

Reason:

The strongest signals in this dataset are related to academic progression. These engineered features summarize semester-level performance into compact, interpretable indicators.

The `+1` in the denominator avoids division by zero and keeps the transformation stable.

## 7. Alignment With Project Requirements

The DSAA2011 project PDF requires data preprocessing to:

1. Examine and handle missing values.
2. Handle non-numeric values, for example with one-hot encoding or Boolean indicators.
3. Further process features, for example by standardization.
4. Document methods and observed dataset patterns.

The final pipeline satisfies these requirements as follows:

| Requirement | How It Is Satisfied |
|---|---|
| Examine missing values | All columns are checked; total missing values are 0. |
| Handle missing values | Median and most-frequent imputers are included in the pipeline for robustness. |
| Handle non-numeric/categorical information | Integer-coded categorical features are identified semantically and one-hot encoded. |
| Boolean indicators | Binary 0/1 fields are passed through directly. |
| Standardize features | Numeric/count features are standardized by `StandardScaler`. |
| Documentation and insight | Class imbalance, feature types, filtered features, and engineered features are explicitly reported. |

## 8. Why This Is the Main High-Score Choice

This pipeline is selected as the main preprocessing strategy because it balances correctness, complexity, interpretability, and performance.

Advantages:

- It respects the semantic difference between numeric and categorical variables.
- It avoids target leakage in t-SNE and clustering.
- It is easy to explain in the report and presentation.
- It is compatible with Logistic Regression, Decision Tree, Random Forest, and other models.
- It follows the examples and expectations in the project guideline.
- It performs competitively in validation.

Clean preprocessing evaluation results:

| Pipeline | Model | CV F1-macro | Test F1-macro | Test AUC-macro |
|---|---|---:|---:|---:|
| OneHotEncoder + StandardScaler | Logistic Regression balanced | 0.7067 +/- 0.0129 | 0.6941 | 0.8790 |
| OneHotEncoder + StandardScaler | Random Forest balanced | 0.6838 +/- 0.0061 | 0.6936 | 0.8923 |

The score is comparable to the previous target-encoding version, but the final version is safer and more defensible.

## 8.1 Diagnostic Visualizations Added

To make the effect and rationale of preprocessing visible in the notebook and report, three diagnostic figures are generated:

- `docs/figures/task1_class_distribution.png`
- `docs/figures/task1_numeric_correlation_heatmap.png`
- `docs/figures/task1_preprocessing_scale_effect.png`

The notebook keeps the earlier semantic feature-classification step focused on listing binary, nominal, and numeric columns. These three Step 5 figures are the single report-ready source for preprocessing diagnostics, avoiding duplicated class-distribution or scaling plots in the final write-up.

These figures serve different purposes:

| Figure | Purpose |
|---|---|
| Class distribution | Shows the moderate imbalance, especially the minority `Enrolled` class. |
| Numeric correlation heatmap | Shows correlation structure after filtering and feature engineering. |
| Raw-vs-standardized scale comparison | Shows why standardization is necessary before distance-based visualization, clustering, SVM, and linear models. |

The high-correlation audit is used with domain judgment. For example, the two semester approved-count variables are highly correlated, but they are kept because they describe different time points in academic progression. This is preferable to mechanically dropping all highly correlated academic variables.

Short interpretation for the report:

- The class distribution plot shows that `Enrolled` is the minority class, so accuracy alone is not sufficient and macro-averaged metrics are necessary later.
- The correlation heatmap confirms that most redundant numeric relationships have been reduced, while the remaining high-correlation academic progression signal is intentionally retained.
- The scale comparison shows that raw numeric variables have very different ranges, justifying standardization before t-SNE, clustering, SVM, and linear models.

## 9. Why Previous Schemes Are Not Used as the Main Plan

The previous preprocessing ideas are preserved for comparison, but they should not be used as the main report pipeline.

### Previous Scheme A: Treating All Numeric-Looking Columns as Continuous

Problem:

All 36 feature columns are stored as numbers, but many are actually category IDs. For example, `Course`, `Application mode`, and parent occupation/qualification codes are labels, not continuous quantities.

Why it is not suitable:

- Standardizing category IDs imposes fake numerical distance.
- A course code of 9991 is not "larger" than a course code of 33 in a meaningful way.
- This weakens interpretability and can mislead distance-based methods such as t-SNE and clustering.

### Previous Scheme B: Target Encoding as the Main Categorical Encoder

Problem:

Target encoding uses the label `Target` during feature construction.

Why it is not suitable as the main preprocessing plan:

- It can leak label information into t-SNE and clustering, which are supposed to be unsupervised analyses.
- The custom multiclass target encoder maps class labels to ordinal numbers, but the class order is not naturally continuous.
- The previous notebook text claimed internal cross-validation leakage prevention, but the custom encoder did not actually implement internal CV.
- It did not improve the final score enough to justify the added leakage and explanation risk.

Target encoding may still be discussed as an experimental supervised-only extension, but it should not be used for the shared preprocessing representation.

### Previous Scheme C: PowerTransformer as Main Numeric Scaling

Problem:

Yeo-Johnson transformation is statistically valid, but it did not improve the clean validation result compared with standard scaling.

Why it is not selected as the main plan:

- `StandardScaler` directly matches the project requirement.
- It is easier to explain clearly.
- It achieved slightly better cross-validation macro-F1 in the clean one-hot pipeline.
- The dataset contains many count and academic-performance features where interpretability is more important than distributional normalization.

PowerTransformer can be mentioned as a tested alternative, but not as the final main preprocessing choice.

## 10. Final Recommendation for Report Writing

In the report, the preprocessing section should emphasize:

1. The dataset has no missing values or duplicates, but the pipeline still includes imputers for robustness.
2. Integer-coded categories are handled semantically rather than blindly by dtype.
3. One-hot encoding is selected because it is leakage-free and appropriate for nominal variables.
4. StandardScaler is used for numeric features to meet the project requirement and stabilize model training.
5. Feature filtering reduces noise and redundancy.
6. Academic trajectory features are added because semester performance is strongly related to dropout risk.
7. More complex encoders were considered, but the final pipeline is chosen because it is safer, interpretable, and equally competitive in validation.
