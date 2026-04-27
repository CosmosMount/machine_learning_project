# Task 1-5 Final Quality Audit

Audit date: 2026-04-28

Scope: Tasks 1-5 only. Task 6 and final discussion are intentionally excluded from this audit.

## 1. Execution Status

The notebook code cells for Tasks 1-5 were executed sequentially:

- Cell 1: imports and global setup
- Cell 3: data loading and basic checks
- Cell 5: semantic feature classification
- Cell 7: feature filtering
- Cell 9: feature engineering
- Cell 11: preprocessing pipeline
- Cell 13: preprocessing diagnostics
- Cell 17: t-SNE
- Cell 19: t-SNE perplexity sensitivity
- Cell 21: clustering
- Cell 23: supervised training and testing
- Cell 25: model evaluation, tuning, and final model choice

Result: `TASK1_5_SEQUENTIAL_RUN_OK`

Key runtime outputs:

| Item | Result |
|---|---:|
| Raw dataset shape | 4424 rows, 37 columns |
| Final transformed feature matrix | `(4424, 172)` |
| t-SNE sensitivity settings | 3 perplexities: 10, 30, 50 |
| Clustering methods evaluated | 7 |
| Valid clustering settings | 43 |
| Task 4 best held-out test model | `GradientBoosting` |
| Task 4 best held-out test F1-macro | `0.7082` |
| Task 5 final selected model | `HistGradientBoosting_tuned` |
| Task 5 selection metric | Best cross-validated F1-macro |
| Task 5 selected-model CV F1-macro | `0.7176` |
| Task 5 selected-model test F1-macro | `0.7072` |
| Held-out test F1 winner, diagnostic only | `ExtraTrees_tuned` |
| Final-model permutation-importance rows | 33 raw input features |

## 2. PDF Requirement Alignment

### Task 1: Data Preprocessing

Requirement coverage:

| Requirement | Current implementation |
|---|---|
| Handle missing values | Dataset has 0 missing values; numeric and categorical imputers remain in the pipeline for robustness. |
| Handle non-numeric or categorical values | Integer-coded nominal categories are semantically identified and one-hot encoded. |
| Boolean handling | Binary indicators are passed through directly. |
| Standardization | Numeric/count features use `StandardScaler`. |
| Explanation and insight | Feature typing, filtering, feature engineering, class imbalance, and diagnostic plots are documented. |

High-score elements:

- Semantic feature classification rather than treating all numeric-looking columns as continuous.
- Near-zero-variance filtering.
- Correlation pruning with a domain-aware exception for semester approved counts.
- Feature engineering: `approval_rate_1st`, `approval_rate_2nd`, and `grade_trend`.
- Report-ready diagnostics:
  - `task1_class_distribution.png`
  - `task1_numeric_correlation_heatmap.png`
  - `task1_preprocessing_scale_effect.png`

Final status: A/A+ quality. The preprocessing is defensible, leakage-free, and aligned with the project requirement.

### Task 2: t-SNE Visualization

Requirement coverage:

| Requirement | Current implementation |
|---|---|
| Visualize high-dimensional data | t-SNE is applied after PCA reduction to 50 dimensions. |
| Use class labels for interpretation | Scatter plot is colored by `Target`; labels are not used to fit t-SNE. |
| Discuss patterns | Centroids, pairwise distances, overlap, and KL divergence are reported. |

High-score elements:

- PCA to 50 dimensions before t-SNE, retaining about `97.44%` variance.
- Main t-SNE plot plus sensitivity analysis at perplexity `10`, `30`, and `50`.
- KL divergence reporting.
- Explicit caveat that t-SNE is qualitative and most reliable for local neighborhood structure.
- Stable conclusion: the outcome classes overlap substantially, especially around `Enrolled`.

Final status: A+ quality.

### Task 3: Clustering

Requirement coverage:

| Requirement | Current implementation |
|---|---|
| At least two clustering algorithms | Seven clustering families are evaluated. |
| Multiple metrics | Silhouette, Calinski-Harabasz, Davies-Bouldin, ARI, NMI, homogeneity, completeness, and V-measure. |
| Visualization | Silhouette-by-k, top-method comparison, and k=3 t-SNE projections. |
| Justification of best result | Internal metrics and target-aligned external diagnostics are separated. |

High-score elements:

- Algorithms: KMeans, MiniBatchKMeans, Agglomerative Ward, BIRCH, Spectral Clustering, Gaussian Mixture, DBSCAN.
- `k = 2` to `8` scan for count-based methods.
- DBSCAN density-parameter scan.
- Clear distinction between:
  - unsupervised internal optimum: `KMeans_k=2`;
  - target-aligned k=3 diagnostic: `KMeans_k=3`.

Final interpretation:

- Internal metrics suggest the strongest natural structure is closer to `k=2`, similar to an `At-Risk` versus `Graduate` pattern.
- Forced `k=3` clustering only partially aligns with `Dropout`, `Enrolled`, and `Graduate`, which helps explain why the supervised multiclass task is not trivially separable.

Final status: A/A+ quality after interpretation repair.

### Task 4: Prediction, Training, and Testing

Requirement coverage:

| Requirement | Current implementation |
|---|---|
| Classification target | `Target` with `Dropout`, `Enrolled`, `Graduate`. |
| At least two simple model classes | Logistic Regression and Decision Tree. |
| Train/test split | 70/30 stratified split. |
| Train/test/all evaluation | Required simple baselines are reported on train, test, and all data. |
| Confusion matrices | Confusion matrices generated for required baselines. |
| Decision boundary visualization | PCA-2D decision-region visualization generated with explicit caveat. |

High-score elements:

- Expanded supervised model pool with nine models.
- 5-fold CV macro-F1 plus held-out test metrics.
- Overfitting diagnosis.
- PCA-2D caveat now explicitly states that the plot is only a visualization and the reported metrics use the full feature matrix.

Main result:

- Best Task 4 held-out test model: `GradientBoosting`.
- Test F1-macro: `0.7082`.

Final status: A/A+ quality.

### Task 5: Evaluation and Choice of Prediction Model

Requirement coverage:

| Requirement | Current implementation |
|---|---|
| Accuracy, precision, recall, F1 | Reported for tuned models and split-level metrics. |
| ROC/AUC | Macro OVR AUC and top tuned ROC curves are generated. |
| Improve models through validation | Nine model families are tuned with train-only GridSearchCV. |
| Discuss strengths/weaknesses | Class-level F1, train-test gaps, AUC, and permutation importance are reported. |
| Final model choice | Final model is selected by validation F1-macro, not by test-set search. |

Critical method repair completed:

- Previous issue: final model selection used held-out test F1, causing selection leakage.
- Current rule: choose the model with the highest `Best_CV_F1_macro`.
- Held-out test metrics are reported only after the validation-based selection.

Final selected model:

| Metric | Value |
|---|---:|
| Model | `HistGradientBoosting_tuned` |
| Best CV F1-macro | `0.7176` |
| Test F1-macro | `0.7072` |
| Test AUC-macro OVR | `0.8894` |
| Train-test F1 gap | `0.1096` |

Important diagnostic:

- `ExtraTrees_tuned` has the highest held-out test F1-macro: `0.7139`.
- It is not selected as the final model because using test F1 for selection would bias the final evaluation.
- It remains useful as a post-selection diagnostic.

Final-model interpretability:

- Added model-agnostic permutation importance for `HistGradientBoosting_tuned`.
- Top drivers include:
  - `Tuition fees up to date`
  - `approval_rate_2nd`
  - `Curricular units 2nd sem (approved)`
  - `approval_rate_1st`

Final status: A/A+ quality after selection-leakage repair.

## 3. Generated Artifacts

Core documentation:

- `docs/preprocessing_final.md`
- `docs/tsne_analysis.md`
- `docs/clustering_analysis.md`
- `docs/task4_prediction_training.md`
- `docs/task5_evaluation_model_choice.md`
- `docs/task1_5_final_audit.md`

Core Task 1-5 figures:

- `docs/figures/task1_class_distribution.png`
- `docs/figures/task1_numeric_correlation_heatmap.png`
- `docs/figures/task1_preprocessing_scale_effect.png`
- `docs/figures/tsne_projection.png`
- `docs/figures/task2_tsne_perplexity_sensitivity.png`
- `docs/figures/clustering_expanded_silhouette_by_k.png`
- `docs/figures/clustering_expanded_top_silhouette.png`
- `docs/figures/clustering_expanded_k3_on_tsne.png`
- `docs/figures/task4_confusion_LogisticRegression_balanced.png`
- `docs/figures/task4_confusion_DecisionTree_default.png`
- `docs/figures/task4_expanded_supervised_comparison.png`
- `docs/figures/task4_expanded_test_f1.png`
- `docs/figures/task4_decision_boundaries_pca2.png`
- `docs/figures/task5_tuned_f1_auc_comparison.png`
- `docs/figures/task5_top_tuned_roc_curves.png`
- `docs/figures/task5_class_f1_heatmap.png`
- `docs/figures/task5_final_model_confusion_test.png`
- `docs/figures/task5_final_model_permutation_importance.png`

Core Task 4-5 tables:

- `docs/tables/supervised_expanded_model_comparison.csv`
- `docs/tables/supervised_expanded_top_reports.txt`
- `docs/tables/task5_tuned_model_summary.csv`
- `docs/tables/task5_tuned_split_metrics.csv`
- `docs/tables/task5_tuned_vs_untuned.csv`
- `docs/tables/task5_class_f1_by_model.csv`
- `docs/tables/task5_best_params.csv`
- `docs/tables/task5_tuned_classification_reports.txt`
- `docs/tables/task5_final_model_permutation_importance.csv`

## 4. Reproducibility

`requirements.txt` now pins exact package versions:

```text
numpy==2.1.3
pandas==2.2.3
matplotlib==3.10.0
seaborn==0.13.2
scikit-learn==1.6.1
jupyter==1.1.1
notebook==7.3.2
pypdf==6.10.2
```

Environment-specific note:

- `LOKY_MAX_CPU_COUNT` is set in the notebook to reduce Windows joblib CPU-detection noise.
- `n_jobs=1` is used in validation-heavy sections to avoid platform-specific multiprocessing warnings.
- `MLPClassifier(early_stopping=True)` is avoided because this sklearn build can fail with string labels during early-stopping validation scoring.

## 5. Final Risk Register

| Risk | Current status |
|---|---|
| Test-set leakage in final model choice | Fixed. Final selection uses CV F1-macro. |
| Categorical variables treated as continuous | Fixed by semantic typing and one-hot encoding. |
| t-SNE over-interpretation | Controlled with perplexity sensitivity and caveat. |
| Clustering k=2 vs k=3 confusion | Fixed in notebook and clustering report. |
| PCA-2D decision-boundary misinterpretation | Fixed with explicit caveat. |
| Class imbalance hiding weak `Enrolled` performance | Controlled with macro metrics and class-level F1. |
| Model interpretability | Strengthened with permutation importance. |
| Fresh-environment reproducibility | Strengthened with pinned requirements. |

Remaining out-of-scope issues:

- Task 6 and final discussion still need separate review.
- The repository root contains temporary audit files (`_audit_*.py/.txt`, `_task5_*.txt`) that are not part of the final submission package.

## 6. Bottom Line

Tasks 1-5 are now methodologically coherent, reproducible, and report-ready. The main scoring-critical issue, test-set-based model selection in Task 5, has been corrected. The current final model choice is `HistGradientBoosting_tuned` by cross-validated F1-macro, with held-out test metrics reported only as final evidence.
