# Task 6: Similar-Class Joint Analysis

## Purpose

Task 6 is an open-ended extension that addresses a consistent finding from earlier tasks: `Enrolled` overlaps with both `Dropout` and `Graduate`, making the three-class problem intrinsically difficult. Instead of only reporting another classifier, this section reframes the problem into a practical two-stage risk analysis.

## Method

1. Collapse `Dropout` and `Enrolled` into `At-Risk`, then compare `At-Risk` vs `Graduate`.
2. Within the At-Risk subset, train a second classifier to distinguish `Dropout` from `Enrolled`.
3. Compare the dedicated At-Risk binary model against the final Task 5 three-class model collapsed after prediction.
4. Use leakage-free pipelines with `clone(preprocessor)`, train-only 5-fold `GridSearchCV`, held-out test metrics, ROC/AUC, confusion matrices, and t-SNE centroid-distance evidence.

## At-Risk Detection Results

| Model | Family | Best_CV_F1_macro | Test_F1_macro | At-Risk_Precision | At-Risk_Recall | At-Risk_F1 | Test_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ExtraTrees_tuned_collapsed_to_At-Risk | final 3-class model collapsed after prediction |  | 0.8381 | 0.8299 | 0.8511 | 0.8404 | 0.9295 |
| Group_LogisticRegression_tuned | linear At-Risk detector | 0.8581 | 0.8319 | 0.8542 | 0.8015 | 0.8270 | 0.9099 |
| Group_HistGradientBoosting_tuned | regularized boosting At-Risk detector | 0.8612 | 0.8386 | 0.8667 | 0.8015 | 0.8328 | 0.9142 |
| Group_RandomForest_tuned | bagged tree At-Risk detector | 0.8505 | 0.8287 | 0.8650 | 0.7805 | 0.8206 | 0.9079 |

Best dedicated At-Risk model by validation F1-macro: `Group_HistGradientBoosting_tuned`.

- Dedicated At-Risk recall: `0.8015`
- Collapsed Task 5 At-Risk recall: `0.8511`
- Recall delta: `-0.0496`

This is a useful negative result: the dedicated binary model improves the overall binary F1/precision trade-off, but it does not necessarily improve At-Risk recall over the collapsed Task 5 model. If the intervention goal prioritizes catching every At-Risk student, the collapsed Task 5 signal remains competitive; if the goal prioritizes fewer false alarms, the dedicated binary model is cleaner.

## Dropout vs Enrolled Subgroup Results

| Model | Family | Best_CV_F1_macro | Test_F1_macro | Enrolled_Precision | Enrolled_Recall | Enrolled_F1 | Test_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Subgroup_HistGradientBoosting_tuned | regularized boosting Dropout-vs-Enrolled classifier | 0.7727 | 0.7654 | 0.6344 | 0.8529 | 0.7276 | 0.8366 |
| Subgroup_RandomForest_tuned | bagged tree Dropout-vs-Enrolled classifier | 0.7726 | 0.7610 | 0.6355 | 0.8277 | 0.7190 | 0.8293 |
| Subgroup_RBF_SVM_tuned | kernel Dropout-vs-Enrolled classifier | 0.7696 | 0.7403 | 0.6216 | 0.7731 | 0.6891 | 0.8308 |
| Subgroup_LogisticRegression_tuned | linear Dropout-vs-Enrolled classifier | 0.7685 | 0.7529 | 0.6287 | 0.8109 | 0.7083 | 0.8426 |

Best subgroup model by validation F1-macro: `Subgroup_HistGradientBoosting_tuned`.

## t-SNE Similarity Evidence

| Class | Dropout | Enrolled | Graduate |
| --- | --- | --- | --- |
| Dropout | 0.0000 | 28.0163 | 39.1288 |
| Enrolled | 28.0163 | 0.0000 | 11.7984 |
| Graduate | 39.1288 | 11.7984 | 0.0000 |

The centroid-distance evidence shows why this extension must be interpreted carefully: `Enrolled` is geometrically closer to `Graduate` than to `Dropout` on the t-SNE plane. Therefore, `At-Risk` is a practical intervention grouping rather than a naturally separated cluster.

## Interpretation

The dedicated At-Risk model should be discussed as an intervention-oriented model, not as a replacement for the final Task 5 three-class classifier. Its purpose is to answer a different question: whether a student is likely to require attention, regardless of whether the final label is `Dropout` or `Enrolled`. The result is a trade-off rather than a simple win. The subgroup model then shows how difficult it remains to separate the two At-Risk classes. This supports the broader project conclusion that moderate macro-F1 is not simply a modeling failure; it reflects genuine overlap in the student outcome structure.

## Saved Outputs

Tables:

- `docs/tables/task6_group_model_comparison.csv`
- `docs/tables/task6_group_best_params.csv`
- `docs/tables/task6_subgroup_model_comparison.csv`
- `docs/tables/task6_subgroup_best_params.csv`
- `docs/tables/task6_task5_risk_class_recall.csv`
- `docs/tables/task6_tsne_centroid_distances.csv`
- `docs/tables/task6_group_classification_reports.txt`
- `docs/tables/task6_subgroup_classification_reports.txt`

Figures:

- `docs/figures/task6_group_confusion_matrix.png`
- `docs/figures/task6_subgroup_confusion_matrix.png`
- `docs/figures/task6_atrisk_detection_comparison.png`
- `docs/figures/task6_group_roc_comparison.png`
- `docs/figures/task6_subgroup_roc_curve.png`
- `docs/figures/task6_tsne_centroid_distance_heatmap.png`
