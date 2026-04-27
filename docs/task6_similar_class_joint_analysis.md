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
| HistGradientBoosting_tuned_collapsed_to_At-Risk | final 3-class model collapsed after prediction |  | 0.8409 | 0.8215 | 0.8722 | 0.8461 | 0.9322 |
| Group_LogisticRegression_tuned | linear At-Risk detector | 0.8506 | 0.8523 | 0.8705 | 0.8286 | 0.8490 | 0.9225 |
| Group_HistGradientBoosting_tuned | regularized boosting At-Risk detector | 0.8506 | 0.8501 | 0.8652 | 0.8301 | 0.8473 | 0.9298 |
| Group_RandomForest_tuned | bagged tree At-Risk detector | 0.8473 | 0.8531 | 0.8707 | 0.8301 | 0.8499 | 0.9341 |

Best dedicated At-Risk model by validation F1-macro: `Group_LogisticRegression_tuned`.

- Dedicated At-Risk recall: `0.8286`
- Collapsed Task 5 At-Risk recall: `0.8722`
- Recall delta: `-0.0436`

This is a useful negative result: the dedicated binary model improves the overall binary F1/precision trade-off, but it does not improve At-Risk recall over the collapsed Task 5 model. If the intervention goal prioritizes catching every At-Risk student, the collapsed Task 5 signal remains competitive; if the goal prioritizes fewer false alarms, the dedicated binary model is cleaner.

## Dropout vs Enrolled Subgroup Results

| Model | Family | Best_CV_F1_macro | Test_F1_macro | Enrolled_Precision | Enrolled_Recall | Enrolled_F1 | Test_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Subgroup_HistGradientBoosting_tuned | regularized boosting Dropout-vs-Enrolled classifier | 0.7722 | 0.7530 | 0.6156 | 0.8613 | 0.7180 | 0.8396 |
| Subgroup_RandomForest_tuned | bagged tree Dropout-vs-Enrolled classifier | 0.7710 | 0.7642 | 0.6467 | 0.8109 | 0.7188 | 0.8370 |
| Subgroup_LogisticRegression_tuned | linear Dropout-vs-Enrolled classifier | 0.7685 | 0.7529 | 0.6295 | 0.8109 | 0.7083 | 0.8426 |
| Subgroup_RBF_SVM_tuned | kernel Dropout-vs-Enrolled classifier | 0.7685 | 0.7362 | 0.6211 | 0.7563 | 0.6818 | 0.8273 |

Best subgroup model by validation F1-macro: `Subgroup_HistGradientBoosting_tuned`.

## t-SNE Similarity Evidence

| Class | Dropout | Enrolled | Graduate |
| --- | --- | --- | --- |
| Dropout | 0.0000 | 27.8978 | 39.1543 |
| Enrolled | 27.8978 | 0.0000 | 11.8804 |
| Graduate | 39.1543 | 11.8804 | 0.0000 |

The centroid-distance evidence shows why this extension must be interpreted carefully: `Enrolled` is geometrically closer to `Graduate` than to `Dropout` on the t-SNE plane. Therefore, `At-Risk` is a practical intervention grouping rather than a naturally separated cluster. This helps explain why the dedicated At-Risk model improves precision/macro-F1 trade-offs but does not automatically improve At-Risk recall.

## Interpretation

The dedicated At-Risk model should be discussed as an intervention-oriented model, not as a replacement for the final Task 5 three-class classifier. Its purpose is to answer a different question: whether a student is likely to require attention, regardless of whether the final label is `Dropout` or `Enrolled`. The result is a trade-off rather than a simple win: dedicated binary modeling raises At-Risk precision and macro-F1, while the collapsed Task 5 model keeps higher At-Risk recall. The subgroup model then shows how difficult it remains to separate the two At-Risk classes. This supports the broader project conclusion that moderate macro-F1 is not simply a modeling failure; it reflects genuine overlap in the student outcome structure.

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
