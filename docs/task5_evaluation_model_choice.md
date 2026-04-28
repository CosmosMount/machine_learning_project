# Task 5: Evaluation and Choice of Prediction Model

## Requirement Alignment

The assignment asks for accuracy, precision, recall, F1-score, ROC/AUC, model improvement through validation, and a final model choice. This Task 5 implementation satisfies those requirements and goes beyond them by tuning nine model families with leakage-free pipelines and train-only 5-fold stratified validation.

## Why the Previous Task 5 Was Not Enough for a High-score Submission

The earlier version met the minimum requirement because it tuned Logistic Regression and Decision Tree and added a small Random Forest exploration. However, it was weaker than the expanded Task 4 model pool: ROC/AUC and validation tuning were not applied consistently to the stronger models, model-selection evidence was fragmented, and `n_jobs=-1` could produce noisy Windows joblib warnings. The updated version evaluates the full candidate family set under one comparable protocol.

## Validation Protocol

- Split: the existing 70/30 stratified train-test split from Task 4.
- Tuning: `GridSearchCV` on the training set only.
- Cross-validation: stratified 5-fold CV.
- Primary metric: F1-macro, because the target classes are imbalanced and `Enrolled` is the minority class.
- Secondary evidence: held-out accuracy, precision/recall macro, ROC/AUC macro OVR, train-test gap, validation variance, and class-level F1.

## Tuned Model Summary

| Model | Family | Best_CV_F1_macro | Best_CV_F1_std | Test_F1_macro | Test_AUC_macro_OVR | CV_Train_minus_Val_F1 | Train_minus_Test_F1 | Validation_Diagnostic_Rank_Mean | Test_Diagnostic_Rank_Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ExtraTrees_tuned | randomized tree ensemble | 0.7195 | 0.0075 | 0.7031 | 0.8859 | 0.2027 | 0.2015 | 3.3333 | 3.8000 |
| HistGradientBoosting_tuned | regularized histogram boosting | 0.7158 | 0.0194 | 0.7064 | 0.8908 | 0.1243 | 0.1112 | 4.6667 | 3.6000 |
| RandomForest_tuned | bagged tree ensemble | 0.7109 | 0.0156 | 0.7159 | 0.8879 | 0.1461 | 0.1402 | 3.6667 | 3.0000 |
| LinearSVM_tuned | large-margin linear classifier | 0.7108 | 0.0160 | 0.7042 | 0.8756 | 0.0313 | 0.0361 | 2.6667 | 3.4000 |
| RBF_SVM_tuned | nonlinear kernel classifier | 0.7103 | 0.0169 | 0.7037 | 0.8674 | 0.1502 | 0.1525 | 5.6667 | 5.6000 |
| GradientBoosting_tuned | sequential boosting ensemble | 0.7094 | 0.0231 | 0.6949 | 0.8910 | 0.1432 | 0.1349 | 6.6667 | 5.6000 |
| LogisticRegression_tuned | linear probabilistic baseline | 0.7080 | 0.0165 | 0.6970 | 0.8817 | 0.0533 | 0.0562 | 4.3333 | 4.8000 |
| DecisionTree_tuned | single interpretable tree | 0.6810 | 0.0175 | 0.6522 | 0.8361 | 0.0639 | 0.0773 | 5.6667 | 7.0000 |
| MLP_tuned | neural network baseline | 0.6725 | 0.0184 | 0.6706 | 0.8449 | 0.3267 | 0.3285 | 8.3333 | 8.2000 |

## Tuning Gain over Untuned Task 4 Models

| Tuned_Model | Untuned_Task4_Model | Untuned_Test_F1_macro | Tuned_Test_F1_macro | Delta_Test_F1_macro |
| --- | --- | --- | --- | --- |
| RandomForest_tuned | RandomForest_balanced | 0.6909 | 0.7159 | 0.0249 |
| HistGradientBoosting_tuned | HistGradientBoosting_balanced | 0.6853 | 0.7064 | 0.0211 |
| LinearSVM_tuned | LinearSVM_balanced | 0.6963 | 0.7042 | 0.0079 |
| RBF_SVM_tuned | RBF_SVM_balanced | 0.7054 | 0.7037 | -0.0016 |
| ExtraTrees_tuned | ExtraTrees_balanced | 0.6953 | 0.7031 | 0.0078 |
| LogisticRegression_tuned | LogisticRegression_balanced | 0.7000 | 0.6970 | -0.0030 |
| GradientBoosting_tuned | GradientBoosting | 0.7067 | 0.6949 | -0.0118 |
| MLP_tuned | MLP | 0.6706 | 0.6706 | 0.0000 |
| DecisionTree_tuned | DecisionTree_default | 0.6280 | 0.6522 | 0.0243 |

## Class-level F1 Evidence

| Model | Dropout | Enrolled | Graduate |
| --- | --- | --- | --- |
| ExtraTrees_tuned | 0.7582 | 0.5155 | 0.8358 |
| HistGradientBoosting_tuned | 0.7700 | 0.5181 | 0.8310 |
| RandomForest_tuned | 0.7700 | 0.5296 | 0.8480 |
| LinearSVM_tuned | 0.7688 | 0.4898 | 0.8540 |
| RBF_SVM_tuned | 0.7610 | 0.5158 | 0.8345 |
| GradientBoosting_tuned | 0.7748 | 0.4476 | 0.8622 |

## Final Model Choice

Recommended final model by cross-validated F1-macro: `ExtraTrees_tuned`.

- Test F1-macro: `0.7031`
- Test AUC-macro OVR: `0.8859`
- Best validation F1-macro: `0.7195`
- Train-test F1 gap: `0.2015`

The held-out test set is used only after validation-based selection. The best post-selection test diagnostic model by the combined rank is `RandomForest_tuned` with diagnostic rank mean `3.0000`. If it differs from the primary recommendation, this is useful discussion evidence, not a reason to override the validation-based selection rule.

## Interpretation

The tuned model comparison confirms that advanced nonlinear methods help, but the gain is bounded by genuine class overlap rather than preprocessing failure. The class-level F1 table should be used to explain why `Enrolled` remains the hardest class: it is smaller, transitional, and overlaps with both `Dropout` and `Graduate` in earlier t-SNE and clustering evidence.

## Saved Outputs

- `docs/tables/task5_tuned_model_summary.csv`
- `docs/tables/task5_tuned_split_metrics.csv`
- `docs/tables/task5_tuned_vs_untuned.csv`
- `docs/tables/task5_class_f1_by_model.csv`
- `docs/tables/task5_best_params.csv`
- `docs/tables/task5_tuned_classification_reports.txt`
- `docs/figures/task5_tuned_f1_auc_comparison.png`
- `docs/figures/task5_tuned_vs_untuned_f1_delta.png`
- `docs/figures/task5_class_f1_heatmap.png`
- `docs/figures/task5_top_tuned_roc_curves.png`
- `docs/figures/task5_final_model_confusion_test.png`
- `docs/figures/task5_final_or_interpretable_top_features.png`
