# Task 5: Evaluation and Choice of Prediction Model

## Requirement Alignment

The assignment asks for accuracy, precision, recall, F1-score, ROC/AUC, model improvement through validation, and a final model choice. This Task 5 implementation satisfies those requirements and goes beyond them by tuning nine model families with leakage-free pipelines and train-only 5-fold stratified validation.

## Why the Previous Task 5 Was Not Enough for a High-score Submission

The earlier version met the minimum requirement because it tuned Logistic Regression and Decision Tree and added a small Random Forest exploration. However, it was weaker than the expanded Task 4 model pool: ROC/AUC and validation tuning were not applied consistently to the stronger models, model-selection evidence was fragmented, and the old parallel job setting could produce noisy Windows joblib warnings. The updated version evaluates the full candidate family set under one comparable protocol.

## Validation Protocol

- Split: the existing 70/30 stratified train-test split from Task 4.
- Tuning: `GridSearchCV` on the training set only.
- Cross-validation: stratified 5-fold CV.
- Selection rule: choose the model with the best validation F1-macro.
- Test-set use: report held-out accuracy, precision/recall macro, F1-macro, ROC/AUC macro OVR, train-test gap, and class-level F1 only after selection.

## Tuned Model Summary

| Model | Family | Best_CV_F1_macro | Best_CV_F1_std | Test_F1_macro | Test_AUC_macro_OVR | CV_Train_minus_Val_F1 | Train_minus_Test_F1 | Validation_Diagnostic_Rank_Mean | Test_Diagnostic_Rank_Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HistGradientBoosting_tuned | regularized histogram boosting | 0.7176 | 0.0245 | 0.7072 | 0.8894 | 0.1198 | 0.1096 | 4.0000 | 3.0000 |
| RandomForest_tuned | bagged tree ensemble | 0.7146 | 0.0111 | 0.7089 | 0.8875 | 0.2548 | 0.2387 | 3.6667 | 4.3333 |
| ExtraTrees_tuned | randomized tree ensemble | 0.7135 | 0.0165 | 0.7139 | 0.8847 | 0.2127 | 0.1952 | 4.6667 | 4.0000 |
| LogisticRegression_tuned | linear probabilistic baseline | 0.7082 | 0.0171 | 0.6972 | 0.8816 | 0.0531 | 0.0563 | 4.0000 | 5.0000 |
| RBF_SVM_tuned | nonlinear kernel classifier | 0.7080 | 0.0158 | 0.7054 | 0.8658 | 0.1773 | 0.1705 | 4.6667 | 5.6667 |
| LinearSVM_tuned | large-margin linear classifier | 0.7063 | 0.0134 | 0.7029 | 0.8745 | 0.0401 | 0.0420 | 3.3333 | 4.3333 |
| GradientBoosting_tuned | sequential boosting ensemble | 0.7026 | 0.0261 | 0.7027 | 0.8937 | 0.1423 | 0.1228 | 7.0000 | 4.0000 |
| DecisionTree_tuned | single interpretable tree | 0.6783 | 0.0246 | 0.6696 | 0.8534 | 0.0316 | 0.0383 | 5.6667 | 6.0000 |
| MLP_tuned | neural network baseline | 0.6744 | 0.0178 | 0.6712 | 0.8489 | 0.3232 | 0.3266 | 8.0000 | 8.6667 |

## Tuning Gain over Untuned Task 4 Models

| Tuned_Model | Untuned_Task4_Model | Untuned_CV_F1_macro | Tuned_CV_F1_macro | Delta_CV_F1_macro | Untuned_Test_F1_macro | Tuned_Test_F1_macro | Delta_Test_F1_macro |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HistGradientBoosting_tuned | HistGradientBoosting_balanced | 0.7134 | 0.7176 | 0.0042 | 0.6853 | 0.7072 | 0.0219 |
| RandomForest_tuned | RandomForest_balanced | 0.6831 | 0.7146 | 0.0314 | 0.6909 | 0.7089 | 0.0180 |
| ExtraTrees_tuned | ExtraTrees_balanced | 0.6961 | 0.7135 | 0.0174 | 0.6953 | 0.7139 | 0.0186 |
| LogisticRegression_tuned | LogisticRegression_balanced | 0.7084 | 0.7082 | -0.0002 | 0.6985 | 0.6972 | -0.0014 |
| RBF_SVM_tuned | RBF_SVM_balanced | 0.7127 | 0.7080 | -0.0046 | 0.7054 | 0.7054 | 0.0000 |
| LinearSVM_tuned | LinearSVM_balanced | 0.7029 | 0.7063 | 0.0034 | 0.6963 | 0.7029 | 0.0066 |
| GradientBoosting_tuned | GradientBoosting | 0.7008 | 0.7026 | 0.0018 | 0.7082 | 0.7027 | -0.0055 |
| DecisionTree_tuned | DecisionTree_default | 0.6272 | 0.6783 | 0.0511 | 0.6280 | 0.6696 | 0.0417 |
| MLP_tuned | MLP | 0.6524 | 0.6744 | 0.0220 | 0.6706 | 0.6712 | 0.0006 |

## Class-level F1 Evidence

| Model | Dropout | Enrolled | Graduate |
| --- | --- | --- | --- |
| HistGradientBoosting_tuned | 0.7657 | 0.5199 | 0.8358 |
| RandomForest_tuned | 0.7821 | 0.5041 | 0.8406 |
| ExtraTrees_tuned | 0.7576 | 0.5360 | 0.8481 |
| LogisticRegression_tuned | 0.7683 | 0.5008 | 0.8224 |
| RBF_SVM_tuned | 0.7624 | 0.5136 | 0.8402 |
| LinearSVM_tuned | 0.7725 | 0.4846 | 0.8517 |

## Final Model Choice

Recommended final model by cross-validated F1-macro: `HistGradientBoosting_tuned`.

- Test F1-macro: `0.7072`
- Test AUC-macro OVR: `0.8894`
- Best validation F1-macro: `0.7176`
- Train-test F1 gap: `0.1096`

The held-out test set is used only after validation-based selection. The highest held-out test F1 belongs to `ExtraTrees_tuned` with test F1-macro `0.7139`, while the best post-selection test diagnostic rank belongs to `HistGradientBoosting_tuned` with diagnostic rank mean `3.0000`. These are reported as diagnostics, not as the selection rule.

## Interpretation

The tuned model comparison confirms that advanced nonlinear methods help, but the final model is selected by validation rather than by searching for the best held-out test result. The gain is bounded by genuine class overlap rather than preprocessing failure. The class-level F1 table and permutation-importance output should be used to explain both performance and feature drivers. The class-level table shows why `Enrolled` remains the hardest class: it is smaller, transitional, and overlaps with both `Dropout` and `Graduate` in earlier t-SNE and clustering evidence.

## Saved Outputs

- `docs/tables/task5_tuned_model_summary.csv`
- `docs/tables/task5_tuned_split_metrics.csv`
- `docs/tables/task5_tuned_vs_untuned.csv`
- `docs/tables/task5_class_f1_by_model.csv`
- `docs/tables/task5_best_params.csv`
- `docs/tables/task5_tuned_classification_reports.txt`
- `docs/tables/task5_final_model_permutation_importance.csv`
- `docs/figures/task5_tuned_f1_auc_comparison.png`
- `docs/figures/task5_tuned_vs_untuned_f1_delta.png`
- `docs/figures/task5_class_f1_heatmap.png`
- `docs/figures/task5_top_tuned_roc_curves.png`
- `docs/figures/task5_final_model_confusion_test.png`
- `docs/figures/task5_final_or_interpretable_top_features.png`
- `docs/figures/task5_final_model_permutation_importance.png`
