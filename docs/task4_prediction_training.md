# Task 4: Expanded Prediction, Training, and Testing

## 1. Task Objective

The project requires training supervised learning models and assessing their performance on:

- the training set,
- the testing set,
- the entire dataset.

The classification target is:

- `Target`: `Dropout`, `Enrolled`, `Graduate`.

To satisfy the requirement and support a high-score report, we use both:

1. Required simple baselines: Logistic Regression and Decision Tree.
2. Expanded advanced model pool: Random Forest, ExtraTrees, Gradient Boosting, HistGradientBoosting, Linear SVM, RBF SVM, and MLP.

## 2. Train-Test Split

We use a 70/30 stratified train-test split:

| Split | Rows |
|---|---:|
| Train | 3096 |
| Test | 1328 |
| All | 4424 |

Class distribution:

| Class | Train | Test | All |
|---|---:|---:|---:|
| Dropout | 994 | 427 | 1421 |
| Enrolled | 556 | 238 | 794 |
| Graduate | 1546 | 663 | 2209 |

Stratification keeps class proportions stable and is especially important because `Enrolled` is the minority class.

## 3. Model Pool

| Model | Role |
|---|---|
| LogisticRegression_balanced | Required simple baseline; stable and interpretable. |
| DecisionTree_default | Required simple nonlinear baseline; useful for overfitting diagnosis. |
| RandomForest_balanced | Ensemble tree model; reduces variance compared with a single tree. |
| ExtraTrees_balanced | More randomized tree ensemble; useful robustness comparison. |
| GradientBoosting | Sequential boosting method; often strong on tabular data. |
| HistGradientBoosting_balanced | Faster modern gradient boosting with class weighting. |
| LinearSVM_balanced | Large-margin linear classifier for high-dimensional one-hot features. |
| RBF_SVM_balanced | Nonlinear kernel SVM; tests nonlinear decision boundaries. |
| MLP | Neural network baseline for open-ended comparison. |

Every model is trained in a `Pipeline`:

```text
preprocessing -> model
```

This prevents test-set leakage because preprocessing is fitted within each training fold or training split.

## 4. Expanded Model Comparison

All models are evaluated using:

- 5-fold cross-validation macro-F1,
- held-out test accuracy,
- held-out test precision/recall/F1 macro.

| Model | CV F1 Macro Mean | CV F1 Macro Std | Test Accuracy | Test Precision Macro | Test Recall Macro | Test F1 Macro |
|---|---:|---:|---:|---:|---:|---:|
| GradientBoosting | 0.7010 | 0.0081 | 0.7764 | 0.7213 | 0.6990 | 0.7067 |
| RBF_SVM_balanced | 0.7127 | 0.0097 | 0.7485 | 0.7067 | 0.7128 | 0.7054 |
| LogisticRegression_balanced | 0.7082 | 0.0118 | 0.7364 | 0.7065 | 0.7120 | 0.7000 |
| LinearSVM_balanced | 0.7029 | 0.0109 | 0.7553 | 0.7021 | 0.6931 | 0.6963 |
| ExtraTrees_balanced | 0.6961 | 0.0074 | 0.7733 | 0.7237 | 0.6843 | 0.6953 |
| RandomForest_balanced | 0.6831 | 0.0075 | 0.7741 | 0.7260 | 0.6805 | 0.6909 |
| HistGradientBoosting_balanced | 0.7134 | 0.0084 | 0.7417 | 0.6880 | 0.6840 | 0.6853 |
| MLP | 0.6524 | 0.0128 | 0.7282 | 0.6708 | 0.6709 | 0.6706 |
| DecisionTree_default | 0.6272 | 0.0119 | 0.6920 | 0.6269 | 0.6293 | 0.6280 |

Best held-out test macro-F1:

- `GradientBoosting`, test F1-macro = `0.7067`.

Best CV macro-F1:

- `HistGradientBoosting_balanced`, CV F1-macro = `0.7134`.

Most balanced interpretation:

- `RBF_SVM_balanced` is also strong, with CV F1-macro `0.7127` and test F1-macro `0.7054`.

## 5. Required Baseline Train/Test/All Evaluation

The project explicitly asks for evaluation on train, test, and all data. We report this for the required simple baselines.

| Model | Split | Accuracy | Precision Macro | Recall Macro | F1 Macro |
|---|---|---:|---:|---:|---:|
| LogisticRegression_balanced | Train | 0.7875 | 0.7532 | 0.7645 | 0.7532 |
| LogisticRegression_balanced | Test | 0.7349 | 0.7051 | 0.7107 | 0.6985 |
| LogisticRegression_balanced | All | 0.7717 | 0.7382 | 0.7483 | 0.7365 |
| DecisionTree_default | Train | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| DecisionTree_default | Test | 0.6920 | 0.6269 | 0.6293 | 0.6280 |
| DecisionTree_default | All | 0.9075 | 0.8869 | 0.8888 | 0.8878 |

## 6. Overfitting Diagnosis

| Model | Train F1 Macro | Test F1 Macro | Train-Test Gap |
|---|---:|---:|---:|
| LogisticRegression_balanced | 0.7532 | 0.6985 | 0.0546 |
| DecisionTree_default | 1.0000 | 0.6280 | 0.3720 |

Interpretation:

- Logistic Regression generalizes reasonably well.
- The default Decision Tree memorizes the training set and overfits heavily.
- This motivates validation-based pruning/tuning in Task 5.

## 7. Why Adding More Models Helps

Adding more supervised models improves the project because:

- It goes beyond the minimum requirement of two simple models.
- It compares linear, tree-based, boosting, kernel, and neural network approaches.
- It provides evidence that the final model choice is not arbitrary.
- It shows that the team understands model-family tradeoffs.

However, the results also show that more complex models are not automatically better. MLP underperforms the top tabular models, and the default Decision Tree overfits severely.

## 8. Visualizations and Tables

Generated figures:

- `docs/figures/task4_confusion_LogisticRegression_balanced.png`
- `docs/figures/task4_confusion_DecisionTree_default.png`
- `docs/figures/task4_expanded_supervised_comparison.png`
- `docs/figures/task4_expanded_test_f1.png`
- `docs/figures/task4_decision_boundaries_pca2.png`

Decision-boundary caveat:

- The decision-boundary plot is fitted only on a 2-D PCA projection for visualization.
- The PCA-2D projection retains `35.5%` of transformed-feature variance.
- The reported train/test/all metrics are computed with the full preprocessed feature space, not with this 2-D visualization model.

Generated tables:

- `docs/tables/supervised_expanded_model_comparison.csv`
- `docs/tables/supervised_expanded_top_reports.txt`

## 9. Report-Ready Summary

We first trained the two required simple model classes, Logistic Regression and Decision Tree, using a 70/30 stratified split and leakage-free pipelines. Logistic Regression generalized well with test macro-F1 `0.7000`, while the default Decision Tree overfit heavily, achieving training macro-F1 `1.0000` but test macro-F1 only `0.6280`. To exceed the baseline requirement, we then compared seven additional advanced models: Random Forest, ExtraTrees, Gradient Boosting, HistGradientBoosting, Linear SVM, RBF SVM, and MLP. Gradient Boosting achieved the best held-out test macro-F1 (`0.7067`), while RBF SVM and HistGradientBoosting showed strong cross-validation performance. The PCA-2D boundary plot is used only to visualize simplified decision regions; final metrics use the full feature matrix. These results demonstrate that boosting and kernel methods improve over simple baselines, but class overlap and the minority `Enrolled` class remain the central challenge.
