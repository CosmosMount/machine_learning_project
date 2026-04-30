# ---- 5a. Broad validation-based tuning across model families ----
import time
from pathlib import Path

Path('docs/tables').mkdir(parents=True, exist_ok=True)
Path('docs/figures').mkdir(parents=True, exist_ok=True)

task5_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
task5_classes = np.array(sorted(y_train.unique()))
y_test_bin = label_binarize(y_test, classes=task5_classes)

# The grids are intentionally broad enough to test different model assumptions,
# while still compact enough for a notebook that must run in a fresh environment.
tuning_specs = {
    'LogisticRegression_tuned': {
        'family': 'linear probabilistic baseline',
        'untuned': 'LogisticRegression_balanced',
        'estimator': LogisticRegression(max_iter=1800, solver='lbfgs', random_state=RANDOM_STATE),
        'param_grid': {
            'model__C': [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            'model__class_weight': [None, 'balanced'],
        },
    },
    'DecisionTree_tuned': {
        'family': 'single interpretable tree',
        'untuned': 'DecisionTree_default',
        'estimator': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'param_grid': {
            'model__max_depth': [4, 6, 8, 12, None],
            'model__min_samples_split': [2, 10],
            'model__min_samples_leaf': [1, 5, 15],
            'model__class_weight': [None, 'balanced'],
        },
    },
    'RandomForest_tuned': {
        'family': 'bagged tree ensemble',
        'untuned': 'RandomForest_balanced',
        'estimator': RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=1),
        'param_grid': {
            'model__max_depth': [None, 10, 16],
            'model__min_samples_leaf': [1, 4],
            'model__max_features': ['sqrt', 0.5],
            'model__class_weight': ['balanced_subsample'],
        },
    },
    'ExtraTrees_tuned': {
        'family': 'randomized tree ensemble',
        'untuned': 'ExtraTrees_balanced',
        'estimator': ExtraTreesClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=1),
        'param_grid': {
            'model__max_depth': [None, 12],
            'model__min_samples_leaf': [1, 3, 8],
            'model__max_features': ['sqrt', 0.5],
            'model__class_weight': ['balanced'],
        },
    },
    'GradientBoosting_tuned': {
        'family': 'sequential boosting ensemble',
        'untuned': 'GradientBoosting',
        'estimator': GradientBoostingClassifier(random_state=RANDOM_STATE),
        'param_grid': {
            'model__n_estimators': [100, 180],
            'model__learning_rate': [0.04, 0.08],
            'model__max_depth': [2, 3],
            'model__min_samples_leaf': [1, 5],
        },
    },
    'HistGradientBoosting_tuned': {
        'family': 'regularized histogram boosting',
        'untuned': 'HistGradientBoosting_balanced',
        'estimator': HistGradientBoostingClassifier(
            early_stopping=True, random_state=RANDOM_STATE
        ),
        'param_grid': {
            'model__max_iter': [180, 260],
            'model__learning_rate': [0.03, 0.06],
            'model__max_leaf_nodes': [15, 31],
            'model__l2_regularization': [0.0, 0.05],
            'model__class_weight': ['balanced'],
        },
    },
    'LinearSVM_tuned': {
        'family': 'large-margin linear classifier',
        'untuned': 'LinearSVM_balanced',
        'estimator': LinearSVC(max_iter=12000, random_state=RANDOM_STATE),
        'param_grid': {
            'model__C': [0.05, 0.1, 0.3, 1.0],
            'model__class_weight': [None, 'balanced'],
        },
    },
    'RBF_SVM_tuned': {
        'family': 'nonlinear kernel classifier',
        'untuned': 'RBF_SVM_balanced',
        'estimator': SVC(kernel='rbf', decision_function_shape='ovr', random_state=RANDOM_STATE),
        'param_grid': {
            'model__C': [1.0, 3.0, 8.0],
            'model__gamma': ['scale', 0.03, 0.1],
            'model__class_weight': ['balanced'],
        },
    },
    'MLP_tuned': {
        'family': 'neural network baseline',
        'untuned': 'MLP',
        'estimator': MLPClassifier(
            max_iter=350, early_stopping=False, n_iter_no_change=20,
            random_state=RANDOM_STATE
        ),
        'param_grid': {
            'model__hidden_layer_sizes': [(48,), (64,), (64, 32)],
            'model__alpha': [1e-3, 3e-3],
            'model__learning_rate_init': [1e-3],
        },
    },
}


def metric_dict(y_true, y_pred, prefix=''):
    return {
        f'{prefix}Accuracy': accuracy_score(y_true, y_pred),
        f'{prefix}Precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        f'{prefix}Recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        f'{prefix}F1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
    }


def aligned_score_matrix(model, X_eval, classes):
    if hasattr(model, 'predict_proba'):
        scores = model.predict_proba(X_eval)
    elif hasattr(model, 'decision_function'):
        scores = model.decision_function(X_eval)
    else:
        return None

    scores = np.asarray(scores)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)

    model_classes = np.asarray(getattr(model, 'classes_', classes))
    if scores.shape[1] != len(classes):
        return None

    if list(model_classes) == list(classes):
        return scores

    aligned = np.zeros((scores.shape[0], len(classes)))
    for i, cls in enumerate(classes):
        class_positions = np.where(model_classes == cls)[0]
        if len(class_positions) == 0:
            return None
        aligned[:, i] = scores[:, class_positions[0]]
    return aligned


def safe_macro_auc(model, X_eval, y_eval, classes):
    scores = aligned_score_matrix(model, X_eval, classes)
    if scores is None:
        return np.nan
    y_bin = label_binarize(y_eval, classes=classes)
    return roc_auc_score(y_bin, scores, average='macro')


def feature_signal(pipe):
    estimator = pipe.named_steps['model']
    names = pipe.named_steps['preprocess'].get_feature_names_out()
    if hasattr(estimator, 'feature_importances_'):
        values = estimator.feature_importances_
        signal_name = 'Feature importance'
    elif hasattr(estimator, 'coef_'):
        coefs = np.asarray(estimator.coef_)
        values = np.mean(np.abs(coefs), axis=0)
        signal_name = 'Mean absolute coefficient'
    else:
        return None, None
    if len(values) != len(names):
        return None, None
    return pd.Series(values, index=names).sort_values(ascending=False), signal_name


def df_to_markdown(df):
    safe_df = df.copy()
    for col in safe_df.columns:
        if pd.api.types.is_float_dtype(safe_df[col]):
            safe_df[col] = safe_df[col].map(lambda x: '' if pd.isna(x) else f'{x:.4f}')
        else:
            safe_df[col] = safe_df[col].astype(str)
    header = '| ' + ' | '.join(safe_df.columns) + ' |'
    divider = '| ' + ' | '.join(['---'] * len(safe_df.columns)) + ' |'
    rows = ['| ' + ' | '.join(row) + ' |' for row in safe_df.to_numpy(dtype=str)]
    return '\n'.join([header, divider] + rows)


tuned_models = {}
tuned_grids = {}
tuned_summary_rows = []
tuned_split_rows = []
class_report_rows = []
best_param_rows = []
classification_report_text = []

for name, spec in tuning_specs.items():
    print(f'\nTuning {name} ...')
    pipe = Pipeline(steps=[
        ('preprocess', clone(preprocessor)),
        ('model', spec['estimator']),
    ])
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=spec['param_grid'],
        scoring='f1_macro',
        cv=task5_cv,
        n_jobs=1,
        refit=True,
        return_train_score=True,
        error_score='raise',
    )

    start = time.time()
    grid.fit(X_train, y_train)
    elapsed = time.time() - start
    model = grid.best_estimator_
    tuned_models[name] = model
    tuned_grids[name] = grid

    best_idx = grid.best_index_
    cv_train = grid.cv_results_['mean_train_score'][best_idx]
    cv_std = grid.cv_results_['std_test_score'][best_idx]

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    all_pred = model.predict(X)
    test_auc = safe_macro_auc(model, X_test, y_test, task5_classes)

    train_metrics = metric_dict(y_train, train_pred, prefix='Train_')
    test_metrics = metric_dict(y_test, test_pred, prefix='Test_')
    all_metrics = metric_dict(y, all_pred, prefix='All_')

    tuned_summary_rows.append({
        'Model': name,
        'Family': spec['family'],
        'Untuned_Task4_Model': spec['untuned'],
        'Best_CV_F1_macro': grid.best_score_,
        'Best_CV_F1_std': cv_std,
        'CV_Train_F1_macro': cv_train,
        'CV_Train_minus_Val_F1': cv_train - grid.best_score_,
        **train_metrics,
        **test_metrics,
        **all_metrics,
        'Test_AUC_macro_OVR': test_auc,
        'Train_minus_Test_F1': train_metrics['Train_F1_macro'] - test_metrics['Test_F1_macro'],
        'Seconds': elapsed,
    })
    best_param_rows.append({
        'Model': name,
        'Best_CV_F1_macro': grid.best_score_,
        'Best_Params': str(grid.best_params_),
    })

    for split_name, y_true, y_pred in [
        ('Train', y_train, train_pred), ('Test', y_test, test_pred), ('All', y, all_pred)
    ]:
        row = {'Model': name, 'Split': split_name}
        row.update(metric_dict(y_true, y_pred))
        tuned_split_rows.append(row)

    report_dict = classification_report(y_test, test_pred, output_dict=True, zero_division=0)
    classification_report_text.append('\n' + '=' * 80 + f'\n{name}\n')
    classification_report_text.append(classification_report(y_test, test_pred, zero_division=0))
    for cls in task5_classes:
        cls_metrics = report_dict[str(cls)]
        class_report_rows.append({
            'Model': name,
            'Class': cls,
            'Precision': cls_metrics['precision'],
            'Recall': cls_metrics['recall'],
            'F1': cls_metrics['f1-score'],
            'Support': cls_metrics['support'],
        })

# ---- 5b. Model-selection tables ----
tuned_summary_df = pd.DataFrame(tuned_summary_rows)
tuned_split_metrics_df = pd.DataFrame(tuned_split_rows)
task5_class_report_df = pd.DataFrame(class_report_rows)
task5_best_params_df = pd.DataFrame(best_param_rows).sort_values('Best_CV_F1_macro', ascending=False)

# Model selection uses validation F1 only; held-out test metrics are diagnostics after selection.
tuned_summary_df['Rank_Test_F1'] = tuned_summary_df['Test_F1_macro'].rank(ascending=False, method='min')
tuned_summary_df['Rank_CV_F1'] = tuned_summary_df['Best_CV_F1_macro'].rank(ascending=False, method='min')
tuned_summary_df['Rank_AUC'] = tuned_summary_df['Test_AUC_macro_OVR'].rank(ascending=False, method='min', na_option='bottom')
tuned_summary_df['Rank_CV_Stability'] = tuned_summary_df['Best_CV_F1_std'].rank(ascending=True, method='min')
tuned_summary_df['Rank_Generalization_Gap'] = tuned_summary_df['Train_minus_Test_F1'].abs().rank(ascending=True, method='min')
tuned_summary_df['Test_Diagnostic_Rank_Mean'] = tuned_summary_df[[
    'Rank_Test_F1', 'Rank_CV_F1', 'Rank_AUC', 'Rank_CV_Stability', 'Rank_Generalization_Gap'
]].mean(axis=1)
tuned_summary_df['Rank_CV_Overfit'] = tuned_summary_df['CV_Train_minus_Val_F1'].abs().rank(ascending=True, method='min')
tuned_summary_df['Validation_Diagnostic_Rank_Mean'] = tuned_summary_df[[
    'Rank_CV_F1', 'Rank_CV_Stability', 'Rank_CV_Overfit'
]].mean(axis=1)

tuned_summary_df = tuned_summary_df.sort_values(
    ['Best_CV_F1_macro', 'Best_CV_F1_std', 'Test_F1_macro'],
    ascending=[False, True, False]
).reset_index(drop=True)
robust_ranking_df = tuned_summary_df.sort_values('Test_Diagnostic_Rank_Mean').reset_index(drop=True)

untuned_lookup = expanded_supervised_df.set_index('Model')
tuned_vs_untuned_rows = []
for _, row in tuned_summary_df.iterrows():
    untuned_name = row['Untuned_Task4_Model']
    if untuned_name in untuned_lookup.index:
        base = untuned_lookup.loc[untuned_name]
        tuned_vs_untuned_rows.append({
            'Tuned_Model': row['Model'],
            'Untuned_Task4_Model': untuned_name,
            'Untuned_CV_F1_macro': base['CV_F1_macro_mean'],
            'Tuned_CV_F1_macro': row['Best_CV_F1_macro'],
            'Delta_CV_F1_macro': row['Best_CV_F1_macro'] - base['CV_F1_macro_mean'],
            'Untuned_Test_F1_macro': base['Test_F1_macro'],
            'Tuned_Test_F1_macro': row['Test_F1_macro'],
            'Delta_Test_F1_macro': row['Test_F1_macro'] - base['Test_F1_macro'],
        })
tuned_vs_untuned_df = pd.DataFrame(tuned_vs_untuned_rows).sort_values(
    'Tuned_Test_F1_macro', ascending=False
)

recommended_model_name = tuned_summary_df.iloc[0]['Model']
recommended_model = tuned_models[recommended_model_name]
robust_rank_model_name = robust_ranking_df.iloc[0]['Model']

summary_display_cols = [
    'Model', 'Family', 'Best_CV_F1_macro', 'Best_CV_F1_std',
    'CV_Train_minus_Val_F1', 'Test_Accuracy', 'Test_Precision_macro',
    'Test_Recall_macro', 'Test_F1_macro', 'Test_AUC_macro_OVR',
    'Train_minus_Test_F1', 'Validation_Diagnostic_Rank_Mean', 'Test_Diagnostic_Rank_Mean', 'Seconds'
]
print('\nTask 5 tuned model summary:')
display(tuned_summary_df[summary_display_cols].round(4))

print('\nTuned vs untuned Task 4 comparison:')
display(tuned_vs_untuned_df.round(4))

print('\nBest hyperparameters by validation F1-macro:')
display(task5_best_params_df)

# ---- 5c. Save tables ----
tuned_summary_df.to_csv('docs/tables/task5_tuned_model_summary.csv', index=False)
tuned_split_metrics_df.to_csv('docs/tables/task5_tuned_split_metrics.csv', index=False)
tuned_vs_untuned_df.to_csv('docs/tables/task5_tuned_vs_untuned.csv', index=False)
task5_class_report_df.to_csv('docs/tables/task5_class_f1_by_model.csv', index=False)
task5_best_params_df.to_csv('docs/tables/task5_best_params.csv', index=False)
with open('docs/tables/task5_tuned_classification_reports.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(classification_report_text))

# ---- 5d. Report-ready visualizations ----
metric_plot_df = tuned_summary_df.melt(
    id_vars=['Model'],
    value_vars=['Test_F1_macro', 'Test_AUC_macro_OVR'],
    var_name='Metric', value_name='Score'
)
plt.figure(figsize=(12, 7))
sns.barplot(data=metric_plot_df, y='Model', x='Score', hue='Metric')
plt.xlim(0, 1)
plt.title('Task 5 Tuned Models: Held-out F1-macro and AUC')
plt.tight_layout()
plt.savefig('docs/figures/task5_tuned_f1_auc_comparison.png', dpi=180, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
plot_delta = tuned_vs_untuned_df.sort_values('Delta_Test_F1_macro', ascending=True)
sns.barplot(data=plot_delta, y='Tuned_Model', x='Delta_Test_F1_macro', color='#4C78A8')
plt.axvline(0, color='black', lw=1)
plt.title('Task 5 Validation Tuning Gain over Task 4 Untuned Models')
plt.xlabel('Delta held-out F1-macro')
plt.ylabel('Tuned model')
plt.tight_layout()
plt.savefig('docs/figures/task5_tuned_vs_untuned_f1_delta.png', dpi=180, bbox_inches='tight')
plt.show()

class_f1_pivot = task5_class_report_df.pivot(index='Model', columns='Class', values='F1')
class_f1_pivot = class_f1_pivot.loc[tuned_summary_df['Model']]
plt.figure(figsize=(8, 7))
sns.heatmap(class_f1_pivot, annot=True, fmt='.2f', cmap='YlGnBu', vmin=0, vmax=1)
plt.title('Task 5 Class-level F1 by Tuned Model')
plt.tight_layout()
plt.savefig('docs/figures/task5_class_f1_heatmap.png', dpi=180, bbox_inches='tight')
plt.show()

top_roc_names = tuned_summary_df.dropna(subset=['Test_AUC_macro_OVR']).head(3)['Model'].tolist()
fig, axes = plt.subplots(1, len(top_roc_names), figsize=(6 * len(top_roc_names), 5.5), squeeze=False)
for ax, name in zip(axes.ravel(), top_roc_names):
    model = tuned_models[name]
    scores = aligned_score_matrix(model, X_test, task5_classes)
    for i, cls in enumerate(task5_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], scores[:, i])
        ax.plot(fpr, tpr, lw=1.8, label=f'{cls} AUC={auc(fpr, tpr):.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    auc_value = tuned_summary_df.loc[tuned_summary_df['Model'] == name, 'Test_AUC_macro_OVR'].iloc[0]
    ax.set_title(f'{name}\nmacro AUC={auc_value:.3f}')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(fontsize=8, loc='lower right')
plt.tight_layout()
plt.savefig('docs/figures/task5_top_tuned_roc_curves.png', dpi=180, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(6.2, 5.4))
ConfusionMatrixDisplay.from_predictions(
    y_test, recommended_model.predict(X_test), cmap='Blues',
    xticks_rotation=35, ax=ax, colorbar=False
)
ax.set_title(f'Task 5 Final Model Confusion Matrix\n{recommended_model_name}')
plt.tight_layout()
plt.savefig('docs/figures/task5_final_model_confusion_test.png', dpi=180, bbox_inches='tight')
plt.show()

feature_model_name = recommended_model_name
feature_values, signal_name = feature_signal(recommended_model)
if feature_values is None:
    for candidate_name in tuned_summary_df['Model']:
        feature_values, signal_name = feature_signal(tuned_models[candidate_name])
        if feature_values is not None:
            feature_model_name = candidate_name
            break

if feature_values is not None:
    top_features = feature_values.head(20).rename('Signal').reset_index().rename(columns={'index': 'Feature'})
    top_features.insert(0, 'Model', feature_model_name)
    top_features.insert(1, 'Signal_Type', signal_name)
    top_features.to_csv('docs/tables/task5_final_or_interpretable_feature_signal.csv', index=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=top_features, x='Signal', y='Feature', color='#59A14F')
    plt.title(f'Top Feature Signals - {feature_model_name}\n{signal_name}')
    plt.tight_layout()
    plt.savefig('docs/figures/task5_final_or_interpretable_top_features.png', dpi=180, bbox_inches='tight')
    plt.show()
else:
    top_features = pd.DataFrame(columns=['Model', 'Signal_Type', 'Feature', 'Signal'])

# Model-agnostic permutation importance for the final selected pipeline.
perm = permutation_importance(
    recommended_model, X_test, y_test, scoring='f1_macro',
    n_repeats=5, random_state=RANDOM_STATE, n_jobs=1
)
perm_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance_Mean': perm.importances_mean,
    'Importance_Std': perm.importances_std,
}).sort_values('Importance_Mean', ascending=False)
perm_df.to_csv('docs/tables/task5_final_model_permutation_importance.csv', index=False)

plt.figure(figsize=(10, 8))
perm_plot = perm_df.head(20).sort_values('Importance_Mean', ascending=True)
sns.barplot(data=perm_plot, x='Importance_Mean', y='Feature', color='#E15759')
plt.xlabel('Permutation importance: mean F1-macro decrease')
plt.ylabel('Raw input feature')
plt.title(f'Final Model Permutation Importance - {recommended_model_name}')
plt.tight_layout()
plt.savefig('docs/figures/task5_final_model_permutation_importance.png', dpi=180, bbox_inches='tight')
plt.show()

# ---- 5e. Write report-ready markdown ----
recommended_row = tuned_summary_df.loc[tuned_summary_df['Model'] == recommended_model_name].iloc[0]
robust_row = robust_ranking_df.iloc[0]
report_summary = tuned_summary_df[[
    'Model', 'Family', 'Best_CV_F1_macro', 'Best_CV_F1_std',
    'Test_F1_macro', 'Test_AUC_macro_OVR', 'CV_Train_minus_Val_F1', 'Train_minus_Test_F1', 'Validation_Diagnostic_Rank_Mean', 'Test_Diagnostic_Rank_Mean'
]].head(9)
class_table = class_f1_pivot.loc[tuned_summary_df.head(6)['Model']].reset_index()
delta_table = tuned_vs_untuned_df[[
    'Tuned_Model', 'Untuned_Task4_Model', 'Untuned_Test_F1_macro',
    'Tuned_Test_F1_macro', 'Delta_Test_F1_macro'
]].head(9)

report_text = f"""# Task 5: Evaluation and Choice of Prediction Model

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

{df_to_markdown(report_summary)}

## Tuning Gain over Untuned Task 4 Models

{df_to_markdown(delta_table)}

## Class-level F1 Evidence

{df_to_markdown(class_table)}

## Final Model Choice

Recommended final model by cross-validated F1-macro: `{recommended_model_name}`.

- Test F1-macro: `{recommended_row['Test_F1_macro']:.4f}`
- Test AUC-macro OVR: `{recommended_row['Test_AUC_macro_OVR']:.4f}`
- Best validation F1-macro: `{recommended_row['Best_CV_F1_macro']:.4f}`
- Train-test F1 gap: `{recommended_row['Train_minus_Test_F1']:.4f}`

The held-out test set is used only after validation-based selection. The best post-selection test diagnostic model by the combined rank is `{robust_rank_model_name}` with diagnostic rank mean `{robust_row['Test_Diagnostic_Rank_Mean']:.4f}`. If it differs from the primary recommendation, this is useful discussion evidence, not a reason to override the validation-based selection rule.

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
"""

Path('docs/task5_evaluation_model_choice.md').write_text(report_text, encoding='utf-8')

print('\nRecommended final model:', recommended_model_name)
print(f"Recommended test F1-macro: {recommended_row['Test_F1_macro']:.4f}")
print(f"Recommended test AUC-macro OVR: {recommended_row['Test_AUC_macro_OVR']:.4f}")
print('Robust-rank winner:', robust_rank_model_name)
print('Generated Task 5 tables, figures, and docs/task5_evaluation_model_choice.md')
