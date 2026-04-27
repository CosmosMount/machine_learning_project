"""Full diagnostic: reproduce the notebook pipeline end-to-end and report
every metric so we can pinpoint where performance is bad."""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, PowerTransformer, label_binarize,
)
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score,
)
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score,
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42

# ============================================================
# 1. LOAD DATA
# ============================================================
try:
    df = pd.read_csv('data/data.csv', sep=';')
except FileNotFoundError:
    df = pd.read_csv('data.csv', sep=';')
df.columns = df.columns.str.strip()
target_col = 'Target'
X_raw = df.drop(columns=[target_col]).copy()
y = df[target_col].copy()

BINARY_COLS = [
    'Daytime/evening attendance', 'Displaced', 'Educational special needs',
    'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder',
    'International',
]
NOMINAL_COLS = [
    'Marital status', 'Application mode', 'Course',
    'Previous qualification', 'Nacionality',
    "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
]

DROP_ALL = [
    'International', "Father's occupation",
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Nacionality', 'Educational special needs',
]

X = X_raw.drop(columns=DROP_ALL).copy()
BINARY_COLS = [c for c in BINARY_COLS if c not in DROP_ALL]
NOMINAL_COLS = [c for c in NOMINAL_COLS if c not in DROP_ALL]
NUMERIC_COLS = [c for c in X.columns if c not in BINARY_COLS + NOMINAL_COLS]

# Feature engineering
X['approval_rate_1st'] = X['Curricular units 1st sem (approved)'] / (X['Curricular units 1st sem (evaluations)'] + 1)
X['approval_rate_2nd'] = X['Curricular units 2nd sem (approved)'] / (X['Curricular units 2nd sem (evaluations)'] + 1)
X['grade_trend'] = X['Curricular units 2nd sem (grade)'] - X['Curricular units 1st sem (grade)']
NUMERIC_COLS = NUMERIC_COLS + ['approval_rate_1st', 'approval_rate_2nd', 'grade_trend']


class MulticlassTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smooth=10):
        self.smooth = smooth
    def fit(self, X, y=None):
        target_map = {c: i for i, c in enumerate(sorted(y.unique()))}
        y_num = y.map(target_map)
        global_mean = y_num.mean()
        self.encodings_ = {}
        X_df = pd.DataFrame(X)
        for col_idx in range(X_df.shape[1]):
            col = X_df.iloc[:, col_idx]
            stats = pd.DataFrame({'val': col, 'target': y_num.values}).groupby('val')['target']
            means = stats.mean()
            counts = stats.count()
            smoothed = (counts * means + self.smooth * global_mean) / (counts + self.smooth)
            self.encodings_[col_idx] = (smoothed.to_dict(), global_mean)
        return self
    def transform(self, X, y=None):
        X_df = pd.DataFrame(X)
        result = np.zeros_like(X_df, dtype=float)
        for col_idx in range(X_df.shape[1]):
            mapping, fallback = self.encodings_[col_idx]
            result[:, col_idx] = X_df.iloc[:, col_idx].map(mapping).fillna(fallback).values
        return result
    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features)
        return np.array([f'target_enc_{i}' for i in range(len(self.encodings_))])


# ============================================================
# 2. BUILD AND COMPARE MULTIPLE PIPELINES
# ============================================================
def make_v3_preprocessor():
    """Current V3: PowerTransformer + TargetEncoder."""
    return ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                          ('pow', PowerTransformer(method='yeo-johnson', standardize=True))]),
         NUMERIC_COLS),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                          ('te', MulticlassTargetEncoder(smooth=10))]),
         NOMINAL_COLS),
        ('bin', 'passthrough', BINARY_COLS),
    ], remainder='drop')

def make_v2_preprocessor():
    """V2: StandardScaler + OneHotEncoder (previous working version)."""
    return ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                          ('sc', StandardScaler())]),
         NUMERIC_COLS),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                          ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))]),
         NOMINAL_COLS),
        ('bin', 'passthrough', BINARY_COLS),
    ], remainder='drop')

def make_v2_no_feat_eng():
    """V2 without feature engineering: like original V2."""
    X_v2 = X_raw.drop(columns=DROP_ALL).copy()
    bc = [c for c in ['Daytime/evening attendance', 'Displaced', 'Debtor',
                       'Tuition fees up to date', 'Gender', 'Scholarship holder'] if c in X_v2.columns]
    nc = [c for c in ['Marital status', 'Application mode', 'Course',
                       'Previous qualification',
                       "Mother's qualification", "Father's qualification",
                       "Mother's occupation"] if c in X_v2.columns]
    nuc = [c for c in X_v2.columns if c not in bc + nc]
    pre = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), nuc),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                          ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))]), nc),
        ('bin', 'passthrough', bc),
    ], remainder='drop')
    return pre, X_v2


print("=" * 70)
print("DIAGNOSTIC REPORT")
print("=" * 70)

# ============================================================
# 3. CHECK FOR DATA LEAKAGE IN TARGET ENCODER
# ============================================================
print("\n--- Check: TargetEncoder leakage ---")
pre_v3 = make_v3_preprocessor()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y)

# Fit on ALL data (leakage) vs fit on train only
pre_v3_leak = make_v3_preprocessor()
X_all_transformed = pre_v3_leak.fit_transform(X, y)
print(f"  fit on ALL data  -> shape: {X_all_transformed.shape}")

pre_v3_clean = make_v3_preprocessor()
X_tr_transformed = pre_v3_clean.fit_transform(X_train, y_train)
X_te_transformed = pre_v3_clean.transform(X_test)
print(f"  fit on TRAIN only -> train shape: {X_tr_transformed.shape}, test shape: {X_te_transformed.shape}")

# Check NaN in test transform
nan_count = np.isnan(X_te_transformed).sum()
print(f"  NaN in test after transform: {nan_count}")

# ============================================================
# 4. CLASSIFICATION METRICS COMPARISON
# ============================================================
print("\n--- Classification: V3 (PowerTransformer + TargetEncoder + FeatEng) ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
models = {
    'LR': LogisticRegression(max_iter=600, random_state=RANDOM_STATE),
    'DT': DecisionTreeClassifier(random_state=RANDOM_STATE),
    'RF': RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    'LR_balanced': LogisticRegression(max_iter=600, random_state=RANDOM_STATE, class_weight='balanced'),
    'RF_balanced': RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight='balanced'),
}

for mname, model in models.items():
    pipe = Pipeline([('pre', make_v3_preprocessor()), ('mdl', model)])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='f1_macro', n_jobs=1)
    print(f"  {mname:20s} CV F1-macro: {scores.mean():.4f} +/- {scores.std():.4f}")

# Train/test split metrics
print("\n--- Test-set metrics (70/30 split) ---")
for mname, model in models.items():
    pipe = Pipeline([('pre', make_v3_preprocessor()), ('mdl', model)])
    pipe.fit(X_train, y_train)
    yp = pipe.predict(X_test)
    acc = accuracy_score(y_test, yp)
    f1 = f1_score(y_test, yp, average='macro')
    prec = precision_score(y_test, yp, average='macro')
    rec = recall_score(y_test, yp, average='macro')
    print(f"  {mname:20s}: Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
    if mname in ['LR', 'RF_balanced']:
        print(f"    Per-class report:\n{classification_report(y_test, yp)}")

# ============================================================
# 5. COMPARE V2 (OHE) vs V3 (TargetEnc) on same data
# ============================================================
print("\n--- V2 (StandardScaler + OHE + FeatEng) for comparison ---")
for mname in ['LR', 'RF']:
    model = models[mname]
    pipe = Pipeline([('pre', make_v2_preprocessor()), ('mdl', model)])
    pipe.fit(X_train, y_train)
    yp = pipe.predict(X_test)
    f1 = f1_score(y_test, yp, average='macro')
    acc = accuracy_score(y_test, yp)
    print(f"  V2 {mname:20s}: Acc={acc:.4f}  F1={f1:.4f}")

    pipe3 = Pipeline([('pre', make_v3_preprocessor()), ('mdl', type(model)(**model.get_params()))])
    pipe3.fit(X_train, y_train)
    yp3 = pipe3.predict(X_test)
    f1_3 = f1_score(y_test, yp3, average='macro')
    acc3 = accuracy_score(y_test, yp3)
    print(f"  V3 {mname:20s}: Acc={acc3:.4f}  F1={f1_3:.4f}")

# ============================================================
# 6. CLUSTERING
# ============================================================
print("\n--- Clustering diagnostics ---")
pre_v3_full = make_v3_preprocessor()
X_dense = pre_v3_full.fit_transform(X, y)
y_codes = pd.Categorical(y).codes

for k in [3, 4, 5]:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
    labels = km.fit_predict(X_dense)
    sil = silhouette_score(X_dense, labels)
    ari = adjusted_rand_score(y_codes, labels)
    nmi = normalized_mutual_info_score(y_codes, labels)
    print(f"  k={k}: Silhouette={sil:.4f}  ARI={ari:.4f}  NMI={nmi:.4f}")

# ============================================================
# 7. CHECK: is the problem TargetEncoder or PowerTransformer?
# ============================================================
print("\n--- Ablation: isolate each component ---")

# A: StandardScaler + OHE (no feat eng)
pre_a, X_a = make_v2_no_feat_eng()
pipe_a = Pipeline([('pre', pre_a), ('mdl', RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))])
X_tr_a, X_te_a, y_tr_a, y_te_a = train_test_split(X_a, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)
pipe_a.fit(X_tr_a, y_tr_a)
f1_a = f1_score(y_te_a, pipe_a.predict(X_te_a), average='macro')
print(f"  A) V2 no feat eng (SS+OHE)       : RF F1={f1_a:.4f}")

# B: StandardScaler + OHE + feat eng
pipe_b = Pipeline([('pre', make_v2_preprocessor()), ('mdl', RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))])
pipe_b.fit(X_train, y_train)
f1_b = f1_score(y_test, pipe_b.predict(X_test), average='macro')
print(f"  B) V2 + feat eng (SS+OHE+FE)     : RF F1={f1_b:.4f}")

# C: PowerTransformer + OHE + feat eng
pre_c = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                      ('pow', PowerTransformer(method='yeo-johnson', standardize=True))]),
     NUMERIC_COLS),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                      ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))]),
     NOMINAL_COLS),
    ('bin', 'passthrough', BINARY_COLS),
], remainder='drop')
pipe_c = Pipeline([('pre', pre_c), ('mdl', RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))])
pipe_c.fit(X_train, y_train)
f1_c = f1_score(y_test, pipe_c.predict(X_test), average='macro')
print(f"  C) PowerTransformer+OHE+FE       : RF F1={f1_c:.4f}")

# D: StandardScaler + TargetEncoder + feat eng
pre_d = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                      ('sc', StandardScaler())]),
     NUMERIC_COLS),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                      ('te', MulticlassTargetEncoder(smooth=10))]),
     NOMINAL_COLS),
    ('bin', 'passthrough', BINARY_COLS),
], remainder='drop')
pipe_d = Pipeline([('pre', pre_d), ('mdl', RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))])
pipe_d.fit(X_train, y_train)
f1_d = f1_score(y_test, pipe_d.predict(X_test), average='macro')
print(f"  D) StandardScaler+TargetEnc+FE   : RF F1={f1_d:.4f}")

# E: Full V3 (PowerTransformer + TargetEncoder + feat eng)
pipe_e = Pipeline([('pre', make_v3_preprocessor()), ('mdl', RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))])
pipe_e.fit(X_train, y_train)
f1_e = f1_score(y_test, pipe_e.predict(X_test), average='macro')
print(f"  E) Full V3 (PT+TE+FE)            : RF F1={f1_e:.4f}")

# ============================================================
# 8. CHECK: tuned models
# ============================================================
print("\n--- Tuned models (GridSearchCV) ---")
logi_grid = GridSearchCV(
    Pipeline([('pre', make_v3_preprocessor()), ('mdl', LogisticRegression(max_iter=600, random_state=RANDOM_STATE))]),
    param_grid={'mdl__C': [0.01, 0.1, 1.0, 10.0], 'mdl__class_weight': [None, 'balanced']},
    cv=cv, scoring='f1_macro', n_jobs=1
)
logi_grid.fit(X_train, y_train)
yp_logi = logi_grid.predict(X_test)
print(f"  LR best: {logi_grid.best_params_}  CV={logi_grid.best_score_:.4f}  Test F1={f1_score(y_test, yp_logi, average='macro'):.4f}")

rf_grid = GridSearchCV(
    Pipeline([('pre', make_v3_preprocessor()), ('mdl', RandomForestClassifier(random_state=RANDOM_STATE))]),
    param_grid={'mdl__n_estimators': [200, 400], 'mdl__max_depth': [None, 15, 25],
                'mdl__class_weight': [None, 'balanced', 'balanced_subsample']},
    cv=cv, scoring='f1_macro', n_jobs=1
)
rf_grid.fit(X_train, y_train)
yp_rf = rf_grid.predict(X_test)
print(f"  RF best: {rf_grid.best_params_}  CV={rf_grid.best_score_:.4f}  Test F1={f1_score(y_test, yp_rf, average='macro'):.4f}")

print("\nDone.")
