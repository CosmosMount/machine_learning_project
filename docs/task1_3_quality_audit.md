# Task 1-3 Quality Audit

## 1. Audit Standard

This audit checks the completed parts of the project against two standards:

1. **Project requirement compliance**: every mandatory item in the DSAA2011 project PDF must be covered.
2. **High-score standard**: the implementation should be statistically valid, reproducible, interpretable, and stronger than the minimum requirement without introducing leakage or unjustified complexity.

## 2. Task 1: Data Preprocessing

### Requirement Coverage

| PDF Requirement | Current Status |
|---|---|
| Examine missing values | Completed. Total missing values = 0. |
| Handle missing values | Completed through pipeline imputers for robustness. |
| Handle non-numeric values | Completed by semantic categorical typing and one-hot encoding. |
| Further processing, e.g. standardization | Completed by `StandardScaler` for numeric/count features. |
| Brief description and insights | Completed in notebook and `preprocessing_final.md`. |

### High-Score Features

- Manual feature typing is based on semantic meaning rather than pandas dtype.
- Integer-coded categories such as `Course` and `Application mode` are not treated as continuous values.
- One-hot encoding avoids fake ordinal distances and avoids target leakage.
- Numeric features are standardized after median imputation.
- Binary features are passed through unchanged.
- Near-zero-variance and high-correlation filtering reduce noise and redundancy.
- Academic trajectory features are added:
  - `approval_rate_1st`
  - `approval_rate_2nd`
  - `grade_trend`
- Additional diagnostic visualizations are included:
  - class distribution / imbalance plot,
  - numeric correlation heatmap,
  - raw-vs-standardized scale comparison.

### Final Output

| Item | Value |
|---|---:|
| Raw dataset shape | `(4424, 37)` |
| Final modeling feature matrix | `(4424, 172)` |
| Missing / infinite values after preprocessing | `0 / 0` |

### Risk Review

The previous target-encoding idea was rejected as the main preprocessing plan because it uses labels during feature construction and can leak information into t-SNE and clustering. The current preprocessing is safer and more defensible.

The high-correlation audit is not applied mechanically. Some highly correlated academic variables are retained when they represent different time points in student progression and carry clear domain meaning.

## 3. Task 2: t-SNE Visualization

### Requirement Coverage

| PDF Requirement | Current Status |
|---|---|
| Visualize high-dimensional data using t-SNE | Completed. |
| Produce 2D scatter plot | Completed. |
| Color by class labels if applicable | Completed. |
| Discuss patterns or clusters | Completed in notebook and `tsne_analysis.md`. |

### High-Score Features

- t-SNE uses the leakage-free preprocessing matrix.
- PCA is applied before t-SNE for denoising and stability.
- PCA keeps 50 dimensions and retains 97.44% variance.
- Target labels are used only for coloring after embedding construction.
- Class centroids and within-class spread are reported to support the visual interpretation.
- Perplexity sensitivity is tested with `10`, `30`, and `50`.
- The notebook explicitly states the t-SNE caveat: local neighborhoods are more reliable than global geometry.

### Main Result

| Item | Value |
|---|---:|
| PCA dimensions before t-SNE | 50 |
| PCA cumulative explained variance | 0.9744 |
| t-SNE KL divergence | 1.4732 |
| Closest class-centroid pair | `Enrolled` - `Graduate` |

### Interpretation

The t-SNE plot shows partial class structure but substantial overlap. This supports the need for supervised classification and macro-averaged metrics.

## 4. Task 3: Clustering Analysis

### Requirement Coverage

| PDF Requirement | Current Status |
|---|---|
| At least two clustering algorithms | Completed with seven families: KMeans, MiniBatchKMeans, Agglomerative Ward, BIRCH, Spectral Clustering, Gaussian Mixture, and DBSCAN. |
| Apply to preprocessed dataset | Completed on 50D PCA representation from the leakage-free preprocessing matrix. |
| Evaluate with multiple metrics | Completed with internal and external metrics. |
| Visualize clusters | Completed. |
| Determine best result and justify | Completed. |

### High-Score Features

- Seven clustering families are compared, not just two.
- `k = 2` to `8` is scanned rather than assuming `k = 3`.
- DBSCAN density settings are scanned separately because DBSCAN discovers its own effective cluster count and can mark noise.
- Internal metrics and external reference metrics are separated methodologically.
- ARI/NMI are used only for interpretation, not for fitting.
- Results are connected back to the t-SNE observation and the need for supervised learning.

### Main Result

| Selection | Result |
|---|---|
| Best internal clustering | `KMeans_k2` |
| Best three-cluster setting | `KMeans_k3` |

The strongest natural structure appears coarser than the three target labels. The best `k=3` clustering only partially aligns with `Dropout`, `Enrolled`, and `Graduate`.

## 5. Current Quality Judgment

Tasks 1-3 are now compliant with the project PDF and meet a high-score standard. The implementation avoids target leakage, uses appropriate preprocessing for tabular data, includes clear visualizations, and provides defensible interpretation.

Further improvement should now focus on supervised learning, because the unsupervised tasks already show that natural clusters cannot fully recover the three labels.

## 6. Files Produced

Documentation:

- `docs/preprocessing_final.md`
- `docs/preprocessing_legacy_report.txt`
- `docs/tsne_analysis.md`
- `docs/clustering_analysis.md`

Figures:

- `docs/figures/task1_class_distribution.png`
- `docs/figures/task1_numeric_correlation_heatmap.png`
- `docs/figures/task1_preprocessing_scale_effect.png`
- `docs/figures/tsne_projection.png`
- `docs/figures/task2_tsne_perplexity_sensitivity.png`
- `docs/figures/clustering_silhouette_by_k.png`
- `docs/figures/clustering_k3_on_tsne.png`
