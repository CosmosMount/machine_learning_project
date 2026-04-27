# Task 2: t-SNE Visualization Analysis

## 1. Task Objective

The project requires visualizing high-dimensional data in a 2D or 3D space using t-SNE, coloring points by class labels where applicable, and discussing observed patterns or clusters.

For this project, t-SNE is used to visualize the Student Dropout dataset after the final leakage-free preprocessing pipeline.

## 2. Method

Input representation:

- Preprocessing source: `X_dense` from the final preprocessing pipeline.
- Shape: `(4424, 172)`.
- The representation uses `StandardScaler` for numeric features, `OneHotEncoder` for nominal categorical features, and passthrough for binary features.
- No target labels are used during preprocessing or embedding construction.

Dimensionality-reduction workflow:

1. Apply PCA from 172 dimensions to 50 dimensions.
2. Run t-SNE from 50 dimensions to 2 dimensions.
3. Color the final 2D points by the true `Target` label only for interpretation.

PCA is used before t-SNE because it reduces noise and improves runtime stability while retaining most of the feature variance.

Key settings:

| Step | Setting |
|---|---|
| PCA dimensions | 50 |
| PCA cumulative explained variance | 0.9744 |
| t-SNE perplexity | 30 |
| t-SNE initialization | PCA |
| t-SNE learning rate | auto |
| Random state | 42 |
| t-SNE KL divergence | 1.4732 |

The generated figure is saved at:

- `docs/figures/tsne_projection.png`

## 2.1 Perplexity Sensitivity

To avoid over-relying on a single t-SNE parameter setting, we also compare:

- `perplexity = 10`
- `perplexity = 30`
- `perplexity = 50`

The sensitivity figure is saved at:

- `docs/figures/task2_tsne_perplexity_sensitivity.png`

The exact geometry changes across perplexity values, as expected, but the main qualitative pattern remains stable: the three outcome classes overlap substantially rather than forming cleanly separated groups.

Important limitation:

t-SNE is mainly reliable for local neighborhood visualization. Axis direction, global distances, and visual cluster sizes should not be interpreted as exact quantitative measurements. For this reason, the centroid and spread statistics are used only as supporting evidence, while final model assessment relies on supervised metrics.

## 3. Results

Class centroids on the t-SNE plane:

| Target | TSNE-1 | TSNE-2 |
|---|---:|---:|
| Dropout | -23.056 | 7.078 |
| Enrolled | 4.013 | 0.330 |
| Graduate | 13.597 | -6.690 |

Within-class spread:

| Target | Mean | Median | Std |
|---|---:|---:|---:|
| Dropout | 41.195 | 40.931 | 15.890 |
| Enrolled | 36.222 | 36.234 | 16.935 |
| Graduate | 39.718 | 40.460 | 17.147 |

Centroid distances:

| Target | Dropout | Enrolled | Graduate |
|---|---:|---:|---:|
| Dropout | 0.000 | 27.898 | 39.154 |
| Enrolled | 27.898 | 0.000 | 11.880 |
| Graduate | 39.154 | 11.880 | 0.000 |

The closest class-centroid pair is:

- `Enrolled` and `Graduate`, distance = `11.880`.

## 4. Interpretation

The t-SNE visualization shows partial class structure, but the three classes are not cleanly separable. This is expected because student outcomes are influenced by overlapping academic, demographic, and socioeconomic factors.

The most visually similar pair is `Enrolled` and `Graduate`, based on centroid distance in the t-SNE plane. This suggests that currently enrolled students may share many academic and demographic characteristics with students who eventually graduate. In contrast, `Dropout` is more separated from `Graduate` in the embedding, implying stronger differences in the learned feature representation.

The large within-class spreads indicate that each class is internally heterogeneous. Therefore, the visualization should be treated as qualitative evidence of overlap rather than proof of hard clusters.

## 5. Why This Supports the Overall Modeling Plan

The t-SNE result supports the use of supervised classification rather than relying on unsupervised structure alone:

- There is some class-related organization in the feature space.
- The classes still overlap substantially.
- Natural clusters are unlikely to perfectly match `Dropout`, `Enrolled`, and `Graduate`.
- Macro-averaged supervised metrics are necessary because the minority class `Enrolled` is both underrepresented and visually mixed.

This also prepares the logic for the next task, clustering analysis: clustering can be used as a supporting exploratory tool, but it should not be expected to recover the target labels perfectly.

## 6. Report-Ready Summary

After leakage-free preprocessing, we first reduced the 172-dimensional feature matrix to 50 PCA components, retaining 97.44% of the variance, and then applied t-SNE to obtain a 2D embedding. The resulting plot shows partial class structure but substantial overlap among outcomes. `Enrolled` and `Graduate` have the closest t-SNE centroids, suggesting that currently enrolled students share many characteristics with eventual graduates. `Dropout` is more separated from `Graduate`, but all classes show wide within-class spread. Therefore, t-SNE provides useful qualitative evidence about class overlap, while also motivating supervised classification and macro-averaged evaluation for the downstream prediction task.
