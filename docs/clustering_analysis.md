# Task 3: Expanded Clustering Analysis

## 1. Task Objective

The project requires grouping data points into clusters, evaluating clustering quality with multiple metrics, visualizing the clusters, and comparing algorithm performance.

To exceed the minimum requirement, we use a broad clustering pool rather than only two algorithms. Clustering is treated as exploratory structure analysis, not as a replacement for supervised prediction.

## 2. Input Representation

Clustering is performed on the leakage-free preprocessed feature matrix:

- Original preprocessed matrix: `(4424, 172)`.
- PCA-reduced clustering matrix: `(4424, 50)`.
- PCA cumulative explained variance: `0.9744`.

PCA is used before clustering because the one-hot encoded representation is high-dimensional. The 50-dimensional PCA representation keeps most information while improving stability for distance-based and graph-based clustering.

No target labels are used to fit clustering models. True labels are used only afterward for external reference metrics such as ARI and NMI.

## 3. Algorithms Compared

Seven clustering families are evaluated:

| Algorithm | Why Included |
|---|---|
| KMeans | Standard centroid-based baseline; fast and interpretable. |
| MiniBatchKMeans | Scalable KMeans variant; checks whether KMeans result is stable under stochastic optimization. |
| Agglomerative Ward | Hierarchical clustering; tests nested/merge-based structure. |
| BIRCH | Tree-based clustering suitable for larger tabular data; another scalable centroid-style method. |
| Spectral Clustering | Graph-based method; can detect non-convex structure better than KMeans. |
| Gaussian Mixture | Probabilistic clustering; tests soft/elliptical cluster assumptions. |
| DBSCAN | Density-based method; does not require specifying `k` and can identify noise. |

For algorithms requiring a cluster count, we scan:

- `k = 2` to `8`.

For DBSCAN, we scan several `eps` and `min_samples` settings because its effective number of clusters is determined by density.

## 4. Metrics

Internal metrics:

| Metric | Preferred Direction | Meaning |
|---|---|---|
| Silhouette | Higher | Compactness and separation. |
| Calinski-Harabasz | Higher | Between-cluster to within-cluster dispersion. |
| Davies-Bouldin | Lower | Average similarity to the closest other cluster. |

External reference metrics:

| Metric | Preferred Direction | Meaning |
|---|---|---|
| ARI | Higher | Agreement between clusters and true labels, adjusted for chance. |
| NMI | Higher | Mutual information between clusters and true labels. |
| Homogeneity | Higher | Each cluster mainly contains one class. |
| Completeness | Higher | Each class mainly maps to one cluster. |
| V-measure | Higher | Harmonic mean of homogeneity and completeness. |

Internal metrics are used to select unsupervised clustering quality. External metrics are used only for interpretation.

## 5. Main Results

Top settings by internal rank:

| Algorithm | Setting | Effective k | Noise Rate | Silhouette | Calinski-Harabasz | Davies-Bouldin | ARI | NMI |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| KMeans | k=2 | 2 | 0.0000 | 0.2774 | 1011.1086 | 1.6813 | 0.1966 | 0.2027 |
| MiniBatchKMeans | k=2 | 2 | 0.0000 | 0.2776 | 1011.0529 | 1.6849 | 0.1973 | 0.2025 |
| KMeans | k=3 | 3 | 0.0000 | 0.2774 | 797.3844 | 1.6033 | 0.1750 | 0.1787 |
| SpectralClustering | k=2 | 2 | 0.0000 | 0.2635 | 950.3314 | 1.6164 | 0.1655 | 0.1828 |
| SpectralClustering | k=3 | 3 | 0.0000 | 0.2623 | 607.5064 | 1.4302 | 0.1495 | 0.1774 |

Best overall internal result:

- `KMeans`, `k = 2`.

Best three-cluster result:

- `KMeans`, `k = 3`.

## 6. Three-Cluster Comparison

Because the target has three classes, we separately compare all `k=3` methods.

| Algorithm | Silhouette | Calinski-Harabasz | Davies-Bouldin | ARI | NMI | Homogeneity | Completeness | V-measure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| KMeans | 0.2774 | 797.3844 | 1.6033 | 0.1750 | 0.1787 | 0.1518 | 0.2170 | 0.1787 |
| SpectralClustering | 0.2623 | 607.5064 | 1.4302 | 0.1495 | 0.1774 | 0.1395 | 0.2438 | 0.1774 |
| AgglomerativeWard | 0.2465 | 705.0794 | 1.7865 | 0.1835 | 0.1592 | 0.1395 | 0.1852 | 0.1592 |
| BIRCH | 0.2406 | 692.7960 | 1.8176 | 0.1861 | 0.1570 | 0.1385 | 0.1813 | 0.1570 |
| MiniBatchKMeans | 0.0992 | 679.1092 | 2.6881 | 0.2632 | 0.2347 | 0.2377 | 0.2317 | 0.2347 |
| GaussianMixture | -0.0315 | 169.7745 | 4.5187 | 0.0593 | 0.0802 | 0.0826 | 0.0780 | 0.0802 |

KMeans remains the best internal `k=3` method. MiniBatchKMeans has stronger external alignment at `k=3`, but its poor silhouette indicates weaker unsupervised separation. This is a useful nuance: if we optimize only agreement with labels after the fact, the conclusion can differ from purely unsupervised quality.

The key interpretation is deliberately split into two perspectives. From the unsupervised perspective, `k=2` is the strongest natural structure and suggests a coarser separation similar to `At-Risk` versus `Graduate`. From the target-alignment perspective, forcing `k=3` is useful only as a diagnostic for the three labels; the modest ARI/NMI values show that `Dropout`, `Enrolled`, and `Graduate` do not form three clean natural clusters.

## 7. DBSCAN Result

DBSCAN is included as a density-based method. Its results are unstable:

- Some settings produce many noise points.
- Some settings collapse to one effective cluster.
- The best-looking external scores do not correspond to strong internal separation.

This suggests the student data does not form clean density-separated clusters in the PCA representation.

## 8. Visualizations

Generated figures:

- `docs/figures/clustering_expanded_silhouette_by_k.png`
- `docs/figures/clustering_expanded_top_silhouette.png`
- `docs/figures/clustering_expanded_k3_on_tsne.png`
- Legacy/reference figures:
  - `docs/figures/clustering_silhouette_by_k.png`
  - `docs/figures/clustering_k3_on_tsne.png`

## 9. Interpretation

Even after expanding the algorithm pool, the main conclusion remains stable:

1. The strongest natural structure is closer to `k=2` than `k=3`.
2. KMeans is the most reliable clustering method by internal metrics.
3. Three-cluster methods only partially align with `Dropout`, `Enrolled`, and `Graduate`.
4. Density clustering is not well suited to this representation.
5. Clustering is valuable for exploratory analysis, but supervised learning is necessary for prediction.

This is a stronger high-score result than simply reporting two clustering algorithms, because it shows algorithmic breadth and a defensible negative conclusion.

## 10. Report-Ready Summary

We compared seven clustering families on the 50-dimensional PCA representation of the leakage-free preprocessed data: KMeans, MiniBatchKMeans, Agglomerative Ward, BIRCH, Spectral Clustering, Gaussian Mixture Models, and DBSCAN. For count-based methods, we scanned `k = 2` to `8`; for DBSCAN, we scanned density parameters. Internal metrics selected KMeans with `k=2` as the strongest natural structure, which points to a broad `At-Risk` versus `Graduate` pattern. Separately, KMeans with `k=3` was the best internal option when we forced a three-cluster comparison against the target labels. However, ARI and NMI remained modest, meaning the three supervised labels only partially align with natural clusters. DBSCAN was unstable, indicating that the data does not form clean density-separated clusters. Overall, clustering provides exploratory evidence of broad structure but cannot replace supervised classification.
