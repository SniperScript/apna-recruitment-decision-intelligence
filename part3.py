"""Part 3 — Segmentation and Dimensionality Reduction (25 marks).

Implements K-means clustering from scratch with pinned initialization, elbow
selection, segment profiling on K=3, agglomerative hierarchical clustering
across single/complete/average linkages on the 6-cohort dissimilarity table,
and 7-step Principal Component Analysis (PCA).

Also executes a 100-seed Monte Carlo K-means stability benchmark in console
diagnostics to contextualize the pinned initialization against global optima.

Writes:
    outputs/part3_results.txt
    outputs/part3_segments.csv
    outputs/part3_hierarchical.csv
"""
from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from pipeline_core import (
    OUTPUTS_DIR,
    load_records,
    sample_mean_1d,
    sample_mean_2d,
    standardise_fit_transform,
    write_csv_file,
    write_results_txt,
)

DISSIMILARITY_TABLE = np.array(
    [
        [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
        [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
        [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
        [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
        [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
        [6.1, 2.0, 1.3, 4.5, 3.4, 0.0],
    ],
    dtype=np.float64,
)


def run_kmeans_from_indices(
    Z: np.ndarray, init_indices: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """Run K-means from scratch given starting row indices."""
    n = Z.shape[0]
    k = len(init_indices)
    centroids = Z[list(init_indices)].copy()
    labels = np.full(n, -1, dtype=np.int64)
    passes = 0

    while True:
        passes += 1
        diff = Z[:, None, :] - centroids[None, :, :]
        sq_dists = np.sum(diff * diff, axis=2)
        new_labels = np.argmin(sq_dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c_idx in range(k):
            members = Z[labels == c_idx]
            if len(members) > 0:
                centroids[c_idx] = sample_mean_2d(members)

    final_diff = Z - centroids[labels]
    wcss = float(np.sum(final_diff * final_diff))
    return labels, centroids, wcss, passes


def run_kmeans_pinned(
    Z: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """Run K-means from scratch with pinned starting centroids:
    indices 0, floor(n/K), ..., (K-1)*floor(n/K).
    """
    n = Z.shape[0]
    step = n // k
    init_indices = [j * step for j in range(k)]
    return run_kmeans_from_indices(Z, init_indices)


def cluster_distance(
    c1: Sequence[int],
    c2: Sequence[int],
    dist_mat: np.ndarray,
    linkage: str,
) -> float:
    """Compute single, complete, or average linkage distance between two clusters."""
    pairwise = [dist_mat[i, j] for i in c1 for j in c2]
    if linkage == "single":
        return float(min(pairwise))
    if linkage == "complete":
        return float(max(pairwise))
    if linkage == "average":
        return float(sum(pairwise) / len(pairwise))
    raise ValueError(f"Unknown linkage: {linkage}")


def format_cluster_name(members: Sequence[int]) -> str:
    """Format sorted member indices (0-based) as 'E1+E3+...'."""
    return "+".join(f"E{idx + 1}" for idx in sorted(members))


def run_hierarchical_clustering(
    dist_mat: np.ndarray, linkage: str
) -> Tuple[List[List[Any]], int, List[str]]:
    """Run agglomerative clustering for a given linkage rule on 6 items."""
    clusters: List[List[int]] = [[i] for i in range(dist_mat.shape[0])]
    merges: List[List[Any]] = []
    heights: List[float] = []
    snapshots_after_merge: List[List[List[int]]] = []

    for merge_no in range(1, dist_mat.shape[0]):
        best_dist = float("inf")
        best_pair = (0, 1)
        m = len(clusters)
        for i in range(m):
            for j in range(i + 1, m):
                d = cluster_distance(clusters[i], clusters[j], dist_mat, linkage)
                if d < best_dist:
                    best_dist = d
                    best_pair = (i, j)

        idx_first, idx_second = best_pair
        c_first = sorted(clusters[idx_first])
        c_second = sorted(clusters[idx_second])
        if c_first[0] < c_second[0]:
            c_a, c_b = c_first, c_second
        else:
            c_a, c_b = c_second, c_first

        merges.append(
            [
                linkage,
                merge_no,
                format_cluster_name(c_a),
                format_cluster_name(c_b),
                best_dist,
            ]
        )
        heights.append(best_dist)

        merged_cluster = sorted(c_a + c_b)
        remaining = [
            clusters[k] for k in range(m) if k not in (idx_first, idx_second)
        ]
        remaining.append(merged_cluster)
        clusters = sorted(remaining, key=lambda cl: min(cl))
        snapshots_after_merge.append([list(cl) for cl in clusters])

    gaps = [heights[i + 1] - heights[i] for i in range(len(heights) - 1)]
    best_gap_idx = int(np.argmax(gaps))
    merges_below_cut = best_gap_idx + 1
    clusters_at_cut = 6 - merges_below_cut
    cut_partition = [
        format_cluster_name(cl)
        for cl in snapshots_after_merge[merges_below_cut - 1]
    ]
    return merges, clusters_at_cut, cut_partition


def run_pca_seven_steps(Z: np.ndarray) -> Dict[str, Any]:
    """Execute the 7-step PCA algorithm on the standardised 800 x 5 matrix."""
    n = Z.shape[0]
    # Step 1: mean vector (zeros after standardising)
    mean_vec = sample_mean_2d(Z)
    # Step 2: centre the data
    X_centered = Z - mean_vec
    # Step 3: covariance matrix S = (1 / (n - 1)) X^T X
    S = (X_centered.T @ X_centered) / (n - 1)
    # Step 4: eigenvalues and eigenvectors via numpy.linalg.eigh
    eigvals, eigvecs = np.linalg.eigh(S)
    # Step 5: sort by eigenvalue, largest first
    desc_order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[desc_order]
    eigvecs = eigvecs[:, desc_order]
    # Sign convention: flip each eigenvector so its entry of largest absolute value is positive
    for j in range(eigvecs.shape[1]):
        col = eigvecs[:, j]
        max_abs_idx = int(np.argmax(np.abs(col)))
        if col[max_abs_idx] < 0.0:
            eigvecs[:, j] = -col

    # Step 6: keep top two eigenvectors
    W = eigvecs[:, :2]
    # Step 7: project Z_proj = X_centered @ W
    Z_proj = X_centered @ W

    eig_sum = float(np.sum(eigvals))
    explained = eigvals / eig_sum
    cumulative = np.cumsum(explained)
    comp_80 = int(next(i + 1 for i, c in enumerate(cumulative) if c >= 0.8))

    return {
        "mean_vec": mean_vec,
        "eigenvalue_sum": eig_sum,
        "eigenvalues": eigvals,
        "explained": explained,
        "cumulative": cumulative,
        "components_for_80pct": comp_80,
        "W": W,
        "Z_proj": Z_proj,
    }


def run_part3(verbose: bool = True) -> Dict[str, Any]:
    """Execute Tasks 3.1 through 3.5 and write all Part 3 outputs."""
    _, X, outcome, value = load_records()
    X5 = X[:, :5]

    # ------------------------------------------------------------------
    # Task 3.1: K-means from scratch on standardised X1..X5
    # ------------------------------------------------------------------
    Z5, _, _ = standardise_fit_transform(X5)
    kmeans_results: Dict[int, Dict[str, Any]] = {}
    for k in (2, 3, 4, 5, 6):
        labels, centroids, wcss, passes = run_kmeans_pinned(Z5, k)
        kmeans_results[k] = {
            "labels": labels,
            "centroids": centroids,
            "wcss": wcss,
            "passes": passes,
        }

    # 100-epoch Monte Carlo K=3 initialization stability check (diagnostic only)
    rng_mc = random.Random(2508)
    mc_min_wcss_k3 = kmeans_results[3]["wcss"]
    for _ in range(100):
        idx_mc = rng_mc.sample(range(len(Z5)), 3)
        _, _, w_mc, _ = run_kmeans_from_indices(Z5, idx_mc)
        if w_mc < mc_min_wcss_k3:
            mc_min_wcss_k3 = w_mc

    # ------------------------------------------------------------------
    # Task 3.2: Choose K with the elbow
    # ------------------------------------------------------------------
    elbow_k = 3
    for k_cand in (3, 4, 5):
        drop_after = kmeans_results[k_cand]["wcss"] - kmeans_results[k_cand + 1]["wcss"]
        drop_before = kmeans_results[k_cand - 1]["wcss"] - kmeans_results[k_cand]["wcss"]
        if drop_after < 0.5 * drop_before:
            elbow_k = k_cand
            break

    # ------------------------------------------------------------------
    # Task 3.3: Profile the K = 3 segments (ordered by ascending mean of X3)
    # ------------------------------------------------------------------
    labels_k3 = kmeans_results[3]["labels"]
    cluster_means_x3 = [
        (c_idx, sample_mean_1d(X5[labels_k3 == c_idx, 2]))
        for c_idx in range(3)
    ]
    ordered_clusters = [
        c_idx for c_idx, _ in sorted(cluster_means_x3, key=lambda item: item[1])
    ]
    segment_names = ["A", "B", "C"]
    segment_masks: Dict[str, np.ndarray] = {}
    segment_rows: List[List[Any]] = []
    segment_sizes: Dict[str, int] = {}

    for seg_name, orig_cluster_id in zip(segment_names, ordered_clusters):
        mask = labels_k3 == orig_cluster_id
        segment_masks[seg_name] = mask
        seg_size = int(np.sum(mask))
        segment_sizes[seg_name] = seg_size
        seg_X5 = X5[mask]
        means_x = [sample_mean_1d(seg_X5[:, j]) for j in range(5)]
        out_rate = sample_mean_1d(outcome[mask].astype(np.float64))
        mean_val = sample_mean_1d(value[mask])
        segment_rows.append(
            [
                seg_name,
                seg_size,
                means_x[0],
                means_x[1],
                means_x[2],
                means_x[3],
                means_x[4],
                out_rate,
                mean_val,
            ]
        )

    write_csv_file(
        os.path.join(OUTPUTS_DIR, "part3_segments.csv"),
        [
            "segment",
            "size",
            "mean_X1",
            "mean_X2",
            "mean_X3",
            "mean_X4",
            "mean_X5",
            "outcome_rate",
            "mean_value",
        ],
        segment_rows,
    )

    # ------------------------------------------------------------------
    # Task 3.4: Hierarchical clustering on supplied 6x6 table
    # ------------------------------------------------------------------
    all_hier_rows: List[List[Any]] = []
    cut_counts: Dict[str, int] = {}
    cut_partitions: Dict[str, List[str]] = {}

    for linkage in ("single", "complete", "average"):
        merges, count_at_cut, partition = run_hierarchical_clustering(
            DISSIMILARITY_TABLE, linkage
        )
        all_hier_rows.extend(merges)
        cut_counts[linkage] = count_at_cut
        cut_partitions[linkage] = partition

    write_csv_file(
        os.path.join(OUTPUTS_DIR, "part3_hierarchical.csv"),
        ["linkage", "merge_no", "cluster_a", "cluster_b", "height"],
        all_hier_rows,
    )

    # ------------------------------------------------------------------
    # Task 3.5: PCA by the seven steps
    # ------------------------------------------------------------------
    pca = run_pca_seven_steps(Z5)
    Z_proj = pca["Z_proj"]
    W = pca["W"]
    eigvals = pca["eigenvalues"]
    explained = pca["explained"]
    cumulative = pca["cumulative"]

    mean_pc1_seg: Dict[str, float] = {}
    for seg_name in segment_names:
        mean_pc1_seg[seg_name] = sample_mean_1d(
            Z_proj[segment_masks[seg_name], 0]
        )

    results_items: List[Tuple[str, Any]] = [
        # Task 3.1
        ("wcss_k2", kmeans_results[2]["wcss"]),
        ("wcss_k3", kmeans_results[3]["wcss"]),
        ("wcss_k4", kmeans_results[4]["wcss"]),
        ("wcss_k5", kmeans_results[5]["wcss"]),
        ("wcss_k6", kmeans_results[6]["wcss"]),
        # Task 3.2
        ("elbow_k", elbow_k),
        # Task 3.3
        ("size_segment_A", segment_sizes["A"]),
        ("size_segment_B", segment_sizes["B"]),
        ("size_segment_C", segment_sizes["C"]),
        # Task 3.4
        ("single_clusters_at_cut", cut_counts["single"]),
        ("complete_clusters_at_cut", cut_counts["complete"]),
        ("average_clusters_at_cut", cut_counts["average"]),
        # Task 3.5
        ("eigenvalue_sum", pca["eigenvalue_sum"]),
        ("eigenvalue_1", float(eigvals[0])),
        ("eigenvalue_2", float(eigvals[1])),
        ("eigenvalue_3", float(eigvals[2])),
        ("eigenvalue_4", float(eigvals[3])),
        ("eigenvalue_5", float(eigvals[4])),
        ("explained_ratio_1", float(explained[0])),
        ("explained_ratio_2", float(explained[1])),
        ("explained_ratio_3", float(explained[2])),
        ("explained_ratio_4", float(explained[3])),
        ("explained_ratio_5", float(explained[4])),
        ("cumulative_ratio_1", float(cumulative[0])),
        ("cumulative_ratio_2", float(cumulative[1])),
        ("cumulative_ratio_3", float(cumulative[2])),
        ("cumulative_ratio_4", float(cumulative[3])),
        ("cumulative_ratio_5", float(cumulative[4])),
        ("components_for_80pct", pca["components_for_80pct"]),
        ("pc1_loading_X1", float(W[0, 0])),
        ("pc1_loading_X2", float(W[1, 0])),
        ("pc1_loading_X3", float(W[2, 0])),
        ("pc1_loading_X4", float(W[3, 0])),
        ("pc1_loading_X5", float(W[4, 0])),
        ("pc2_loading_X1", float(W[0, 1])),
        ("pc2_loading_X2", float(W[1, 1])),
        ("pc2_loading_X3", float(W[2, 1])),
        ("pc2_loading_X4", float(W[3, 1])),
        ("pc2_loading_X5", float(W[4, 1])),
        ("row1_pc1", float(Z_proj[0, 0])),
        ("row1_pc2", float(Z_proj[0, 1])),
        ("row2_pc1", float(Z_proj[1, 0])),
        ("row2_pc2", float(Z_proj[1, 1])),
        ("row3_pc1", float(Z_proj[2, 0])),
        ("row3_pc2", float(Z_proj[2, 1])),
        ("mean_pc1_segment_A", mean_pc1_seg["A"]),
        ("mean_pc1_segment_B", mean_pc1_seg["B"]),
        ("mean_pc1_segment_C", mean_pc1_seg["C"]),
    ]

    write_results_txt(
        os.path.join(OUTPUTS_DIR, "part3_results.txt"), results_items
    )

    if verbose:
        print("=" * 76)
        print("PART 3 — SEGMENTATION & DIMENSIONALITY REDUCTION (800 RECORDS, X1..X5)")
        print("=" * 76)
        print(
            "  Task 3.1 | Pinned K-Means WCSS: "
            + ", ".join(
                f"K={k}: {kmeans_results[k]['wcss']:.4f} ({kmeans_results[k]['passes']} passes)"
                for k in (2, 3, 4, 5, 6)
            )
        )
        print(
            f"           | 100-Epoch Monte Carlo K=3 Stability Audit: "
            f"Pinned WCSS={kmeans_results[3]['wcss']:.4f} vs 100-Run Min={mc_min_wcss_k3:.4f}"
        )
        print(
            f"  Task 3.2 | Elbow K={elbow_k} | Segment Sizes: A={segment_sizes['A']}, "
            f"B={segment_sizes['B']}, C={segment_sizes['C']}"
        )
        print(
            f"  Task 3.4 | Dendrogram Cuts: Single={cut_counts['single']} {cut_partitions['single']}, "
            f"Complete={cut_counts['complete']} {cut_partitions['complete']}, "
            f"Average={cut_counts['average']} {cut_partitions['average']}"
        )
        print(
            f"  Task 3.5 | PCA: eig_sum={pca['eigenvalue_sum']:.4f}, "
            f"top-3 cumulative={cumulative[2]:.4f} (components_for_80pct={pca['components_for_80pct']})"
        )
        print("=" * 76)

    return dict(results_items)


if __name__ == "__main__":
    run_part3(verbose=True)
