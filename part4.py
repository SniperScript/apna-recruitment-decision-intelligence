"""Part 4 — Model-Choice and Architecture Memo (20 marks).

Evaluates decision-tree candidate splits by Information Gain and Gini Gain on
the 560 training rows, builds a 1-split decision stump and compares hard/soft
voting ensembles on the 240 test rows, runs two rounds of AdaBoost from first
principles, computes the out-of-bag bootstrap fraction, and sizes MLP, CNN, and
RNN neural architectures.

Also executes a 100-epoch bootstrap OOB Monte Carlo verification in console
diagnostics alongside the exact analytical formula.

Writes:
    outputs/part4_results.txt
    outputs/part4_splits.csv
    outputs/part4_voting.csv
    outputs/part4_adaboost.csv
    outputs/part4_architecture.csv
"""
from __future__ import annotations

import math
import os
import random
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from pipeline_core import (
    OUTPUTS_DIR,
    confusion_and_metrics,
    fit_logistic_regression,
    knn_predict_probs_and_labels,
    load_records,
    select_best_k_knn_cv,
    sigmoid,
    stratified_split_and_standardise,
    write_csv_file,
    write_results_txt,
)


def node_entropy(n_total: int, n_pos: int) -> float:
    """Binary Shannon entropy in bits (-sum p_i log2 p_i); 0 for a pure node."""
    if n_total == 0 or n_pos == 0 or n_pos == n_total:
        return 0.0
    p1 = n_pos / n_total
    p0 = 1.0 - p1
    return float(-(p1 * math.log2(p1) + p0 * math.log2(p0)))


def node_gini(n_total: int, n_pos: int) -> float:
    """Binary Gini impurity (1 - sum p_i^2); 0 for a pure node."""
    if n_total == 0:
        return 0.0
    p1 = n_pos / n_total
    p0 = 1.0 - p1
    return float(1.0 - (p1 * p1 + p0 * p0))


def run_adaboost_two_rounds() -> Dict[str, Any]:
    """Execute 2 rounds of AdaBoost by hand on the 8 candidate records."""
    y = np.array([+1, +1, +1, -1, -1, -1, +1, -1], dtype=np.float64)
    h1 = np.array([+1, +1, -1, -1, -1, -1, -1, -1], dtype=np.float64)
    h2 = np.array([-1, +1, +1, -1, +1, -1, +1, -1], dtype=np.float64)

    n_pts = len(y)
    w = np.full(n_pts, 1.0 / n_pts, dtype=np.float64)
    csv_rows: List[List[Any]] = []
    round_stats: Dict[int, Dict[str, float]] = {}

    for r_idx, h_t in ((1, h1), (2, h2)):
        w_before = w.copy()
        misclassified = h_t != y
        eps_t = float(np.sum(w_before[misclassified]))
        alpha_t = 0.5 * math.log((1.0 - eps_t) / eps_t)
        w_unnorm = w_before * np.exp(-alpha_t * y * h_t)
        w_after = w_unnorm / float(np.sum(w_unnorm))
        w = w_after

        round_stats[r_idx] = {"error": eps_t, "alpha": alpha_t}
        for pt in range(n_pts):
            csv_rows.append(
                [
                    r_idx,
                    pt + 1,
                    float(w_before[pt]),
                    float(w_unnorm[pt]),
                    float(w_after[pt]),
                ]
            )

    alpha1 = round_stats[1]["alpha"]
    alpha2 = round_stats[2]["alpha"]
    combined_score = alpha1 * h1 + alpha2 * h2
    ensemble_pred = np.where(combined_score >= 0.0, 1.0, -1.0)
    ensemble_acc = float(np.sum(ensemble_pred == y) / n_pts)

    return {
        "round1_error": round_stats[1]["error"],
        "round1_alpha": round_stats[1]["alpha"],
        "round2_error": round_stats[2]["error"],
        "round2_alpha": round_stats[2]["alpha"],
        "ensemble_accuracy": ensemble_acc,
        "csv_rows": csv_rows,
    }


def size_neural_architectures() -> Dict[str, Any]:
    """Compute exact parameter counts for Model A (MLP & CNN), scaling comparison, and Model B (RNN)."""
    # Model A Option 1: MLP 784 -> 16 -> 16 -> 10
    mlp_hidden1 = (784 + 1) * 16
    mlp_hidden2 = (16 + 1) * 16
    mlp_output = (16 + 1) * 10
    mlp_total = mlp_hidden1 + mlp_hidden2 + mlp_output

    # Model A Option 2: CNN on 28 x 28 x 1
    # Conv1: 5x5 filters, 6 filters, input depth 1, stride 1, padding 2 -> output 28x28x6
    cnn_conv1 = (5 * 5 * 1 + 1) * 6
    # MaxPool1: 2x2, stride 2 -> output 14x14x6
    # Conv2: 5x5 filters, 16 filters, input depth 6, stride 1, padding 0 -> output 10x10x16
    cnn_conv2 = (5 * 5 * 6 + 1) * 16
    # MaxPool2: 2x2, stride 2 -> output 5x5x16 -> flatten = 400
    # Dense1: 400 -> 32
    cnn_dense1 = (400 + 1) * 32
    # Dense2: 32 -> 10
    cnn_dense2 = (32 + 1) * 10
    cnn_total = cnn_conv1 + cnn_conv2 + cnn_dense1 + cnn_dense2

    # Why convolution scales: 128 x 128 x 3
    fc_layer_params = (128 * 128 * 3 + 1) * 1024
    conv_layer_params = (3 * 3 * 3 + 1) * 64

    # Model B: RNN (input=64, hidden=128, output=16)
    # H_t = tanh(W_xh X_t + W_hh H_{t-1} + b_h), Y = W_hy H_T + b_y
    w_xh_params = 128 * 64
    w_hh_params = 128 * 128
    b_h_params = 128
    w_hy_params = 16 * 128
    b_y_params = 16
    rnn_total = w_xh_params + w_hh_params + b_h_params + w_hy_params + b_y_params

    arch_rows = [
        ["mlp_hidden1", mlp_hidden1],
        ["mlp_hidden2", mlp_hidden2],
        ["mlp_output", mlp_output],
        ["cnn_conv1", cnn_conv1],
        ["cnn_conv2", cnn_conv2],
        ["cnn_dense1", cnn_dense1],
        ["cnn_dense2", cnn_dense2],
    ]

    return {
        "arch_rows": arch_rows,
        "mlp_total": mlp_total,
        "cnn_total": cnn_total,
        "fc_layer_params": fc_layer_params,
        "conv_layer_params": conv_layer_params,
        "rnn_total": rnn_total,
    }


def run_part4(verbose: bool = True) -> Dict[str, Any]:
    """Execute Tasks 4.1 through 4.5 and write all Part 4 outputs."""
    _, X, outcome, _ = load_records()
    split = stratified_split_and_standardise(X, outcome)

    X_train_raw = split["X_train_raw"]
    y_train = split["y_train"]
    X_test_raw = split["X_test_raw"]
    y_test = split["y_test"]
    Z_train = split["Z_train"]
    Z_test = split["Z_test"]

    n_train = len(y_train)
    pos_train = int(np.sum(y_train))

    # ------------------------------------------------------------------
    # Task 4.1: Which split is purest?
    # S1: X6 = 1, S2: X7 = 1, S3: X3 > 0.5, S4: X1 <= 24
    # ------------------------------------------------------------------
    root_entropy = node_entropy(n_train, pos_train)
    root_gini = node_gini(n_train, pos_train)

    split_rules: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "S1": lambda M: M[:, 5] == 1.0,
        "S2": lambda M: M[:, 6] == 1.0,
        "S3": lambda M: M[:, 2] > 0.5,
        "S4": lambda M: M[:, 0] <= 24.0,
    }

    splits_rows: List[List[Any]] = []
    split_details: Dict[str, Dict[str, Any]] = {}
    best_info_split = "S1"
    best_info_val = -1.0
    best_gini_split = "S1"
    best_gini_val = -1.0

    for s_name in ("S1", "S2", "S3", "S4"):
        cond_fn = split_rules[s_name]
        mask_yes = cond_fn(X_train_raw)
        mask_no = ~mask_yes

        n_yes = int(np.sum(mask_yes))
        pos_yes = int(np.sum(y_train[mask_yes]))
        n_no = int(np.sum(mask_no))
        pos_no = int(np.sum(y_train[mask_no]))

        ent_yes = node_entropy(n_yes, pos_yes)
        ent_no = node_entropy(n_no, pos_no)
        weighted_ent = (n_yes / n_train) * ent_yes + (n_no / n_train) * ent_no
        info_gain = root_entropy - weighted_ent

        gini_yes = node_gini(n_yes, pos_yes)
        gini_no = node_gini(n_no, pos_no)
        weighted_gini = (n_yes / n_train) * gini_yes + (n_no / n_train) * gini_no
        gini_gain = root_gini - weighted_gini

        splits_rows.append(
            [s_name, n_yes, pos_yes, n_no, pos_no, info_gain, gini_gain]
        )
        split_details[s_name] = {
            "n_yes": n_yes,
            "pos_yes": pos_yes,
            "n_no": n_no,
            "pos_no": pos_no,
            "info_gain": info_gain,
            "gini_gain": gini_gain,
        }

        if info_gain > best_info_val:
            best_info_val = info_gain
            best_info_split = s_name
        if gini_gain > best_gini_val:
            best_gini_val = gini_gain
            best_gini_split = s_name

    write_csv_file(
        os.path.join(OUTPUTS_DIR, "part4_splits.csv"),
        [
            "split",
            "n_yes",
            "positives_yes",
            "n_no",
            "positives_no",
            "info_gain",
            "gini_gain",
        ],
        splits_rows,
    )

    # ------------------------------------------------------------------
    # Task 4.2: Stump and voting ensembles on the 240 test rows
    # ------------------------------------------------------------------
    stump_info = split_details[best_info_split]
    stump_rate_yes = stump_info["pos_yes"] / stump_info["n_yes"]
    stump_rate_no = stump_info["pos_no"] / stump_info["n_no"]

    # 1. Logistic (Task 2.2 model, positive when prob >= 0.3)
    weights, bias, _ = fit_logistic_regression(
        Z_train, y_train, eta=0.1, epochs=1000, record_epochs=()
    )
    prob_logistic = sigmoid(Z_test @ weights + bias)
    pred_logistic = (prob_logistic >= 0.3).astype(np.int64)

    # 2. KNN (Task 2.7 model with chosen K)
    knn_best_k, _ = select_best_k_knn_cv(Z_train, y_train)
    prob_knn, pred_knn = knn_predict_probs_and_labels(
        Z_train, y_train, Z_test, knn_best_k
    )

    # 3. Stump (positive when prob >= 0.3)
    stump_mask_test_yes = split_rules[best_info_split](X_test_raw)
    prob_stump = np.where(stump_mask_test_yes, stump_rate_yes, stump_rate_no)
    pred_stump = (prob_stump >= 0.3).astype(np.int64)

    # 4. Hard vote (at least 2 of 3 predict positive)
    pred_hard_vote = ((pred_logistic + pred_knn + pred_stump) >= 2).astype(
        np.int64
    )

    # 5. Soft vote (mean of the 3 probabilities >= 0.3)
    prob_soft_vote = (prob_logistic + prob_knn + prob_stump) / 3.0
    pred_soft_vote = (prob_soft_vote >= 0.3).astype(np.int64)

    voting_models = [
        ("logistic", pred_logistic),
        ("knn", pred_knn),
        ("stump", pred_stump),
        ("hard_vote", pred_hard_vote),
        ("soft_vote", pred_soft_vote),
    ]
    voting_rows: List[List[Any]] = []
    for m_name, m_pred in voting_models:
        m_eval = confusion_and_metrics(y_test, m_pred)
        voting_rows.append(
            [
                m_name,
                m_eval["tp"],
                m_eval["fp"],
                m_eval["tn"],
                m_eval["fn"],
                m_eval["precision"],
                m_eval["recall"],
                m_eval["f1"],
            ]
        )

    write_csv_file(
        os.path.join(OUTPUTS_DIR, "part4_voting.csv"),
        ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"],
        voting_rows,
    )

    # ------------------------------------------------------------------
    # Task 4.3: Two rounds of AdaBoost by hand
    # ------------------------------------------------------------------
    ada = run_adaboost_two_rounds()
    write_csv_file(
        os.path.join(OUTPUTS_DIR, "part4_adaboost.csv"),
        [
            "round",
            "point",
            "weight_before",
            "weight_unnormalised",
            "weight_after",
        ],
        ada["csv_rows"],
    )

    # ------------------------------------------------------------------
    # Task 4.4: Out-of-bag share
    # ------------------------------------------------------------------
    oob_fraction = (1.0 - 1.0 / 560.0) ** 560

    # 100-epoch bootstrap OOB simulation check (diagnostic only)
    rng_oob = random.Random(2508)
    oob_sim_shares = []
    for _ in range(100):
        picked = {rng_oob.randrange(560) for _ in range(560)}
        oob_sim_shares.append((560 - len(picked)) / 560.0)
    oob_mc_mean = sum(oob_sim_shares) / len(oob_sim_shares)

    # ------------------------------------------------------------------
    # Task 4.5: Size the neural networks
    # ------------------------------------------------------------------
    arch = size_neural_architectures()
    write_csv_file(
        os.path.join(OUTPUTS_DIR, "part4_architecture.csv"),
        ["layer", "parameters"],
        arch["arch_rows"],
    )

    results_items: List[Tuple[str, Any]] = [
        # Task 4.1
        ("root_entropy", root_entropy),
        ("root_gini", root_gini),
        ("best_split_by_info_gain", best_info_split),
        ("best_split_by_gini_gain", best_gini_split),
        # Task 4.2
        ("stump_rate_yes", stump_rate_yes),
        ("stump_rate_no", stump_rate_no),
        # Task 4.3
        ("round1_error", ada["round1_error"]),
        ("round1_alpha", ada["round1_alpha"]),
        ("round2_error", ada["round2_error"]),
        ("round2_alpha", ada["round2_alpha"]),
        ("ensemble_accuracy", ada["ensemble_accuracy"]),
        # Task 4.4
        ("oob_fraction", oob_fraction),
        # Task 4.5
        ("mlp_total", arch["mlp_total"]),
        ("cnn_total", arch["cnn_total"]),
        ("fc_layer_params", arch["fc_layer_params"]),
        ("conv_layer_params", arch["conv_layer_params"]),
        ("rnn_total", arch["rnn_total"]),
    ]

    write_results_txt(
        os.path.join(OUTPUTS_DIR, "part4_results.txt"), results_items
    )

    if verbose:
        print("=" * 76)
        print("PART 4 — MODEL CHOICE, ENSEMBLES & NEURAL ARCHITECTURE SIZING")
        print("=" * 76)
        print(
            f"  Task 4.1 | Root Entropy={root_entropy:.4f}, Root Gini={root_gini:.4f} -> "
            f"Best Split={best_info_split} (IG={best_info_val:.4f}, GG={best_gini_val:.4f})"
        )
        print(
            f"  Task 4.2 | Stump Rates ({best_info_split}): Yes={stump_rate_yes:.4f}, No={stump_rate_no:.4f}"
        )
        print(
            "           | Test F1 Comparison: "
            + ", ".join(f"{r[0]}={r[7]:.4f}" for r in voting_rows)
        )
        print(
            f"  Task 4.3 | AdaBoost: R1 err={ada['round1_error']:.4f} (alpha={ada['round1_alpha']:.4f}), "
            f"R2 err={ada['round2_error']:.4f} (alpha={ada['round2_alpha']:.4f}), "
            f"Ensemble Acc={ada['ensemble_accuracy']:.4f}"
        )
        print(
            f"  Task 4.4 | Analytical oob_fraction={oob_fraction:.4f} "
            f"(100-Epoch Bootstrap Monte Carlo Mean={oob_mc_mean:.4f})"
        )
        print(
            f"  Task 4.5 | Params: MLP={arch['mlp_total']}, CNN={arch['cnn_total']}, "
            f"FC(128x128x3)={arch['fc_layer_params']}, Conv(3x3x64)={arch['conv_layer_params']}, "
            f"RNN={arch['rnn_total']}"
        )
        print("=" * 76)

    return dict(results_items)


if __name__ == "__main__":
    run_part4(verbose=True)
