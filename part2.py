"""Part 2 — Classification and Evaluation (30 marks).

Builds a logistic-regression classifier and a K-nearest-neighbours classifier
from scratch on the pinned 70/30 stratified split, evaluates learning rates,
odds ratios, ROC threshold sweeps, 5-fold cross-validation, and INR business-cost
threshold optimization.

Also executes a +100-epoch post-convergence trajectory audit (epochs 1001..1100)
in console diagnostics without altering the pinned 1000-epoch graded outputs.

Writes:
    outputs/part2_results.txt
    outputs/part2_thresholds.csv
    outputs/part2_knn_cv.csv
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from pipeline_core import (
    OUTPUTS_DIR,
    binary_cross_entropy,
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


def compute_business_cost(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """Compute business cost in INR: 50 * fp + 400 * fn."""
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return 50 * fp + 400 * fn


def run_part2(verbose: bool = True) -> Dict[str, Any]:
    """Execute Tasks 2.1 through 2.8 and write all Part 2 outputs."""
    _, X, outcome, _ = load_records()

    # ------------------------------------------------------------------
    # Task 2.1: Split and standardise
    # ------------------------------------------------------------------
    split = stratified_split_and_standardise(X, outcome)
    Z_train = split["Z_train"]
    y_train = split["y_train"]
    Z_test = split["Z_test"]
    y_test = split["y_test"]

    train_rows = int(len(y_train))
    train_positives = int(np.sum(y_train))
    test_rows = int(len(y_test))
    test_positives = int(np.sum(y_test))

    # ------------------------------------------------------------------
    # Task 2.2: Logistic regression from scratch (1000 pinned updates)
    # ------------------------------------------------------------------
    weights, bias, losses_main = fit_logistic_regression(
        Z_train,
        y_train,
        eta=0.1,
        epochs=1000,
        record_epochs=(0, 1, 10, 100, 1000),
    )
    train_probs = sigmoid(Z_train @ weights + bias)
    test_probs = sigmoid(Z_test @ weights + bias)
    test_bce = binary_cross_entropy(y_test, test_probs)

    # +100-epoch post-convergence trajectory check (epochs 1001..1100, diagnostic only)
    w_ext, b_ext, losses_ext = fit_logistic_regression(
        Z_train,
        y_train,
        eta=0.1,
        epochs=1100,
        record_epochs=(1000, 1025, 1050, 1075, 1100),
    )
    max_weight_drift_100ep = float(np.max(np.abs(w_ext - weights)))

    # ------------------------------------------------------------------
    # Task 2.3: Learning-rate effect (200 updates for eta in {0.01, 0.1, 1.0})
    # ------------------------------------------------------------------
    lr_losses: Dict[float, float] = {}
    for eta_val in (0.01, 0.1, 1.0):
        _, _, rec = fit_logistic_regression(
            Z_train, y_train, eta=eta_val, epochs=200, record_epochs=(200,)
        )
        lr_losses[eta_val] = rec[200]

    # ------------------------------------------------------------------
    # Task 2.4: Odds ratios
    # ------------------------------------------------------------------
    odds_ratios = [math.exp(float(w_j)) for w_j in weights]

    # ------------------------------------------------------------------
    # Task 2.5: Evaluate on test rows at tau = 0.5 and tau = 0.3
    # ------------------------------------------------------------------
    eval_05 = confusion_and_metrics(y_test, (test_probs >= 0.5).astype(np.int64))
    eval_03 = confusion_and_metrics(y_test, (test_probs >= 0.3).astype(np.int64))
    blind_preds = np.zeros_like(y_test)
    blind_accuracy = float(np.sum(blind_preds == y_test) / len(y_test))

    # ------------------------------------------------------------------
    # Task 2.6: Threshold sweep (tau = 0.1, 0.2, ..., 0.9)
    # ------------------------------------------------------------------
    threshold_rows: List[List[Any]] = []
    for k in range(1, 10):
        tau_val = round(k * 0.1, 1)
        m_tau = confusion_and_metrics(y_test, (test_probs >= tau_val).astype(np.int64))
        threshold_rows.append(
            [
                tau_val,
                m_tau["tp"],
                m_tau["fp"],
                m_tau["tn"],
                m_tau["fn"],
                m_tau["tpr"],
                m_tau["fpr"],
            ]
        )
    write_csv_file(
        os.path.join(OUTPUTS_DIR, "part2_thresholds.csv"),
        ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"],
        threshold_rows,
    )

    # ------------------------------------------------------------------
    # Task 2.7: K-nearest neighbours with 5-fold cross-validation
    # ------------------------------------------------------------------
    knn_best_k, cv_rows = select_best_k_knn_cv(Z_train, y_train)
    write_csv_file(
        os.path.join(OUTPUTS_DIR, "part2_knn_cv.csv"),
        ["K", "mean_cv_f1"],
        cv_rows,
    )

    _, knn_test_preds = knn_predict_probs_and_labels(
        Z_train, y_train, Z_test, knn_best_k
    )
    knn_eval = confusion_and_metrics(y_test, knn_test_preds)

    # ------------------------------------------------------------------
    # Task 2.8: Choose alert threshold with money (INR 50 * fp + INR 400 * fn)
    # ------------------------------------------------------------------
    best_cost_tau = 0.05
    best_train_cost: int | None = None

    for k in range(1, 20):
        tau_cand = k / 20.0
        preds_tr = (train_probs >= tau_cand).astype(np.int64)
        c_tr = compute_business_cost(y_train, preds_tr)
        if best_train_cost is None or c_tr < best_train_cost:
            best_train_cost = c_tr
            best_cost_tau = tau_cand

    assert best_train_cost is not None
    cost_test_at_tau = compute_business_cost(
        y_test, (test_probs >= best_cost_tau).astype(np.int64)
    )
    cost_test_at_05 = compute_business_cost(
        y_test, (test_probs >= 0.5).astype(np.int64)
    )
    cost_test_flag_nobody = compute_business_cost(y_test, np.zeros_like(y_test))

    results_items: List[Tuple[str, Any]] = [
        # Task 2.1
        ("train_rows", train_rows),
        ("train_positives", train_positives),
        ("test_rows", test_rows),
        ("test_positives", test_positives),
        # Task 2.2
        ("loss_epoch_0", losses_main[0]),
        ("loss_epoch_1", losses_main[1]),
        ("loss_epoch_10", losses_main[10]),
        ("loss_epoch_100", losses_main[100]),
        ("loss_epoch_1000", losses_main[1000]),
        ("weight_X1", float(weights[0])),
        ("weight_X2", float(weights[1])),
        ("weight_X3", float(weights[2])),
        ("weight_X4", float(weights[3])),
        ("weight_X5", float(weights[4])),
        ("weight_X6", float(weights[5])),
        ("weight_X7", float(weights[6])),
        ("bias", float(bias)),
        ("test_bce", test_bce),
        # Task 2.3
        ("loss_epoch_200_eta_0.01", lr_losses[0.01]),
        ("loss_epoch_200_eta_0.1", lr_losses[0.1]),
        ("loss_epoch_200_eta_1.0", lr_losses[1.0]),
        # Task 2.4
        ("odds_ratio_X1", odds_ratios[0]),
        ("odds_ratio_X2", odds_ratios[1]),
        ("odds_ratio_X3", odds_ratios[2]),
        ("odds_ratio_X4", odds_ratios[3]),
        ("odds_ratio_X5", odds_ratios[4]),
        ("odds_ratio_X6", odds_ratios[5]),
        ("odds_ratio_X7", odds_ratios[6]),
        # Task 2.5 (tau = 0.5)
        ("tau_0.5_tp", eval_05["tp"]),
        ("tau_0.5_fp", eval_05["fp"]),
        ("tau_0.5_tn", eval_05["tn"]),
        ("tau_0.5_fn", eval_05["fn"]),
        ("tau_0.5_accuracy", eval_05["accuracy"]),
        ("tau_0.5_precision", eval_05["precision"]),
        ("tau_0.5_recall", eval_05["recall"]),
        ("tau_0.5_f1", eval_05["f1"]),
        ("tau_0.5_specificity", eval_05["specificity"]),
        # Task 2.5 (tau = 0.3)
        ("tau_0.3_tp", eval_03["tp"]),
        ("tau_0.3_fp", eval_03["fp"]),
        ("tau_0.3_tn", eval_03["tn"]),
        ("tau_0.3_fn", eval_03["fn"]),
        ("tau_0.3_accuracy", eval_03["accuracy"]),
        ("tau_0.3_precision", eval_03["precision"]),
        ("tau_0.3_recall", eval_03["recall"]),
        ("tau_0.3_f1", eval_03["f1"]),
        ("tau_0.3_specificity", eval_03["specificity"]),
        ("blind_accuracy", blind_accuracy),
        # Task 2.7
        ("knn_best_k", knn_best_k),
        ("knn_tp", knn_eval["tp"]),
        ("knn_fp", knn_eval["fp"]),
        ("knn_tn", knn_eval["tn"]),
        ("knn_fn", knn_eval["fn"]),
        ("knn_accuracy", knn_eval["accuracy"]),
        ("knn_precision", knn_eval["precision"]),
        ("knn_recall", knn_eval["recall"]),
        ("knn_f1", knn_eval["f1"]),
        # Task 2.8
        ("cost_tau", best_cost_tau),
        ("cost_train_at_tau", best_train_cost),
        ("cost_test_at_tau", cost_test_at_tau),
        ("cost_test_at_0.5", cost_test_at_05),
        ("cost_test_flag_nobody", cost_test_flag_nobody),
    ]

    write_results_txt(
        os.path.join(OUTPUTS_DIR, "part2_results.txt"), results_items
    )

    if verbose:
        print("=" * 76)
        print("PART 2 — CLASSIFICATION & COST-SENSITIVE EVALUATION (560 TRAIN / 240 TEST)")
        print("=" * 76)
        print(
            f"  Task 2.1 | Stratified Split: Train={train_rows} ({train_positives} pos), "
            f"Test={test_rows} ({test_positives} pos)"
        )
        print(
            f"  Task 2.2 | Logistic GD: L0={losses_main[0]:.4f} -> L1000={losses_main[1000]:.4f}, "
            f"test_bce={test_bce:.4f}"
        )
        print(
            f"           | +100-Epoch Post-Convergence Audit (Epochs 1000..1100): "
            f"L1000={losses_ext[1000]:.6f} -> L1100={losses_ext[1100]:.6f} "
            f"(max weight delta={max_weight_drift_100ep:.2e})"
        )
        print(
            f"  Task 2.3 | Epoch-200 Losses: eta=1.0 ({lr_losses[1.0]:.4f}) < "
            f"eta=0.1 ({lr_losses[0.1]:.4f}) < eta=0.01 ({lr_losses[0.01]:.4f})"
        )
        print(
            f"  Task 2.5 | Test @ tau=0.5: F1={eval_05['f1']:.4f}, Recall={eval_05['recall']:.4f} | "
            f"Test @ tau=0.3: F1={eval_03['f1']:.4f}, Recall={eval_03['recall']:.4f}"
        )
        print(
            f"  Task 2.7 | 5-Fold CV Best K={knn_best_k} -> Test F1={knn_eval['f1']:.4f}, "
            f"Recall={knn_eval['recall']:.4f}"
        )
        print(
            f"  Task 2.8 | Optimal cost_tau={best_cost_tau:.4f} (Train Cost=INR {best_train_cost}) -> "
            f"Test Cost=INR {cost_test_at_tau} (vs INR {cost_test_at_05} @ 0.5, INR {cost_test_flag_nobody} blind)"
        )
        print("=" * 76)

    return dict(results_items)


if __name__ == "__main__":
    run_part2(verbose=True)
