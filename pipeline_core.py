"""Shared statistical, data-loading, preprocessing, and model primitives
for the Apna Recruitment Decision Intelligence Capstone Pipeline.

All mathematical and machine learning routines are implemented from first
principles using only the Python standard library and NumPy array operations,
strictly following the pinned conventions of the specification.
"""
from __future__ import annotations

import csv
import math
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

DATA_PATH = os.path.join("data", "records.csv")
OUTPUTS_DIR = "outputs"

PREDICTOR_NAMES = [
    "experience_months",              # X1
    "interview_reschedules_90d",      # X2
    "salary_gap_ratio",               # X3
    "expected_salary_inr_thousands",  # X4
    "profile_completeness_score",     # X5
    "competing_offer_flag",           # X6
    "referred_flag",                  # X7
]

PREDICTOR_CODES = ["X1", "X2", "X3", "X4", "X5", "X6", "X7"]


def load_records(filepath: str = DATA_PATH) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """Load data/records.csv in exact file order.

    Returns:
        record_ids: List of string identifiers (R0001 .. R0800)
        X: float64 array of shape (800, 7) corresponding to X1 .. X7
        outcome: int64 array of shape (800,) corresponding to offer_declined (0 or 1)
        value: float64 array of shape (800,) corresponding to days_to_hire
    """
    record_ids: List[str] = []
    x_rows: List[List[float]] = []
    outcomes: List[int] = []
    values: List[float] = []

    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        expected_header = [
            "record_id",
            *PREDICTOR_NAMES,
            "offer_declined",
            "days_to_hire",
        ]
        if header != expected_header:
            raise ValueError(f"Unexpected CSV header: {header}")
        for row in reader:
            record_ids.append(row[0])
            x_rows.append([float(v) for v in row[1:8]])
            outcomes.append(int(row[8]))
            values.append(float(row[9]))

    return (
        record_ids,
        np.array(x_rows, dtype=np.float64),
        np.array(outcomes, dtype=np.int64),
        np.array(values, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# First-Principles Sample Statistics (divisor n - 1 everywhere)
# ---------------------------------------------------------------------------

def sample_mean_1d(x: np.ndarray) -> float:
    """Compute arithmetic mean of a 1D array from first principles."""
    return float(np.sum(x) / len(x))


def sample_var_1d(x: np.ndarray) -> float:
    """Compute unbiased sample variance (divisor n - 1) of a 1D array."""
    mu = sample_mean_1d(x)
    diff = x - mu
    return float(np.sum(diff * diff) / (len(x) - 1))


def sample_sd_1d(x: np.ndarray) -> float:
    """Compute sample standard deviation (divisor n - 1) of a 1D array."""
    return math.sqrt(sample_var_1d(x))


def sample_cov_1d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute sample covariance (divisor n - 1) between two 1D arrays."""
    mx = sample_mean_1d(x)
    my = sample_mean_1d(y)
    return float(np.sum((x - mx) * (y - my)) / (len(x) - 1))


def sample_mean_2d(X: np.ndarray) -> np.ndarray:
    """Column-wise arithmetic mean of a 2D array (n, p)."""
    return np.sum(X, axis=0) / X.shape[0]


def sample_sd_2d(X: np.ndarray, mean_vec: np.ndarray | None = None) -> np.ndarray:
    """Column-wise sample standard deviation (divisor n - 1) of a 2D array (n, p)."""
    if mean_vec is None:
        mean_vec = sample_mean_2d(X)
    diff = X - mean_vec
    var_vec = np.sum(diff * diff, axis=0) / (X.shape[0] - 1)
    return np.sqrt(var_vec)


def standardise_fit_transform(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit column-wise mean and sample SD on X, and return (Z, mean_vec, sd_vec)."""
    mean_vec = sample_mean_2d(X)
    sd_vec = sample_sd_2d(X, mean_vec)
    Z = (X - mean_vec) / sd_vec
    return Z, mean_vec, sd_vec


def standardise_transform(X: np.ndarray, mean_vec: np.ndarray, sd_vec: np.ndarray) -> np.ndarray:
    """Standardise X using pre-computed (training) mean_vec and sd_vec."""
    return (X - mean_vec) / sd_vec


# ---------------------------------------------------------------------------
# Stratified Split & Standardisation (Task 2.1, reused in Part 4)
# ---------------------------------------------------------------------------

def stratified_split_and_standardise(
    X: np.ndarray, outcome: np.ndarray
) -> Dict[str, Any]:
    """Perform the pinned 70/30 stratified split preserving file order.

    Take rows with outcome 0 in file order; the first floor(7 * count / 10)
    go to training and the rest to test; do the same for outcome 1.
    Both training and test sets preserve original file order.
    Standardise all 7 columns using training mean and training sample SD.
    """
    idx_neg = [i for i, y_val in enumerate(outcome) if y_val == 0]
    idx_pos = [i for i, y_val in enumerate(outcome) if y_val == 1]

    n_train_neg = (7 * len(idx_neg)) // 10
    n_train_pos = (7 * len(idx_pos)) // 10

    train_indices = sorted(idx_neg[:n_train_neg] + idx_pos[:n_train_pos])
    test_indices = sorted(idx_neg[n_train_neg:] + idx_pos[n_train_pos:])

    X_train_raw = X[train_indices]
    y_train = outcome[train_indices]
    X_test_raw = X[test_indices]
    y_test = outcome[test_indices]

    Z_train, train_mean, train_sd = standardise_fit_transform(X_train_raw)
    Z_test = standardise_transform(X_test_raw, train_mean, train_sd)

    return {
        "train_indices": train_indices,
        "test_indices": test_indices,
        "X_train_raw": X_train_raw,
        "y_train": y_train,
        "X_test_raw": X_test_raw,
        "y_test": y_test,
        "Z_train": Z_train,
        "Z_test": Z_test,
        "train_mean": train_mean,
        "train_sd": train_sd,
    }


# ---------------------------------------------------------------------------
# Logistic Regression from Scratch (Task 2.2, reused in Part 4)
# ---------------------------------------------------------------------------

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic sigmoid 1 / (1 + exp(-z))."""
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Average binary cross-entropy loss: -(1/n) sum [y ln(p) + (1-y) ln(1-p)]."""
    n = len(y_true)
    loss = -np.sum(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)) / n
    return float(loss)


def fit_logistic_regression(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    eta: float = 0.1,
    epochs: int = 1000,
    record_epochs: Sequence[int] = (0, 1, 10, 100, 1000),
) -> Tuple[np.ndarray, float, Dict[int, float]]:
    """Train logistic regression via full-batch gradient descent from zero initialization."""
    n, p = Z_train.shape
    w = np.zeros(p, dtype=np.float64)
    b = 0.0
    recorded_losses: Dict[int, float] = {}
    record_set = set(record_epochs)

    if 0 in record_set:
        p0 = sigmoid(Z_train @ w + b)
        recorded_losses[0] = binary_cross_entropy(y_train, p0)

    for epoch in range(1, epochs + 1):
        y_hat = sigmoid(Z_train @ w + b)
        err = y_hat - y_train
        grad_w = (Z_train.T @ err) / n
        grad_b = float(np.sum(err) / n)
        w = w - eta * grad_w
        b = b - eta * grad_b

        if epoch in record_set:
            y_hat_post = sigmoid(Z_train @ w + b)
            recorded_losses[epoch] = binary_cross_entropy(y_train, y_hat_post)

    return w, b, recorded_losses


# ---------------------------------------------------------------------------
# Confusion Matrix & Classification Metrics (Conventions, Part 2 & Part 4)
# ---------------------------------------------------------------------------

def confusion_and_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute confusion matrix counts and classification metrics from first principles."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "tpr": float(recall),
        "specificity": float(specificity),
        "fpr": float(fpr),
        "f1": float(f1),
    }


# ---------------------------------------------------------------------------
# K-Nearest Neighbours from Scratch (Task 2.7, reused in Part 4)
# ---------------------------------------------------------------------------

def knn_predict_probs_and_labels(
    Z_ref: np.ndarray,
    y_ref: np.ndarray,
    Z_query: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict positive neighbour share and binary labels for Z_query using K-NN.

    Distances are Euclidean in standardised space. Ties in distance are broken
    by position in file order among reference rows via kind='stable'.
    Query is predicted positive when at least (K + 1) / 2 of the K neighbours are positive.
    """
    n_query = Z_query.shape[0]
    probs = np.zeros(n_query, dtype=np.float64)
    preds = np.zeros(n_query, dtype=np.int64)
    threshold_count = (k + 1) // 2

    for i in range(n_query):
        diff = Z_ref - Z_query[i]
        dists = np.sqrt(np.sum(diff * diff, axis=1))
        nearest_idx = np.argsort(dists, kind="stable")[:k]
        pos_count = int(np.sum(y_ref[nearest_idx]))
        probs[i] = pos_count / k
        preds[i] = 1 if pos_count >= threshold_count else 0

    return probs, preds


def select_best_k_knn_cv(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    k_values: Sequence[int] = tuple(range(1, 22, 2)),
) -> Tuple[int, List[Tuple[int, float]]]:
    """Run 5-fold cross-validation on the 560 training rows (fold = index % 5)."""
    n_train = Z_train.shape[0]
    folds = np.arange(n_train) % 5
    cv_rows: List[Tuple[int, float]] = []
    best_k = k_values[0]
    best_f1 = -1.0

    for k in k_values:
        fold_f1s: List[float] = []
        for f in range(5):
            val_mask = folds == f
            ref_mask = ~val_mask
            _, y_pred_fold = knn_predict_probs_and_labels(
                Z_train[ref_mask], y_train[ref_mask], Z_train[val_mask], k
            )
            metrics = confusion_and_metrics(y_train[val_mask], y_pred_fold)
            fold_f1s.append(metrics["f1"])
        mean_f1 = float(sum(fold_f1s) / len(fold_f1s))
        cv_rows.append((k, mean_f1))
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_k = k

    return best_k, cv_rows


# ---------------------------------------------------------------------------
# Output File Writers (Strict Formatting Rules)
# ---------------------------------------------------------------------------

def format_value(val: Any) -> str:
    """Format a scalar according to specification:
    - bool -> 'yes' or 'no'
    - int / np.integer -> integer string
    - float / np.floating -> 4 decimal places
    - str -> as given
    """
    if isinstance(val, bool) or isinstance(val, np.bool_):
        return "yes" if bool(val) else "no"
    if isinstance(val, (int, np.integer)):
        return f"{int(val)}"
    if isinstance(val, (float, np.floating)):
        return f"{float(val):.4f}"
    return str(val)


def write_results_txt(filepath: str, items: Sequence[Tuple[str, Any]]) -> None:
    """Write key = value pairs to outputs/partN_results.txt."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8", newline="\n") as f:
        for key, val in items:
            f.write(f"{key} = {format_value(val)}\n")


def write_csv_file(
    filepath: str, header: Sequence[str], rows: Sequence[Sequence[Any]]
) -> None:
    """Write a CSV file with exact header and 4-decimal float formatting."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow([format_value(cell) for cell in row])
