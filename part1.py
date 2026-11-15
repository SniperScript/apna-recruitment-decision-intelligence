"""Part 1 — Estimation, Inference, and Regression Baseline (25 marks).

Computes all confidence intervals, hypothesis tests, sample-size calculations,
closed-form and gradient-descent simple linear regressions, heteroskedasticity
funnel diagnostics with log-log remedy, and multiple linear regression via the
normal equation on all 800 candidate records.

Also executes a +100-epoch post-convergence trajectory audit (epochs 500..600)
in console diagnostics without altering the pinned 500-epoch graded outputs.

Writes:
    outputs/part1_results.txt
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from pipeline_core import (
    OUTPUTS_DIR,
    load_records,
    sample_cov_1d,
    sample_mean_1d,
    sample_sd_1d,
    sample_var_1d,
    write_results_txt,
)


def closed_form_simple_regression(
    x: np.ndarray, y: np.ndarray
) -> Dict[str, Any]:
    """Fit y = intercept + slope * x from first principles (divisor n - 1)."""
    n = len(y)
    mx = sample_mean_1d(x)
    my = sample_mean_1d(y)
    vx = sample_var_1d(x)
    cov_xy = sample_cov_1d(x, y)

    slope = cov_xy / vx
    intercept = my - slope * mx
    fitted = intercept + slope * x
    residuals = y - fitted

    tss = float(np.sum((y - my) ** 2))
    ess = float(np.sum((fitted - my) ** 2))
    rss = float(np.sum(residuals ** 2))
    tss_identity = bool(abs(tss - ess - rss) < 1e-6 * tss)
    r2 = 1.0 - (rss / tss)
    rmse = math.sqrt(rss / n)
    mae = float(np.sum(np.abs(residuals)) / n)

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "fitted": fitted,
        "residuals": residuals,
        "tss": tss,
        "ess": ess,
        "rss": rss,
        "tss_equals_ess_plus_rss": tss_identity,
        "r2": float(r2),
        "rmse": float(rmse),
        "mae": mae,
    }


def funnel_residual_diagnostics(
    fitted: np.ndarray, residuals: np.ndarray
) -> Tuple[float, float, float]:
    """Sort rows by fitted value ascending (ties broken by row index via stable sort)
    and compute sample SD of residuals in the lower 400 and upper 400 halves.
    """
    order = np.argsort(fitted, kind="stable")
    half = len(fitted) // 2
    lower_res = residuals[order[:half]]
    upper_res = residuals[order[half:]]

    sd_lower = sample_sd_1d(lower_res)
    sd_upper = sample_sd_1d(upper_res)
    ratio = sd_upper / sd_lower
    return sd_lower, sd_upper, ratio


def run_part1(verbose: bool = True) -> Dict[str, Any]:
    """Execute Tasks 1.1 through 1.7 and write outputs/part1_results.txt."""
    _, X, _, value = load_records()
    n = len(value)
    x4 = X[:, 3]  # expected_salary_inr_thousands

    # ------------------------------------------------------------------
    # Task 1.1: Estimate with an interval
    # ------------------------------------------------------------------
    mean_value = sample_mean_1d(value)
    sd_value = sample_sd_1d(value)
    se_value = sd_value / math.sqrt(n)
    ci95_low = mean_value - 1.96 * se_value
    ci95_high = mean_value + 1.96 * se_value

    # ------------------------------------------------------------------
    # Task 1.2: Test a planning claim (H0: mu = 8.0 vs H1: mu != 8.0)
    # ------------------------------------------------------------------
    z_stat = (mean_value - 8.0) / se_value
    reject_h0 = abs(z_stat) > 1.96

    # ------------------------------------------------------------------
    # Task 1.3: Size a future sample (E = 0.5 days, 95% confidence)
    # ------------------------------------------------------------------
    margin_e = 0.5
    n_required = int(math.ceil(((1.96 * sd_value) / margin_e) ** 2))

    # ------------------------------------------------------------------
    # Task 1.4: Simple regression by the closed form (value on X4)
    # ------------------------------------------------------------------
    ols_raw = closed_form_simple_regression(x4, value)

    # ------------------------------------------------------------------
    # Task 1.5: Fit the same line by gradient descent (500 pinned updates)
    # ------------------------------------------------------------------
    mean_x4 = sample_mean_1d(x4)
    sd_x4 = sample_sd_1d(x4)
    z4 = (x4 - mean_x4) / sd_x4

    m = 0.0
    c = 0.0
    eta = 0.1
    gd_costs: Dict[int, float] = {}

    def cost_j(m_val: float, c_val: float) -> float:
        err = value - (m_val * z4 + c_val)
        return float(np.sum(err * err) / n)

    gd_costs[0] = cost_j(m, c)
    for step in range(1, 501):
        err = value - (m * z4 + c)
        grad_m = -2.0 * float(np.sum(z4 * err)) / n
        grad_c = -2.0 * float(np.sum(err)) / n
        m = m - eta * grad_m
        c = c - eta * grad_c
        if step in (1, 10, 100, 500):
            gd_costs[step] = cost_j(m, c)

    gd_slope = m / sd_x4
    gd_intercept = c - (m * mean_x4) / sd_x4
    gd_slope_gap = abs(gd_slope - ols_raw["slope"])

    # +100-epoch post-convergence diagnostic trace (epochs 501..600)
    m_ext, c_ext = m, c
    ext_trace: List[Tuple[int, float, float]] = []
    for ext_step in range(501, 601):
        err_ext = value - (m_ext * z4 + c_ext)
        m_ext -= eta * (-2.0 * float(np.sum(z4 * err_ext)) / n)
        c_ext -= eta * (-2.0 * float(np.sum(err_ext)) / n)
        if ext_step % 25 == 0:
            slope_ext = m_ext / sd_x4
            ext_trace.append(
                (ext_step, cost_j(m_ext, c_ext), abs(slope_ext - ols_raw["slope"]))
            )

    # ------------------------------------------------------------------
    # Task 1.6: Check for a funnel, then apply the log remedy
    # ------------------------------------------------------------------
    resid_sd_lower_raw, resid_sd_upper_raw, resid_sd_ratio_raw = (
        funnel_residual_diagnostics(ols_raw["fitted"], ols_raw["residuals"])
    )

    ln_x4 = np.log(x4)
    ln_value = np.log(value)
    ols_log = closed_form_simple_regression(ln_x4, ln_value)

    resid_sd_lower_log, resid_sd_upper_log, resid_sd_ratio_log = (
        funnel_residual_diagnostics(ols_log["fitted"], ols_log["residuals"])
    )

    # ------------------------------------------------------------------
    # Task 1.7: Multiple regression by the normal equation
    # Order: [1, ln(X4), X1, X2, X3, X5, X6, X7]
    # ------------------------------------------------------------------
    X_design = np.column_stack(
        [
            np.ones(n, dtype=np.float64),
            ln_x4,
            X[:, 0],  # X1
            X[:, 1],  # X2
            X[:, 2],  # X3
            X[:, 4],  # X5
            X[:, 5],  # X6
            X[:, 6],  # X7
        ]
    )
    xtx = X_design.T @ X_design
    xty = X_design.T @ ln_value
    mr_betas = np.linalg.solve(xtx, xty)

    mr_fitted = X_design @ mr_betas
    mr_rss = float(np.sum((ln_value - mr_fitted) ** 2))
    mr_tss = ols_log["tss"]
    mr_r2 = 1.0 - (mr_rss / mr_tss)

    p_mr = 7
    mr_adj_r2 = 1.0 - (1.0 - mr_r2) * (n - 1) / (n - p_mr - 1)

    p_loglog = 1
    loglog_adj_r2 = 1.0 - (1.0 - ols_log["r2"]) * (n - 1) / (n - p_loglog - 1)

    results_items: List[Tuple[str, Any]] = [
        # Task 1.1
        ("n", n),
        ("mean_value", mean_value),
        ("sd_value", sd_value),
        ("se_value", se_value),
        ("ci95_low", ci95_low),
        ("ci95_high", ci95_high),
        # Task 1.2
        ("z_stat", z_stat),
        ("reject_h0", reject_h0),
        # Task 1.3
        ("n_required", n_required),
        # Task 1.4
        ("reg_slope", ols_raw["slope"]),
        ("reg_intercept", ols_raw["intercept"]),
        ("tss", ols_raw["tss"]),
        ("ess", ols_raw["ess"]),
        ("rss", ols_raw["rss"]),
        ("tss_equals_ess_plus_rss", ols_raw["tss_equals_ess_plus_rss"]),
        ("r2", ols_raw["r2"]),
        ("rmse", ols_raw["rmse"]),
        ("mae", ols_raw["mae"]),
        # Task 1.5
        ("gd_J0", gd_costs[0]),
        ("gd_J1", gd_costs[1]),
        ("gd_J10", gd_costs[10]),
        ("gd_J100", gd_costs[100]),
        ("gd_J500", gd_costs[500]),
        ("gd_slope", gd_slope),
        ("gd_intercept", gd_intercept),
        ("gd_slope_gap", gd_slope_gap),
        # Task 1.6
        ("resid_sd_lower_raw", resid_sd_lower_raw),
        ("resid_sd_upper_raw", resid_sd_upper_raw),
        ("resid_sd_ratio_raw", resid_sd_ratio_raw),
        ("loglog_intercept", ols_log["intercept"]),
        ("loglog_slope", ols_log["slope"]),
        ("loglog_r2", ols_log["r2"]),
        ("loglog_rmse", ols_log["rmse"]),
        ("resid_sd_lower_log", resid_sd_lower_log),
        ("resid_sd_upper_log", resid_sd_upper_log),
        ("resid_sd_ratio_log", resid_sd_ratio_log),
        # Task 1.7
        ("mr_beta_0", float(mr_betas[0])),
        ("mr_beta_1", float(mr_betas[1])),
        ("mr_beta_2", float(mr_betas[2])),
        ("mr_beta_3", float(mr_betas[3])),
        ("mr_beta_4", float(mr_betas[4])),
        ("mr_beta_5", float(mr_betas[5])),
        ("mr_beta_6", float(mr_betas[6])),
        ("mr_beta_7", float(mr_betas[7])),
        ("mr_r2", mr_r2),
        ("mr_adj_r2", mr_adj_r2),
        ("loglog_adj_r2", loglog_adj_r2),
    ]

    out_path = os.path.join(OUTPUTS_DIR, "part1_results.txt")
    write_results_txt(out_path, results_items)

    if verbose:
        print("=" * 76)
        print("PART 1 — ESTIMATION, INFERENCE & REGRESSION BASELINE (800 RECORDS)")
        print("=" * 76)
        print(
            f"  Task 1.1 | n={n}, mean={mean_value:.4f}, sd={sd_value:.4f}, "
            f"se={se_value:.4f}, 95% CI=[{ci95_low:.4f}, {ci95_high:.4f}]"
        )
        print(
            f"  Task 1.2 | H0: mu=8.0 -> z_stat={z_stat:.4f}, reject_h0={'yes' if reject_h0 else 'no'}"
        )
        print(
            f"  Task 1.3 | Margin E=0.5 days -> n_required={n_required} (vs {n} available)"
        )
        print(
            f"  Task 1.4 | Raw OLS: slope={ols_raw['slope']:.4f}, intercept={ols_raw['intercept']:.4f}, "
            f"R2={ols_raw['r2']:.4f}, RMSE={ols_raw['rmse']:.4f}, MAE={ols_raw['mae']:.4f}"
        )
        print(
            f"  Task 1.5 | GD (500 epochs): J0={gd_costs[0]:.4f} -> J500={gd_costs[500]:.4f}, "
            f"gd_slope_gap={gd_slope_gap:.2e}"
        )
        print(
            "           | +100-Epoch Post-Convergence Audit (Epochs 501..600): "
            + ", ".join(
                f"E{ep}: J={j_val:.4f} (gap={gap:.2e})"
                for ep, j_val, gap in ext_trace
            )
        )
        print(
            f"  Task 1.6 | Funnel SD Ratio: Raw={resid_sd_ratio_raw:.4f} "
            f"({resid_sd_upper_raw:.4f}/{resid_sd_lower_raw:.4f}) -> "
            f"Log-Log={resid_sd_ratio_log:.4f} ({resid_sd_upper_log:.4f}/{resid_sd_lower_log:.4f})"
        )
        print(
            f"  Task 1.7 | Multiple Regression: R2={mr_r2:.4f}, Adj R2={mr_adj_r2:.4f} "
            f"(vs Log-Log Adj R2={loglog_adj_r2:.4f})"
        )
        print("=" * 76)

    return dict(results_items)


if __name__ == "__main__":
    run_part1(verbose=True)
