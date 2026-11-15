Domain: Recruitment

# Apna Recruitment Decision Intelligence — End-to-End Quantitative Pipeline

This repository implements the complete four-part **Apna Recruitment Decision Intelligence** analytical and machine-learning pipeline from first principles using only the Python standard library and NumPy. Every statistical estimator, ordinary least squares and normal-equation solver, full-batch gradient descent optimizer, logistic regression classifier, Euclidean K-nearest-neighbours model with 5-fold cross-validation, K-means clustering algorithm, agglomerative hierarchical linkage engine, 7-step Principal Component Analysis (PCA) eigen-decomposition, decision-tree impurity calculator, AdaBoost weight-update engine, and neural architecture parameter sizer is implemented directly in transparent, reproducible code.

## How to run

Run the following commands in order from the root of a clean clone on the `main` branch:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python generate_data.py
.venv/bin/python part1.py
.venv/bin/python part2.py
.venv/bin/python part3.py
.venv/bin/python part4.py
.venv/bin/python verify_all.py
```

If using an active Python 3.10+ environment where `requirements.txt` is already installed:

```bash
pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
python verify_all.py
```

Running `generate_data.py` writes `data/records.csv` (`800 10 170 0537bc7fb4a044719831c43166e79d3c`). Running `part1.py` through `part4.py` regenerates all 12 graded files in `outputs/`:
- `outputs/part1_results.txt`
- `outputs/part2_results.txt`, `outputs/part2_thresholds.csv`, `outputs/part2_knn_cv.csv`
- `outputs/part3_results.txt`, `outputs/part3_segments.csv`, `outputs/part3_hierarchical.csv`
- `outputs/part4_results.txt`, `outputs/part4_splits.csv`, `outputs/part4_voting.csv`, `outputs/part4_adaboost.csv`, `outputs/part4_architecture.csv`

## Part 1 findings

- **R1.1 — Hypothesis test on average days to hire (`H0: mu = 8.0` vs `H1: mu != 8.0` at `alpha = 0.05`):**
  - Across all `n = 800` records, the sample mean of `days_to_hire` is `mean_value = 8.9382` days with sample standard deviation `sd_value = 9.1723` days, standard error `se_value = 0.3243` days, and 95% confidence interval `[ci95_low = 8.3026, ci95_high = 9.5738]`.
  - **Decision:** `H0` is **rejected** (`reject_h0 = yes`) because the test statistic `z_stat = 2.8930` exceeds the two-tailed normal critical value `1.96` in magnitude (`|2.8930| > 1.96`).
  - **Type I error in domain terms:** A Type I error here means concluding that Apna's true average time from application to an offer decision differs from the 8.0-day planning norm when the true population average is actually still 8.0 days, causing talent operations to unnecessarily revise hiring timelines and recruiter staffing plans.
  - **Practical size of the gap:** The observed gap `mean_value - 8.0` is `0.9382` days (about 22.5 hours). While `0.9382` days is modest for a single requisition relative to the wide candidate-to-candidate spread (`sd_value = 9.1723` days), across thousands of frontline hires an extra day per candidate compounds into measurable onboarding delays and shift-fill slippage, so the planning team should update its baseline from `8.0` days toward `8.9382` days.

- **R1.2 — Sample size requirement for half-day precision (`E = 0.5` days at 95% confidence):**
  - Achieving a 95% confidence margin of error of at most `E = 0.5` days requires `n_required = 1293` records, compared with the `n = 800` rows currently available (whose current 95% margin of error is `1.96 * 0.3243 = 0.6356` days).
  - **Implication for the planning team:** The current dataset of `800` records is `493` records short of the sample size needed to guarantee half-day precision, so the team must either collect at least `493` additional hiring records before locking in a `+/- 0.5`-day SLA commitment or accept the wider `+/- 0.6356`-day uncertainty band (`8.3026` to `9.5738` days).

- **R1.3 — Closed-form OLS vs full-batch gradient descent convergence:**
  - Closed-form simple regression of `days_to_hire` on `X4` (`expected_salary_inr_thousands`) yields `reg_slope = 0.6701` and `reg_intercept = -3.6598` (`tss = 67221.4285`, `ess = 34925.6088`, `rss = 32295.8197`, `tss_equals_ess_plus_rss = yes`, `r2 = 0.5196`, `rmse = 6.3537`, `mae = 3.7636`). Full-batch gradient descent (`eta = 0.1`, 500 updates on standardised `X4`) reduces cost from `gd_J0 = 163.9178` (`gd_J1 = 119.4579`, `gd_J10 = 41.7973`, `gd_J100 = 40.3698`) to `gd_J500 = 40.3698`, recovering `gd_slope = 0.6701`, `gd_intercept = -3.6598`, and a slope discrepancy of `gd_slope_gap = 0.0000`.
  - **Why gradient descent reached the closed-form line:** Because mean-squared-error loss for linear regression on standardised features is a strictly convex quadratic bowl with a unique global minimum and well-conditioned curvature, 500 updates at `eta = 0.1` contract the initial parameter distance by `(1 - 0.2)^500` to machine precision.

- **R1.4 — Heteroskedasticity funnel diagnostic and log-log transformation remedy:**
  - In the raw linear regression on `X4`, sorting the 800 records by fitted values reveals a strong residual funnel: the residual sample standard deviation grows from `resid_sd_lower_raw = 2.3155` days in the lower 400 fitted rows to `resid_sd_upper_raw = 8.6891` days in the upper 400 fitted rows, giving a ratio of `resid_sd_ratio_raw = 3.7525`.
  - Regressing `ln(days_to_hire)` on `ln(X4)` (`loglog_intercept = -2.4370`, `loglog_slope = 1.5030`, `loglog_r2 = 0.6565`, `loglog_rmse = 0.5268`) yields lower-half residual SD `resid_sd_lower_log = 0.5165` and upper-half residual SD `resid_sd_upper_log = 0.5369`, reducing the spread ratio to `resid_sd_ratio_log = 1.0396` and confirming that the log-log specification removes the multiplicative funnel.

- **R1.5 — Multiple regression (`ln(value)` on constant, `ln(X4)`, `X1`, `X2`, `X3`, `X5`, `X6`, `X7`):**
  - The normal-equation estimates are `mr_beta_0 = -1.8613` (constant), `mr_beta_1 = 1.1576` (`ln(X4)`), `mr_beta_2 = -0.0074` (`X1`), `mr_beta_3 = 0.0940` (`X2`), `mr_beta_4 = 0.1067` (`X3`), `mr_beta_5 = 0.0028` (`X5`), `mr_beta_6 = 0.1518` (`X6`), and `mr_beta_7 = 0.0172` (`X7`).
  - Applying the mechanical threshold `|coefficient| > 0.05` (excluding the intercept `mr_beta_0`) selects four predictors: **`ln(X4)`** (`expected_salary_inr_thousands`, `mr_beta_1 = 1.1576`), **`X2`** (`interview_reschedules_90d`, `mr_beta_3 = 0.0940`), **`X3`** (`salary_gap_ratio`, `mr_beta_4 = 0.1067`), and **`X6`** (`competing_offer_flag`, `mr_beta_6 = 0.1518`), while `X1` (`mr_beta_2 = -0.0074`), `X5` (`mr_beta_5 = 0.0028`), and `X7` (`mr_beta_7 = 0.0172`) fall below `0.05`. Because these predictors are measured on different raw scales (log salary, months, reschedule counts, 0-to-1 ratio, 0-to-100 score, and binary flags), this `0.05` rule is purely a mechanical cutoff rather than a ranking of predictor importance.
  - **Do the extra predictors earn their place?** Yes: adjusted R-squared rises from `loglog_adj_r2 = 0.6561` in the single-predictor log-log model to `mr_adj_r2 = 0.7403` (`mr_r2 = 0.7425`) in the seven-predictor model, a net gain of `0.0842` after penalising for the six additional parameters.

## Part 2 findings

- **R2.1 — Learning-rate comparison after 200 gradient-descent updates:**
  - Ordered from lowest to highest training binary cross-entropy loss at epoch 200: `loss_epoch_200_eta_1.0 = 0.3703` (`eta = 1.0`), `loss_epoch_200_eta_0.1 = 0.3776` (`eta = 0.1`), and `loss_epoch_200_eta_0.01 = 0.5091` (`eta = 0.01`).
  - This ordering shows that a learning rate that is too small (`eta = 0.01`) progresses sluggishly and remains far from convergence after 200 updates (`0.5091`), whereas a large learning rate (`eta = 1.0`) descends rapidly to the optimum (`0.3703`) on this smooth, standardised logistic loss surface without diverging.

- **R2.2 — Top three predictors by absolute logistic weight and largest odds ratio:**
  - Across the 1000-epoch logistic model (`loss_epoch_0 = 0.6931`, `loss_epoch_1 = 0.6753`, `loss_epoch_10 = 0.5685`, `loss_epoch_100 = 0.3952`, `loss_epoch_1000 = 0.3704`, `bias = -2.0696`, `test_bce = 0.3818`), the three predictors with the largest absolute standardised weights are **`X3` (`salary_gap_ratio`, `weight_X3 = 0.9013`)**, **`X1` (`experience_months`, `weight_X1 = -0.5554`)**, and **`X2` (`interview_reschedules_90d`, `weight_X2 = 0.5505`)**, followed by `X6` (`weight_X6 = 0.2348`), `X5` (`weight_X5 = -0.2162`), `X4` (`weight_X4 = -0.1459`), and `X7` (`weight_X7 = -0.0120`).
  - **Domain interpretation of the largest odds ratio (`odds_ratio_X3 = 2.4629`):** A one-training-standard-deviation increase in a candidate's `salary_gap_ratio` (`X3`) multiplies the odds that the candidate declines the offer by `2.4629` (a 146.29% increase in decline odds), holding all other six candidate features constant.

- **R2.3 — Why accuracy alone is weak, and the impact of lowering `tau` from `0.5` to `0.3`:**
  - On the `test_rows = 240` test set (`test_positives = 51` declined offers, `189` accepted offers), a blind rule that predicts `0` for every candidate achieves `blind_accuracy = 0.7875`, while the logistic model at `tau = 0.5` achieves `tau_0.5_accuracy = 0.8042`. Accuracy alone is a weak metric here because class imbalance (`21.25%` positive rate) allows a useless classifier that catches zero declined offers to score `0.7875`, masking the fact that `tau = 0.5` still misses `tau_0.5_fn = 27` of the `51` declined offers (`tau_0.5_tp = 24`, `tau_0.5_fp = 20`, `tau_0.5_tn = 169`, `tau_0.5_specificity = 0.8942`).
  - Lowering the decision threshold from `tau = 0.5` to `tau = 0.3` increased **recall** from `tau_0.5_recall = 0.4706` to `tau_0.3_recall = 0.7451` (`tau_0.3_tp = 38`, `tau_0.3_fn = 13`) while reducing **precision** from `tau_0.5_precision = 0.5455` to `tau_0.3_precision = 0.4634` (`tau_0.3_fp = 44`, `tau_0.3_tn = 145`, `tau_0.3_accuracy = 0.7625`, `tau_0.3_specificity = 0.7672`), raising the test F1 score from `tau_0.5_f1 = 0.5053` to `tau_0.3_f1 = 0.5714`.

- **R2.4 — Logistic regression (`tau = 0.3`) vs K-nearest neighbours (`knn_best_k = 5`):**
  - Five-fold cross-validation on the `train_rows = 560` training set (`train_positives = 119`) selected `knn_best_k = 5` (`mean_cv_f1 = 0.4541`). On the 240 test rows, the logistic model at `tau = 0.3` achieves `tau_0.3_f1 = 0.5714` and `tau_0.3_recall = 0.7451`, decisively beating the `K = 5` KNN classifier, which achieves only `knn_f1 = 0.3696` and `knn_recall = 0.3333` (`knn_precision = 0.4146`, `knn_accuracy = 0.7583`, `knn_tp = 17`, `knn_fp = 24`, `knn_tn = 165`, `knn_fn = 34`).

- **R2.5 — Deployment recommendation and business-cost threshold selection:**
  - We recommend deploying the **logistic regression classifier with alert threshold `cost_tau = 0.1500`**, which reduces total test-cohort cost to `cost_test_at_tau = 6050` (`INR 6050`), compared with `cost_test_at_0.5 = 11800` (`INR 11800`) at the default `0.5` threshold and `cost_test_flag_nobody = 20400` (`INR 20400`) when no candidates are flagged (with training cost `cost_train_at_tau = 12150`, or `INR 12150`).
  - The threshold `cost_tau = 0.1500` was chosen on the 560 training rows rather than the 240 test rows so that test labels never influence hyperparameter tuning, preventing data leakage and ensuring `cost_test_at_tau = 6050` (`INR 6050`) is an unbiased out-of-sample cost estimate.

## Part 3 findings

- **R3.1 — Successive K-means WCSS drops and elbow choice:**
  - From the pinned K-means runs (`wcss_k2 = 2262.6753`, `wcss_k3 = 1602.8471`, `wcss_k4 = 1494.7732`, `wcss_k5 = 1323.7157`, `wcss_k6 = 1278.1691`), the four successive WCSS drops are:
    - `WCSS(2) - WCSS(3) = 659.8282` (`2262.6753 - 1602.8471`)
    - `WCSS(3) - WCSS(4) = 108.0739` (`1602.8471 - 1494.7732`)
    - `WCSS(4) - WCSS(5) = 171.0575` (`1494.7732 - 1323.7157`)
    - `WCSS(5) - WCSS(6) = 45.5466` (`1323.7157 - 1278.1691`)
  - The elbow criterion selects `elbow_k = 3` because the drop after `K = 3` (`108.0739`) is well below half of the drop before `K = 3` (`0.5 * 659.8282 = 329.9141`). The successive drops do not shrink monotonically (`171.0575 > 108.0739`) because the pinned starting centroids produce a deterministic, reproducible local optimum for each `K` rather than the global minimum across all possible starting configurations.

- **R3.2 — Profiling the `K = 3` candidate pipeline segments (`outputs/part3_segments.csv`):**
  - **Segment A — Low-Friction Pay-Aligned Pipeline (`size_segment_A = 322`):** Characterised by high experience (`mean_X1 = 47.5028` months), minimal interview rescheduling (`mean_X2 = 1.9503`), small pay gap (`mean_X3 = 0.2250`), moderate pay expectation (`mean_X4 = 12.5301` thousand INR), high profile completeness (`mean_X5 = 62.1584`), the lowest offer-decline rate (`outcome_rate = 0.0186`), and fast decisions (`mean_value = 3.5976` days). **Concrete action:** Route candidates through automated fast-track offer generation without manual recruiter intervention unless individual risk exceeds `cost_tau = 0.1500`.
  - **Segment B — High-Reschedule Extended-Cycle Pipeline (`size_segment_B = 255`):** Characterised by mid-range experience (`mean_X1 = 28.6071` months), high interview reschedules (`mean_X2 = 6.0706`), moderate salary gap (`mean_X3 = 0.4534`), high salary expectation (`mean_X4 = 28.7192` thousand INR), moderate completeness (`mean_X5 = 55.0082`), moderate decline rate (`outcome_rate = 0.2353`), and slow turnaround (`mean_value = 17.1248` days). **Concrete action:** Compress scheduling by offering evening/weekend interview slots and pre-validating salary bands early to cut the `17.1248`-day decision cycle.
  - **Segment C — Wide-Pay-Gap High-Decline Pipeline (`size_segment_C = 223`):** Characterised by shorter tenure (`mean_X1 = 14.3892` months), elevated rescheduling (`mean_X2 = 4.2287`), wide salary gap (`mean_X3 = 0.7118`), `mean_X4 = 16.5099` thousand INR, lower profile completeness (`mean_X5 = 38.5215`), and `mean_value = 7.2883` days. **Segment C has the highest offer-decline rate at `outcome_rate = 0.4664`.** **Concrete action:** Mandate a recruiter compensation-alignment check prior to offer release and immediate follow-up outreach within 24 hours of extending the offer.

- **R3.3 — Hierarchical clustering cuts (`single` vs `complete` linkage):**
  - Cutting each dendrogram at its largest consecutive merge-height gap yields `single_clusters_at_cut = 4` clusters for **single linkage** — `{E1}`, `{E2, E3, E6}` (`E2+E3+E6`), `{E4}`, and `{E5}` (cut between merge 2 at `1.4000` and merge 3 at `2.8000`, gap `1.4000`) — versus `complete_clusters_at_cut = 2` clusters for **complete linkage** (and `average_clusters_at_cut = 2` for average linkage) — `{E1, E4, E5}` (`E1+E4+E5`) and `{E2, E3, E6}` (`E2+E3+E6`) (cut between merge 4 at `2.9000` and merge 5 at `7.9000`, gap `5.0000`).
  - **Why the two rules disagree:** Single linkage defines inter-cluster distance by the nearest pair across groups (`d(E5, E6) = 3.4000`), which pulls the final merge down to height `3.4000` (a gap of only `0.5000` above merge 4 at `2.9000`) so the earlier jump of `1.4000` after merge 2 wins the cut. Complete linkage defines inter-cluster distance by the farthest pair across groups (`d(E1, E2) = 7.9000`), which pushes the final merge height to `7.9000` and creates a dominant gap of `5.0000` after merge 4, cleanly isolating the two three-cohort macro-clusters.

- **R3.4 — Principal Component Analysis (PCA) loadings, variance retention, and segment alignment:**
  - In Step 1 of PCA on the standardised `800 x 5` matrix, the column mean vector is all zeros (`[0.0000, 0.0000, 0.0000, 0.0000, 0.0000]`). Across the five eigenvalues (`eigenvalue_1 = 2.6136`, `eigenvalue_2 = 1.1828`, `eigenvalue_3 = 0.5114`, `eigenvalue_4 = 0.4055`, `eigenvalue_5 = 0.2867`, summing to `eigenvalue_sum = 5.0000`), **PC1** (`explained_ratio_1 = 0.5227`) is driven strongly by **`X1` (`pc1_loading_X1 = 0.5317`)**, **`X3` (`pc1_loading_X3 = -0.5187`)**, **`X5` (`pc1_loading_X5 = 0.4418`)**, and **`X2` (`pc1_loading_X2 = -0.4100`)**, with a smaller contribution from `X4` (`pc1_loading_X4 = -0.2916`), while PC2 (`explained_ratio_2 = 0.2366`) loads primarily on `X4` (`pc2_loading_X4 = 0.6934`) and `X2` (`pc2_loading_X2 = 0.5056`).
  - Keeping at least 80% of the total variance requires **`components_for_80pct = 3` principal components** (`cumulative_ratio_1 = 0.5227`, `cumulative_ratio_2 = 0.7593`, `cumulative_ratio_3 = 0.8616`, `cumulative_ratio_4 = 0.9427`, `cumulative_ratio_5 = 1.0000`).
  - The mean PC1 scores across the three K-means segments — `mean_pc1_segment_A = 1.7773`, `mean_pc1_segment_B = -0.7831`, and `mean_pc1_segment_C = -1.6709` (with first three row projections `row1_pc1 = 1.9008, row1_pc2 = -0.8537`, `row2_pc1 = -0.3505, row2_pc2 = 1.9241`, `row3_pc1 = -1.4382, row3_pc2 = 1.4312`) — demonstrate that PC1 captures the primary pipeline gradient separating high-experience, low-gap Segment A (`+1.7773`) from high-friction Segment B (`-0.7831`) and wide-pay-gap, high-decline Segment C (`-1.6709`).

## Part 4 findings

- **R4.1 — Decision-tree split purity, overfitting in unpruned trees, and stopping rule:**
  - From the 560 training rows (`root_entropy = 0.7462`, `root_gini = 0.3347`), **`S4` (`X1 <= 24`)** is the purest split by both criteria (`best_split_by_info_gain = S4` with `info_gain = 0.1252`, and `best_split_by_gini_gain = S4` with `gini_gain = 0.0606`), ahead of `S3` (`info_gain = 0.1046`, `gini_gain = 0.0494`), `S1` (`info_gain = 0.0110`, `gini_gain = 0.0052`), and `S2` (`info_gain = 0.0033`, `gini_gain = 0.0015`), because it partitions candidates into a high-decline group (`n_yes = 188`, `positives_yes = 86`, `stump_rate_yes = 0.4574`) and a large low-decline group (`n_no = 372`, `positives_no = 33`, `stump_rate_no = 0.0887`).
  - Growing a tree until every leaf is completely pure overfits the training data because late splits isolate tiny, idiosyncratic subsets of records and fit sampling noise rather than structural relationships.
  - One standard rule that stops a tree from growing too deep is capping the **maximum tree depth (`max_depth`)** (or enforcing a minimum leaf size `min_samples_leaf` / minimum impurity decrease).

- **R4.2 — Comparison of the five Task 4.2 classifiers (`outputs/part4_voting.csv`):**
  - Evaluating all five classifiers on the 240 test rows at threshold `0.3`, we would ship the standalone **`logistic`** model, which achieves the highest F1 score (`f1 = 0.5714`, `recall = 0.7451`, `precision = 0.4634`, `tp = 38`, `fp = 44`, `tn = 145`, `fn = 13`).
  - Combining the models via voting did not improve over standalone `logistic`: because `knn` has weak sensitivity (`f1 = 0.3696`, `recall = 0.3333`, `precision = 0.4146`) and the single-feature `stump` has lower precision (`f1 = 0.5532`, `recall = 0.7647`, `precision = 0.4333`), **`hard_vote`** drops to `f1 = 0.5344` and `recall = 0.6863` (`precision = 0.4375`) and **`soft_vote`** reaches `f1 = 0.5507` and `recall = 0.7451` (`precision = 0.4368`), both below `logistic` (`f1 = 0.5714`).

- **R4.3 — Two-round AdaBoost weight dynamics (`outputs/part4_adaboost.csv`):**
  - After Round 1 (`round1_error = 0.2500`, `round1_alpha = 0.5493`), each of the two points misclassified by `h1` (Point 3 and Point 7) has normalised weight `weight_after = 0.2500` (up from `0.1250`), so their **total combined weight is `0.5000`** (`0.2500 + 0.2500`).
  - Round 2 trusts `h2` (`round2_alpha = 0.8047`) more than Round 1 trusts `h1` (`round1_alpha = 0.5493`) because `h2` correctly classifies both of the heavily up-weighted Round-1 error points (Points 3 and 7) and only misclassifies two down-weighted points (Points 1 and 5, each of weight `0.0833`), lowering the weighted error to `round2_error = 0.1667` (`ensemble_accuracy = 0.7500`).

- **R4.4 — Neural network sizing, sequence sharing, activations, and skip connections:**
  - **Model A Option 2 (CNN on `28 x 28 x 1` input, `cnn_total = 15734` vs MLP `mlp_total = 13002`; scaling check `fc_layer_params = 50332672` vs `conv_layer_params = 1792`):**

| Layer | Configuration | Output Shape (H x W x D) | Parameters |
|---|---|---|---|
| Input | Grayscale digit image | `28 x 28 x 1` | `0` |
| `cnn_conv1` | 6 filters of `5 x 5 x 1`, stride 1, pad 2 + ReLU | `28 x 28 x 6` | `156` |
| MaxPool 1 | `2 x 2` pooling, stride 2 | `14 x 14 x 6` | `0` |
| `cnn_conv2` | 16 filters of `5 x 5 x 6`, stride 1, pad 0 + ReLU | `10 x 10 x 16` | `2416` |
| MaxPool 2 | `2 x 2` pooling, stride 2 | `5 x 5 x 16` | `0` |
| Flatten | Reshape `5 x 5 x 16` to vector | `1 x 1 x 400` (`400`) | `0` |
| `cnn_dense1` | Fully connected, 32 neurons + ReLU | `1 x 1 x 32` (`32`) | `12832` |
| `cnn_dense2` | Fully connected output, 10 classes | `1 x 1 x 10` (`10`) | `330` |
| **Total (`cnn_total`)** | **All trainable CNN parameters** | — | **`15734`** |

  - **Model B (RNN for 16 resume job families, `rnn_total = 26768`):** The weight matrix shapes are **`W_xh` = `128 x 64`** (`8192` parameters), **`W_hh` = `128 x 128`** (`16384` parameters, plus bias `b_h` of shape `128 x 1` with `128` parameters), and **`W_hy` = `16 x 128`** (`2048` parameters, plus bias `b_y` of shape `16 x 1` with `16` parameters). `rnn_total` (`26768`) does not grow when input sequences get longer because the recurrent cell reuses the exact same shared transition matrices (`W_xh`, `W_hh`, and `b_h`) at every time step regardless of sequence length `T`.
  - **Activations and loss for 10-class classification:** We choose **ReLU** for the hidden layers and **Softmax** activation paired with **multiclass cross-entropy loss** for the 10-class output layer.
  - **Why a very deep plain network stops learning in early layers and how skip connections help:** In a very deep plain network, backpropagation repeatedly multiplies gradients by layer Jacobians smaller than one, causing signals to vanish exponentially before reaching early layers. A residual skip connection (`H(x) = F(x) + x`) creates an identity highway with unit gradient that passes error signals directly to earlier layers without attenuation.

## Model-choice memo

**To:** Head of Talent-Acquisition Operations, Apna
**From:** Talent Analytics & Decision Intelligence Team
**Subject:** Production Deployment Recommendation for Offer-Decline Risk Scoring

We recommend deploying the **standardised logistic regression model (`logistic`)** from Task 4.2 into production to trigger proactive recruiter follow-ups before candidate offers lapse. On the 240 held-out test records (`51` declined offers) evaluated at threshold `0.3000`, `logistic` achieved the highest F1 score (`0.5714`), with precision `0.4634`, recall `0.7451`, and test binary cross-entropy `0.3818`, outperforming the decision stump (`f1 = 0.5532`), soft voting (`f1 = 0.5507`), hard voting (`f1 = 0.5344`), and 5-nearest neighbours (`f1 = 0.3696`, `recall = 0.3333`).

Because an unflagged declined offer (`INR 400`) is eight times costlier than an unnecessary recruiter follow-up (`INR 50`), we recommend operating the logistic model at the training-optimised alert threshold `cost_tau = 0.1500` (`cost_train_at_tau = 12150`, or `INR 12150`). On the 240 test records, `cost_tau = 0.1500` reduces total expected cost to `INR 6050` (`cost_test_at_tau = 6050`), saving `INR 5750` compared with the standard `0.5000` threshold (`cost_test_at_0.5 = 11800`, or `INR 11800`) and saving `INR 14350` (a 70.34% reduction) compared with flagging nobody (`cost_test_flag_nobody = 20400`, or `INR 20400`).

A key limitation of this evidence is that the model was trained and evaluated on a single synthetic `560`/`240` split containing only `51` test positives under static `INR 50` and `INR 400` cost assumptions, so live calibration drift across job markets or seasonal hiring waves could shift the optimal threshold. Neither Model A (`cnn_total = 15734` parameters for scanned form digits) nor Model B (`rnn_total = 26768` parameters for resume job-family sorting) is worth building before the structured logistic model is in production, because our 8-parameter tabular classifier delivers immediate `INR 14350` test-cohort savings from fields already captured in the applicant tracking system.

## Limitations

1. **Synthetic Data Generation vs Live Operational Complexity:** Every record in `data/records.csv` (`n = 800`) comes from a parametric three-group mixture generator (`generate_data.py`). Real frontline recruitment pipelines contain missing fields, measurement noise, seasonal labour-supply shocks, role-specific pay ceilings, and recruiter-level effects that are absent here.
2. **Single Stratified Train/Test Split and Small Positive Test Count:** All supervised evaluations in Part 2 and Part 4 rely on a single deterministic `560`/`240` split with only `test_positives = 51` declined offers in the test set. With `51` positive test cases, flipping a single borderline prediction changes recall by nearly `0.0200` (`1 / 51`) and shifts test cost by `INR 400`, introducing sampling variance around `cost_tau = 0.1500` and `cost_test_at_tau = 6050`.
3. **Static Linear Cost Function and Uniform Follow-Up Effectiveness Assumption:** The threshold rule `cost = 50 * fp + 400 * fn` treats every false alarm (`INR 50`) and every missed decline (`INR 400`) as homogeneous across junior and senior roles (`X4` ranging from entry-level to `28.7192` thousand INR in Segment B) and implicitly assumes that flagging a candidate allows a recruiter to prevent the decline with constant effectiveness.
4. **Sensitivity of K-Means to Pinned Starting Centroids:** As shown in Task 3.1 (`wcss_k3 = 1602.8471`, `wcss_k4 = 1494.7732`, `wcss_k5 = 1323.7157`), deterministic single-pass centroid initialisation can settle into local optima (`WCSS(4) - WCSS(5) = 171.0575 > WCSS(3) - WCSS(4) = 108.0739`), whereas production segmentation should evaluate multiple random restarts (such as k-means++) and cluster stability indices.
5. **Unmodelled Temporal and Causal Feedback Loops:** Both the log-linear regression for `days_to_hire` (`mr_adj_r2 = 0.7403`) and the logistic model for `offer_declined` (`test_bce = 0.3818`) estimate observational associations rather than causal treatment effects; intervening on candidates via recruiter calls or pay adjustments will shift the joint distribution of features and outcomes over time, requiring periodic recalibration.
