import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import chi2_contingency, fisher_exact, binomtest
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint
import math
from statistics import NormalDist
from typing import Tuple, Optional


def paired_recall_binomial_test(y_true, y_pred_A, y_pred_B):
    y_true = np.asarray(y_true)
    y_pred_A = np.asarray(y_pred_A)
    y_pred_B = np.asarray(y_pred_B)

    pos = y_true == 1

    A = y_pred_A[pos]
    B = y_pred_B[pos]

    A_better = np.sum((A == 1) & (B == 0))
    B_better = np.sum((A == 0) & (B == 1))

    n_discordant = A_better + B_better

    if n_discordant == 0:
        p_value = 1.0
    else:
        p_value = binomtest(
            A_better,
            n_discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue

    return {
        "p_value": p_value,
        "A_better": A_better,
        "B_better": B_better,
        "n_discordant": n_discordant,
    }




def cohens_h(p1, p2):
    """
    Positive if p1 > p2, negative if p1 < p2.
    """
    p1 = np.clip(p1, 1e-10, 1 - 1e-10)
    p2 = np.clip(p2, 1e-10, 1 - 1e-10)
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))


def make_ipw_weights(y_true, N_neg_total: int):
    """
    Inverse-probability weights for the 'all positives + sampled negatives'
    Pos: weight 1 (as all included)
    Neg: weight = N_neg_total / n_neg_sample
    """
    y_true = np.asarray(y_true)
    n_neg_sample = int(np.sum(y_true == 0))
    if n_neg_sample == 0:
        raise ValueError("No sampled negatives found in y_true (y_true==0).")

    w_neg = float(N_neg_total) / float(n_neg_sample)
    w = np.ones_like(y_true, dtype=float)
    w[y_true == 0] = w_neg
    return w, w_neg


def bootstrap_metric_ci(
        y_true,
        y_pred_A,
        y_pred_B,
        metric_func,
        n_bootstrap=10000,
        random_state=None,
        sample_weight=None,
):
    """
    Paired bootstrap to get CI for metric difference (A - B).
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_pred_A = np.asarray(y_pred_A)
    y_pred_B = np.asarray(y_pred_B)
    n = len(y_true)

    metric_A = metric_func(y_true, y_pred_A, sample_weight=sample_weight)
    metric_B = metric_func(y_true, y_pred_B, sample_weight=sample_weight)
    diff_obs = metric_A - metric_B
    h_obs = cohens_h(metric_A, metric_B)

    diffs = np.empty(n_bootstrap)
    idx = np.arange(n)
    hs = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        sample_idx = rng.choice(idx, size=n, replace=True)
        sw = None if sample_weight is None else sample_weight[sample_idx]
        mA = metric_func(y_true[sample_idx], y_pred_A[sample_idx], sample_weight=sw)
        mB = metric_func(y_true[sample_idx], y_pred_B[sample_idx], sample_weight=sw)

        diffs[i] = mA - mB
        hs[i] = cohens_h(mA, mB)

    ci_low = np.percentile(diffs, 2.5)
    ci_high = np.percentile(diffs, 97.5)

    h_ci_low = np.percentile(hs, 2.5)
    h_ci_high = np.percentile(hs, 97.5)

    return {
        "metric_A": metric_A,
        "metric_B": metric_B,
        "diff": diff_obs,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "effect_size_h": h_obs,
        "effect_size_h_ci_low": h_ci_low,
        "effect_size_h_ci_high": h_ci_high,
    }


def permutation_test_metric(y_true, y_pred_A, y_pred_B, metric_func, n_perm=10000,
                            random_state=None, sample_weight=None):
    """
    Paired permutation test for difference in a metric between two models.
    Returns a two-sided p-value for H0: metric_A == metric_B.
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    A = np.asarray(y_pred_A)
    B = np.asarray(y_pred_B)
    n = len(y_true)

    diff_obs = metric_func(y_true, A, sample_weight=sample_weight) - metric_func(y_true, B, sample_weight=sample_weight)

    diffs = np.empty(n_perm)
    for i in range(n_perm):
        swap = rng.integers(0, 2, size=n).astype(bool)
        A_swapped = np.where(swap, B, A)
        B_swapped = np.where(swap, A, B)
        diffs[i] = (
                metric_func(y_true, A_swapped, sample_weight=sample_weight)
                - metric_func(y_true, B_swapped, sample_weight=sample_weight)
        )

    p_value = np.mean(np.abs(diffs) >= abs(diff_obs))

    return {
        "diff": diff_obs,
        "p_value": p_value,
    }


def test_2x2(correct_a, total_a, correct_b, total_b):
    tab = np.array([
        [correct_a, total_a - correct_a],
        [correct_b, total_b - correct_b]
    ])
    chi2, p_chi2, dof, expected = chi2_contingency(tab)
    if (expected < 5).any():
        _, p = fisher_exact(tab, alternative="two-sided")
        method = "fisher_exact"
        stat = np.nan
    else:
        p = p_chi2
        method = "chi2"
        stat = chi2
    return stat, p, method, tab, expected


def pairwise_segment_tests(df: pd.DataFrame, correction="holm") -> pd.DataFrame:
    segs = df.index.tolist()
    pairs = list(combinations(segs, 2))

    rows = []
    raw_p = []
    for a, b in pairs:
        da = df.loc[df.index == a].iloc[0]
        db = df.loc[df.index == b].iloc[0]
        stat, p, method, tab, expected = test_2x2(
            int(da["Correct"]), int(da["Support"]),
            int(db["Correct"]), int(db["Support"])
        )
        rows.append({
            "seg_A": a,
            "seg_B": b,
            "recall_A": da["Recall"],
            "recall_B": db["Recall"],
            "delta_recall": da["Recall"] - db["Recall"],
            "effect_h": cohens_h(da["Recall"], db["Recall"]),  # NEW
            "stat": stat,
            "test_method": method,
            "p_raw": p,
            "table_2x2": tab
        })
        raw_p.append(p)
    # Multiple-testing correction
    rej, p_adj, _, _ = multipletests(raw_p, method=correction)
    for r, pA, R in zip(rows, p_adj, rej):
        r["p_adj"] = pA
        r["significant"] = bool(R)

    out = pd.DataFrame(rows).sort_values("p_adj")
    return out


def holm_bonferroni(p_values):
    items = sorted(p_values.items(), key=lambda kv: kv[1])  # sort by p
    m = len(items)
    adjusted = {}
    prev_adj = 0.0

    for i, (name, p) in enumerate(items, start=1):
        adj = (m - i + 1) * p
        adj = max(adj, prev_adj)
        adj = min(adj, 1.0)
        adjusted[name] = adj
        prev_adj = adj

    return adjusted


def benjamini_hochberg(p_values):
    items = sorted(p_values.items(), key=lambda kv: kv[1])  # sort by p
    m = len(items)
    adjusted = {}
    prev_adj = 1.0

    for i, (name, p) in reversed(list(enumerate(items, start=1))):
        adj = p * m / i
        adj = min(adj, prev_adj)
        adj = min(adj, 1.0)
        adjusted[name] = adj
        prev_adj = adj

    return adjusted
