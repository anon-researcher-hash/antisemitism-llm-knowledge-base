import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from utils.stats_helpers import cohens_h, make_ipw_weights, bootstrap_metric_ci, permutation_test_metric, \
    holm_bonferroni, benjamini_hochberg, paired_recall_binomial_test, pairwise_segment_tests



def precision_binary(y_true, y_pred, sample_weight=None):
    return precision_score(y_true, y_pred, pos_label=1, zero_division=0, sample_weight=sample_weight)


def recall_binary(y_true, y_pred, sample_weight=None):
    return recall_score(y_true, y_pred, pos_label=1, zero_division=0, sample_weight=sample_weight)


def f1_binary(y_true, y_pred, sample_weight=None):
    return f1_score(y_true, y_pred, pos_label=1, zero_division=0, sample_weight=sample_weight)


def compare_models_vs_baseline(
        y_true,
        model_preds,
        baseline_name,
        metric_func,
        n_bootstrap=10000,
        n_perm=10000,
        random_state=None,
        correction="holm",
        N_neg_total=None,
        sample_weight=None,
        recall_is_exact=True,
):
    """
    model_preds: dict {model_name: y_pred_array}
    baseline_name: name of the baseline model in model_preds
    metric_func: e.g. precision_binary, recall_binary, f1_binary
    """
    baseline_pred = np.asarray(model_preds[baseline_name])
    w_neg = None

    if sample_weight is None and N_neg_total is not None:
        sample_weight, w_neg = make_ipw_weights(y_true, N_neg_total)

    results = {}
    p_values = {}

    for name, y_pred in model_preds.items():
        if name == baseline_name:
            continue

        if recall_is_exact and (metric_func.__name__ == "recall_binary"):
            pos_mask = (np.asarray(y_true) == 1)
            metric_A = metric_func(y_true[pos_mask], y_pred[pos_mask], sample_weight=None)
            metric_B = metric_func(y_true[pos_mask], baseline_pred[pos_mask], sample_weight=None)

            ci_res = {
                "metric_A": metric_A,
                "metric_B": metric_B,
                "diff": metric_A - metric_B,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "effect_size_h": cohens_h(metric_A, metric_B),
                "effect_size_h_ci_low": np.nan,
                "effect_size_h_ci_high": np.nan,
            }
        else:

            # Bootstrap for CI
            ci_res = bootstrap_metric_ci(
                y_true,
                y_pred_A=y_pred,
                y_pred_B=baseline_pred,
                metric_func=metric_func,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
                sample_weight=sample_weight,  # NEW
            )

        # Permutation test for p-value
        if metric_func.__name__ == "recall_binary":
            perm_res = paired_recall_binomial_test(
                y_true,
                y_pred_A=y_pred,
                y_pred_B=baseline_pred,
            )
        else:
            perm_res = permutation_test_metric(
                y_true,
                y_pred_A=y_pred,
                y_pred_B=baseline_pred,
                metric_func=metric_func,
                n_perm=n_perm,
                random_state=random_state,
                sample_weight=sample_weight,
            )

        # Merge results
        res = {
            **ci_res,
            "p_value": perm_res["p_value"],
        }
        results[name] = res
        p_values[name] = res["p_value"]

    # Correction
    if correction is not None and len(p_values) > 0:
        if correction.lower() == "holm":
            p_adj = holm_bonferroni(p_values)
        elif correction.lower() in ["bh", "fdr"]:
            p_adj = benjamini_hochberg(p_values)
        else:
            raise ValueError("Unknown correction method")
    else:
        p_adj = {k: v for k, v in p_values.items()}

    for name in results:
        results[name]["p_value_adj"] = p_adj[name]

    return results


def wrap_model_comparison(df, N_neg_total, baseline_name="no_kb", with_lex=False):
    # 'N_neg_total' is the full number of negative samples in original data if inverse weighting; if class balance then same as the number of positiv samples
    # y_true: positives (all) + the SAME sample of 1000 negatives for all models
    y_true = np.array(df["label"].values, dtype=int)

    # model_preds contains predictions for each model on the same examples
    model_preds = {
        "no_kb": np.array(df["classification_no_kb_cleaned"].map(lambda x: 1 if x == "Yes" else 0).values, dtype=int),
        "ihra": np.array(df["classification_ihra_explanation_cleaned"].map(lambda x: 1 if x == "Yes" else 0).values,
                         dtype=int),
        "tax": np.array(df["classification_tax"].map(lambda x: 1 if x == "Yes" else 0).values, dtype=int),
        "tax_ex": np.array(df["classification_tax_ex"].map(lambda x: 1 if x == "Yes" else 0).values, dtype=int),
    }
    if with_lex:
        model_preds["lex"] = np.array(df["classification_lexicon"].map(lambda x: 1 if x == "Yes" else 0).values, dtype=int)
    res_all = []
    for metric_name, metric_func in [
        ("precision", precision_binary),
        ("recall", recall_binary),
        ("f1", f1_binary),
    ]:
        res = compare_models_vs_baseline(
            y_true,
            model_preds,
            baseline_name=baseline_name,
            metric_func=metric_func,
            n_bootstrap=10000,
            random_state=42,
            correction="holm",  # or "bh" or None
            N_neg_total=N_neg_total  # total number of negatives
        )
        res_all.append((metric_name, res))
    return res_all


def make_ipw_weights_by_dataset(y_true, dataset_id, N_neg_total_by_dataset):
    y_true = np.asarray(y_true)
    dataset_id = np.asarray(dataset_id)

    weights = np.ones(len(y_true), dtype=float)

    for ds, N_neg_total in N_neg_total_by_dataset.items():
        ds_mask = dataset_id == ds
        neg_mask = ds_mask & (y_true == 0)

        n_neg_sampled = neg_mask.sum()
        if n_neg_sampled == 0:
            raise ValueError(f"No sampled negatives for dataset {ds}")

        weights[neg_mask] = N_neg_total / n_neg_sampled

    return weights


def wrap_model_comparison_union(df, N_neg_total_by_dataset, baseline_name="no_kb", with_lex=False):
    y_true = np.array(df["label"].values, dtype=int)
    dataset_id = np.array(df["dataset_id"].values)

    sample_weight = make_ipw_weights_by_dataset(
        y_true,
        dataset_id,
        N_neg_total_by_dataset,
    )

    model_preds = {
        "no_kb": np.array(df["classification_no_kb_cleaned"].map(lambda x: 1 if x == "Yes" else 0), dtype=int),
        "ihra": np.array(df["classification_ihra_explanation_cleaned"].map(lambda x: 1 if x == "Yes" else 0),
                         dtype=int),
        "tax": np.array(df["classification_tax"].map(lambda x: 1 if x == "Yes" else 0), dtype=int),
        "tax_ex": np.array(df["classification_tax_ex"].map(lambda x: 1 if x == "Yes" else 0), dtype=int),
    }
    if with_lex:
        model_preds["lex"] = np.array(df["classification_lexicon"].map(lambda x: 1 if x == "Yes" else 0), dtype=int)

    res_all = []
    for metric_name, metric_func in [
        ("precision", precision_binary),
        ("recall", recall_binary),
        ("f1", f1_binary),
    ]:
        res = compare_models_vs_baseline(
            y_true,
            model_preds,
            baseline_name=baseline_name,
            metric_func=metric_func,
            n_bootstrap=10000,
            random_state=42,
            correction="holm",
            sample_weight=sample_weight,
            recall_is_exact=True,
        )
        res_all.append((metric_name, res))
    return res_all


def print_model_comparison_results(res_all, baseline_name="no_kb"):
    for res in res_all:
        print(f"\n===== {res[0].upper()} vs {baseline_name} =====")
        for name, r in res[1].items():
            print(f"\nModel: {name}")
            print(f"metric({name}): {r['metric_A']:.4f}")
            print(f"metric({baseline_name}): {r['metric_B']:.4f}")
            print(f"Diff (model - baseline): {r['diff']:.4f}")
            print(f"95% CI: [{r['ci_low']:.4f}, {r['ci_high']:.4f}]")
            print(f"p-value (raw): {r['p_value']:.4g}")
            print(f"p-value (adjusted): {r['p_value_adj']:.4g}")
            print(
                f"Effect size h: {r['effect_size_h']:.4f} (95% CI: [{r['effect_size_h_ci_low']:.4f}, {r['effect_size_h_ci_high']:.4f}])")


def combine_model_comparison_outputs(
        res_all_by_dataset,
        dataset_weights=None,
        correction="holm",
):
    """
    Combine outputs from wrap_model_comparison across datasets.
    """

    dataset_names = list(res_all_by_dataset.keys())

    if dataset_weights is None:
        dataset_weights = {
            ds: 1.0 / len(dataset_names)
            for ds in dataset_names
        }
    else:
        total_w = sum(dataset_weights.values())
        dataset_weights = {
            ds: w / total_w
            for ds, w in dataset_weights.items()
        }

    # Convert each dataset's result list to dict by metric name
    res_by_dataset_metric = {
        ds: dict(res_all)
        for ds, res_all in res_all_by_dataset.items()
    }

    metric_names = list(next(iter(res_by_dataset_metric.values())).keys())

    combined_res_all = []

    for metric_name in metric_names:
        combined_metric_res = {}

        model_names = list(
            next(iter(res_by_dataset_metric.values()))[metric_name].keys()
        )

        p_values = {}

        for model_name in model_names:
            combined = {}

            for key in [
                "metric_A",
                "metric_B",
                "diff",
                "effect_size_h",
            ]:
                combined[key] = sum(
                    dataset_weights[ds]
                    * res_by_dataset_metric[ds][metric_name][model_name][key]
                    for ds in dataset_names
                )

            combined["ci_low"] = np.nan
            combined["ci_high"] = np.nan
            combined["effect_size_h_ci_low"] = np.nan
            combined["effect_size_h_ci_high"] = np.nan

            combined["p_value"] = max(
                res_by_dataset_metric[ds][metric_name][model_name]["p_value"]
                for ds in dataset_names
            )

            combined_metric_res[model_name] = combined
            p_values[model_name] = combined["p_value"]

        if correction is not None and len(p_values) > 0:
            if correction.lower() == "holm":
                p_adj = holm_bonferroni(p_values)
            elif correction.lower() in ["bh", "fdr"]:
                p_adj = benjamini_hochberg(p_values)
            else:
                raise ValueError("Unknown correction method")
        else:
            p_adj = dict(p_values)

        for model_name in combined_metric_res:
            combined_metric_res[model_name]["p_value_adj"] = p_adj[model_name]

        combined_res_all.append((metric_name, combined_metric_res))

    return combined_res_all


def extract_metric_values(res_all, setting_name, baseline_name="no_kb"):
    rows = []

    for metric_name, metric_res in res_all:

        first_model = next(iter(metric_res.values()))
        rows.append({
            "setting": setting_name,
            "metric": metric_name,
            "model": baseline_name,
            "value": float(first_model["metric_B"]),
        })

        for model_name, r in metric_res.items():
            rows.append({
                "setting": setting_name,
                "metric": metric_name,
                "model": model_name,
                "value": float(r["metric_A"]),
            })

    return pd.DataFrame(rows)


# Analysis NB 2

def calculate_recall(df, classification_column, value='Yes', split_by=None, two_annotators=False):
    """Calculate recall by taking into account that we have two annotations per instance."""
    if split_by:
        split_by_values = df[split_by].unique()
        true_positives = {k: ((df[split_by] == k) & (df[classification_column] == value)).sum() for k in split_by_values}
        support = {k: (df[split_by] == k).sum() for k in split_by_values}
        recall = {k: true_positives[k] / support[k] if support[k] > 0 else 0 for k in split_by_values}
        correct = {k: ((df[split_by] == k) & (df[classification_column] == value)).sum() for k in split_by_values}
        if two_annotators:
            support = {k: int(support[k] / 2) for k in support}
            correct = {k: int(correct[k] / 2) for k in correct}
    else:
        true_positives = (df[classification_column] == value).sum()
        support = len(df)
        recall = 100*true_positives / support
        correct = (df[classification_column] == value).sum()
        if two_annotators:
            support = int(support / 2)
            correct = int(correct / 2)
    return recall, support, correct



def summarize_recall_stats(recall, support, correct):
    summary = pd.DataFrame({
        'Recall': pd.Series(recall),
        'Support': pd.Series(support),
        'Correct': pd.Series(correct)
    })
    summary['Recall'] = summary['Recall'].map(lambda x: np.round(x, 2))
    summary['Missed'] = summary['Support'] - summary['Correct']
    return summary


def group_dfs_by_row_index_pivot(dfs):
    """
    Reorganize {dataset_name: DataFrame} into
    {keyword: DataFrame indexed by dataset_name}.
    """

    long = []
    for name, df in dfs.items():
        temp = df.copy()
        temp["dataset"] = name
        temp["keyword"] = temp.index
        long.append(temp)

    long = pd.concat(long, ignore_index=True)

    # Pivot: rows = dataset, columns = metrics, one table per keyword
    result = {}
    for keyword, sub in long.groupby("keyword"):
        tbl = sub.pivot_table(
            index="dataset",
            values=["Recall", "Support", "Correct", "Missed"],
            aggfunc="first"   # each dataset appears exactly once per keyword
        )
        result[keyword] = tbl

    return result

def explode_df(df, col):
    df_expanded = df.explode(col).reset_index(drop=True)
    return df_expanded


def bootstrap_balanced_metrics(
    y_true_B, y_pred_B, sample_weight_B,
    y_true_D, y_pred_D, sample_weight_D,
    metric_func,
    n_bootstrap=10000,
    random_state=None,
):
    rng = np.random.default_rng(random_state)

    y_true_B = np.asarray(y_true_B)
    y_pred_B = np.asarray(y_pred_B)
    sw_B = np.asarray(sample_weight_B)

    y_true_D = np.asarray(y_true_D)
    y_pred_D = np.asarray(y_pred_D)
    sw_D = np.asarray(sample_weight_D)

    n_B = len(y_true_B)
    n_D = len(y_true_D)

    boot_vals = []

    for _ in range(n_bootstrap):
        idx_B = rng.integers(0, n_B, size=n_B)
        idx_D = rng.integers(0, n_D, size=n_D)

        m_B = metric_func(
            y_true_B[idx_B],
            y_pred_B[idx_B],
            sample_weight=sw_B[idx_B],
        )

        m_D = metric_func(
            y_true_D[idx_D],
            y_pred_D[idx_D],
            sample_weight=sw_D[idx_D],
        )

        # equal dataset weighting
        boot_vals.append(0.5 * m_B + 0.5 * m_D)

    boot_vals = np.asarray(boot_vals)

    m_B = metric_func(y_true_B, y_pred_B, sample_weight=sw_B)
    m_D = metric_func(y_true_D, y_pred_D, sample_weight=sw_D)

    point = 0.5 * m_B + 0.5 * m_D

    ci_low, ci_high = np.percentile(boot_vals, [2.5, 97.5])

    return {
        "value": point,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }

def compute_balanced_metrics(
    df_B,
    df_D,
    N_neg_total_B,
    N_neg_total_D,
    n_bootstrap=10000,
    random_state=42,
):
    y_true_B = df_B["label"].values.astype(int)
    y_true_D = df_D["label"].values.astype(int)

    sw_B, _ = make_ipw_weights(y_true_B, N_neg_total_B)
    sw_D, _ = make_ipw_weights(y_true_D, N_neg_total_D)

    model_cols = {
        "no_kb": "classification_no_kb_cleaned",
        "ihra": "classification_ihra_explanation_cleaned",
        "tax": "classification_tax",
        "tax_ex": "classification_tax_ex",
    }

    results = []

    for model_name, col in model_cols.items():

        y_pred_B = (df_B[col] == "Yes").astype(int).values
        y_pred_D = (df_D[col] == "Yes").astype(int).values

        for metric_name, metric_func in [
            ("precision", precision_binary),
            ("recall", recall_binary),
            ("f1", f1_binary),
        ]:

            res = bootstrap_balanced_metrics(
                y_true_B, y_pred_B, sw_B,
                y_true_D, y_pred_D, sw_D,
                metric_func,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
            )

            results.append({
                "model": model_name,
                "metric": metric_name,
                "value": res["value"],
                "ci_low": res["ci_low"],
                "ci_high": res["ci_high"],
            })

    return pd.DataFrame(results)


def test_recall_per_content_group_compared_to_base(class_cols_to_recalls, comparison):
    """comparison: 0 for IHRA vs NO_KB, 3 for NO_KB vs. TAX, and 4 for NO_KB vs. TAX_EX"""
    grouped = group_dfs_by_row_index_pivot(class_cols_to_recalls)
    model_comparison_per_group = pd.DataFrame(columns=["seg_A", "seg_B", "p_raw", "p_adj", "significant", "effect_h"])

    index = []
    for i, k in enumerate(grouped.keys()):
        pairwise = pairwise_segment_tests(grouped[k], correction="holm")
        model_comparison_per_group.loc[i] = pairwise.loc[comparison]
        if comparison == 0:  # for IHRA the comparison with NO_KB is the other way around
            model_comparison_per_group.loc[i, "seg_A"] = "NO_KB"
            model_comparison_per_group.loc[i, "seg_B"] = "IHRA"
            model_comparison_per_group.loc[i, "effect_h"] = -model_comparison_per_group.loc[i, "effect_h"]
        index.append(k)
    model_comparison_per_group["content_group"] = index
    model_comparison_per_group.set_index("content_group", inplace=True)
    return model_comparison_per_group



def combine_pvalues_stouffer(pvals, weights=None):
    pvals = np.asarray(pvals)

    # avoid exact 0/1
    eps = 1e-15
    pvals = np.clip(pvals, eps, 1 - eps)

    if weights is None:
        weights = np.ones(len(pvals))

    weights = np.asarray(weights)

    z = norm.isf(pvals / 2)   # two-sided
    z_comb = np.sum(weights * z) / np.sqrt(np.sum(weights**2))

    return 2 * norm.sf(abs(z_comb))


def combine_dataset_tests(
    df_B,
    df_D,
    weights=(0.5, 0.5),
    correction="holm",
):
    # reset index because content_group currently index
    a = df_B.reset_index()
    b = df_D.reset_index()

    merged = a.merge(
        b,
        on=["content_group", "seg_A", "seg_B"],
        suffixes=("_B", "_D")
    )

    rows = []

    for _, r in merged.iterrows():

        p_comb = combine_pvalues_stouffer(
            [r["p_raw_B"], r["p_raw_D"]],
            weights=weights,
        )

        rows.append({
            "content_group": r["content_group"],
            "seg_A": r["seg_A"],
            "seg_B": r["seg_B"],

            "p_raw": p_comb,

            # equal dataset weighting
            "effect_h":
                weights[0] * r["effect_h_B"]
                + weights[1] * r["effect_h_D"],
        })

    out = pd.DataFrame(rows)

    # multiple testing correction AFTER combination
    rej, p_adj, _, _ = multipletests(
        out["p_raw"],
        method=correction,
    )

    out["p_adj"] = p_adj
    out["significant"] = rej

    return out.sort_values("p_adj")