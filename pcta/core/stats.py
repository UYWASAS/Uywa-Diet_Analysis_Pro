"""
Inferential statistics for PCTA (with safety rule).

Safety rule (critical):
- If replication per treatment < 2, DO NOT output p-values.
  => This module returns disabled results with a reason and warnings.

When replication exists (per metric after NA drop):
- Shapiro-Wilk normality (per group where n>=3)
- Levene homogeneity (median center; requires n>=2 per group)
- Choose omnibus:
    * ANOVA (normal + homoscedastic)
    * Welch ANOVA (normal + heteroscedastic)
    * Kruskal-Wallis (non-normal)
- Posthoc:
    * ANOVA -> Tukey HSD
    * Welch -> Games-Howell (approx; implemented via pairwise Welch t-tests + Holm adjust)
    * Kruskal -> Dunn-like (approx; implemented via pairwise Mann-Whitney + Holm adjust)

Effect sizes:
- ANOVA: eta^2
- Kruskal: epsilon^2

Implementation notes:
- Works on a DataFrame of unit KPIs (one row per unit).
- Missing values for a metric are dropped only for that metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from .schemas import AnalysisWarning, WarningCode


@dataclass(frozen=True)
class StatsOptions:
    alpha: float = 0.05
    enable_posthoc: bool = True


def _rep_by_treatment(sub: pd.DataFrame, treatment_col: str) -> Dict[str, int]:
    return sub.groupby(treatment_col).size().astype(int).to_dict()


def _min_n(rep: Dict[str, int]) -> int:
    return int(min(rep.values())) if rep else 0


def _shapiro_min_p(groups: List[np.ndarray], alpha: float) -> Tuple[Optional[float], bool, str]:
    """
    Returns (min_p, normal_bool, note).
    normal_bool is True if all available p-values > alpha and at least one test ran.
    """
    pvals: List[float] = []
    for g in groups:
        if g.size < 3:
            continue
        try:
            pvals.append(float(sps.shapiro(g).pvalue))
        except Exception:
            continue

    if not pvals:
        return None, False, "Shapiro skipped for groups with n<3 (or failed)."
    min_p = float(min(pvals))
    normal = all(p > alpha for p in pvals)
    return min_p, normal, "Shapiro run per group where n>=3; stored value is min p across groups."


def _levene_p(groups: List[np.ndarray]) -> Optional[float]:
    if len(groups) < 2:
        return None
    if any(g.size < 2 for g in groups):
        return None
    try:
        return float(sps.levene(*groups, center="median").pvalue)
    except Exception:
        return None


def _holm_adjust(pvals: List[float]) -> List[float]:
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    p_sorted = np.array([pvals[i] for i in order], dtype=float)
    adj_sorted = np.empty(m, dtype=float)

    running_max = 0.0
    for j, p in enumerate(p_sorted):
        adj = (m - j) * p
        running_max = max(running_max, float(adj))
        adj_sorted[j] = min(1.0, running_max)

    out = np.empty(m, dtype=float)
    for idx, orig_i in enumerate(order):
        out[orig_i] = adj_sorted[idx]
    return out.tolist()


def _eta_squared(y: np.ndarray, groups: List[np.ndarray]) -> Optional[float]:
    try:
        grand_mean = float(np.mean(y))
        ss_total = float(np.sum((y - grand_mean) ** 2))
        if ss_total == 0:
            return None
        ss_between = 0.0
        for g in groups:
            if g.size == 0:
                continue
            ss_between += float(g.size * (np.mean(g) - grand_mean) ** 2)
        return ss_between / ss_total
    except Exception:
        return None


def _epsilon_squared(h: float, n: int, k: int) -> Optional[float]:
    if n <= k or k < 2:
        return None
    return float((h - k + 1) / (n - k))


def _pairwise_table(
    pairs: List[Tuple[str, str]],
    p_raw: List[float],
    p_adj: List[float],
    *,
    method: str,
) -> Dict[str, Any]:
    return {
        "method": method,
        "comparisons": [
            {"group_a": a, "group_b": b, "p_raw": float(pr), "p_adj": float(pa)}
            for (a, b), pr, pa in zip(pairs, p_raw, p_adj, strict=True)
        ],
    }


def analyze_metric(
    df: pd.DataFrame,
    *,
    metric: str,
    treatment_col: str = "treatment",
    alpha: float,
    enable_posthoc: bool,
) -> Tuple[Dict[str, Any], List[AnalysisWarning]]:
    """
    Analyze a single metric; returns (result_dict, warnings).
    The result_dict is designed to become a row in STATS sheet.
    """
    warnings: List[AnalysisWarning] = []

    sub = df[[treatment_col, metric]].dropna()
    if sub.empty:
        return (
            {
                "metric": metric,
                "test": None,
                "p_value": None,
                "effect_size": None,
                "assumptions_shapiro_min_p": None,
                "assumptions_levene_p": None,
                "posthoc": None,
                "disabled_reason": "No data available for metric.",
            },
            warnings,
        )

    rep = _rep_by_treatment(sub, treatment_col)
    min_n = _min_n(rep)
    if min_n < 2:
        msg = "Inferential statistics disabled due to lack of replication."
        warnings.append(
            AnalysisWarning(
                code=WarningCode.inferential_disabled_no_replication,
                message=msg,
                context={"metric": metric, "replication_by_treatment": rep, "min_n_per_treatment": min_n},
            )
        )
        return (
            {
                "metric": metric,
                "test": None,
                "p_value": None,
                "effect_size": None,
                "assumptions_shapiro_min_p": None,
                "assumptions_levene_p": None,
                "posthoc": None,
                "disabled_reason": msg,
            },
            warnings,
        )

    treatments = list(rep.keys())
    groups = [sub.loc[sub[treatment_col] == t, metric].to_numpy(dtype=float) for t in treatments]
    y = sub[metric].to_numpy(dtype=float)

    shapiro_min_p, normal, shapiro_note = _shapiro_min_p(groups, alpha)
    lev_p = _levene_p(groups)
    homoscedastic = (lev_p is not None) and (lev_p > alpha)

    result: Dict[str, Any] = {
        "metric": metric,
        "test": None,
        "p_value": None,
        "effect_size": None,
        "assumptions_shapiro_min_p": shapiro_min_p,
        "assumptions_levene_p": lev_p,
        "assumptions_notes": shapiro_note,
        "posthoc": None,
        "disabled_reason": None,
    }

    # Choose test
    if normal and homoscedastic:
        try:
            f = sps.f_oneway(*groups)
            result["test"] = "anova"
            result["p_value"] = float(f.pvalue)
            result["effect_size"] = _eta_squared(y, groups)
        except Exception as e:
            result["disabled_reason"] = f"ANOVA failed: {e}"
            return result, warnings

        if enable_posthoc:
            try:
                tuk = pairwise_tukeyhsd(
                    endog=sub[metric].to_numpy(float),
                    groups=sub[treatment_col].to_numpy(str),
                    alpha=alpha,
                )
                header, *rows = tuk._results_table.data  # type: ignore[attr-defined]
                comps = []
                for r in rows:
                    comps.append(
                        {
                            "group_a": r[0],
                            "group_b": r[1],
                            "mean_diff": float(r[2]),
                            "p_adj": float(r[3]),
                            "ci_lower": float(r[4]),
                            "ci_upper": float(r[5]),
                            "reject": bool(r[6]),
                        }
                    )
                result["posthoc"] = {"method": "tukey_hsd", "comparisons": comps}
            except Exception as e:
                warnings.append(
                    AnalysisWarning(
                        code=WarningCode.validation_adjustment,
                        message="Posthoc (Tukey HSD) failed; returning omnibus result only.",
                        context={"metric": metric, "error": str(e)},
                    )
                )

        return result, warnings

    if normal and not homoscedastic:
        # Welch ANOVA via statsmodels anova_oneway
        try:
            from statsmodels.stats.oneway import anova_oneway  # type: ignore

            welch = anova_oneway(
                sub[metric].to_numpy(float),
                sub[treatment_col].to_numpy(str),
                use_var="unequal",
                welch_correction=True,
            )
            result["test"] = "welch_anova"
            result["p_value"] = float(welch.pvalue)
            result["effect_size"] = None
            result["df_num"] = float(getattr(welch, "df_num", np.nan))
            result["df_denom"] = float(getattr(welch, "df_denom", np.nan))
        except Exception as e:
            result["disabled_reason"] = f"Welch ANOVA failed: {e}"
            return result, warnings

        if enable_posthoc:
            pairs: List[Tuple[str, str]] = []
            p_raw: List[float] = []
            for a, b in combinations(treatments, 2):
                ga = sub.loc[sub[treatment_col] == a, metric].to_numpy(float)
                gb = sub.loc[sub[treatment_col] == b, metric].to_numpy(float)
                t = sps.ttest_ind(ga, gb, equal_var=False, nan_policy="omit")
                pairs.append((a, b))
                p_raw.append(float(t.pvalue))
            p_adj = _holm_adjust(p_raw)
            result["posthoc"] = _pairwise_table(pairs, p_raw, p_adj, method="games_howell_approx_holm")

        return result, warnings

    # Non-normal -> Kruskal
    try:
        h = sps.kruskal(*groups)
        result["test"] = "kruskal"
        result["p_value"] = float(h.pvalue)
        result["effect_size"] = _epsilon_squared(float(h.statistic), int(y.size), len(groups))
    except Exception as e:
        result["disabled_reason"] = f"Kruskal-Wallis failed: {e}"
        return result, warnings

    if enable_posthoc:
        pairs = []
        p_raw = []
        for a, b in combinations(treatments, 2):
            ga = sub.loc[sub[treatment_col] == a, metric].to_numpy(float)
            gb = sub.loc[sub[treatment_col] == b, metric].to_numpy(float)
            u = sps.mannwhitneyu(ga, gb, alternative="two-sided")
            pairs.append((a, b))
            p_raw.append(float(u.pvalue))
        p_adj = _holm_adjust(p_raw)
        result["posthoc"] = _pairwise_table(pairs, p_raw, p_adj, method="dunn_approx_mannwhitney_holm")

    return result, warnings


def run_inferential_statistics(
    unit_kpis: List[Any],
    *,
    metrics: List[str],
    options: StatsOptions,
) -> Tuple[pd.DataFrame, Dict[str, int], int, bool, List[AnalysisWarning]]:
    """
    Run inferential statistics for the given list of metrics.

    Returns:
        stats_df, replication_by_treatment (unit counts), min_n_per_treatment, inferential_enabled_overall, warnings
    """
    df = pd.DataFrame([getattr(x, "model_dump", lambda: x)() for x in unit_kpis])  # support pydantic models or dicts

    rep_all = df.groupby("treatment").size().astype(int).to_dict()
    min_n_all = int(min(rep_all.values())) if rep_all else 0
    enabled_overall = min_n_all >= 2

    warnings: List[AnalysisWarning] = []
    if not enabled_overall:
        warnings.append(
            AnalysisWarning(
                code=WarningCode.inferential_disabled_no_replication,
                message="Inferential statistics disabled due to lack of replication.",
                context={"replication_by_treatment": rep_all, "min_n_per_treatment": min_n_all},
            )
        )

    rows: List[Dict[str, Any]] = []
    for metric in metrics:
        res, w = analyze_metric(
            df,
            metric=metric,
            alpha=float(options.alpha),
            enable_posthoc=bool(options.enable_posthoc),
        )
        rows.append(res)
        warnings.extend(w)

    stats_df = pd.DataFrame(rows)
    return stats_df, rep_all, min_n_all, enabled_overall, warnings
