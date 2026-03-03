"""
Statistical analysis routines for PCTA.

Implements the statistical safety rule:
- If n per treatment < 2, inferential testing is disabled.
- Never produce p-values without replication.

Approach (when replication is sufficient):
1) Assumption checks:
   - Normality by group: Shapiro-Wilk (per treatment if n>=3, else skipped)
   - Homogeneity: Levene test
2) Omnibus test selection:
   - If normal & homoscedastic: one-way ANOVA
   - If normal but heteroscedastic: Welch ANOVA
   - If non-normal: Kruskal-Wallis
3) Posthoc (optional, depending on test):
   - ANOVA: Tukey HSD
   - Kruskal: pairwise Mann-Whitney with Holm correction (approx)
   - Welch: pairwise t-tests with Holm correction (approx)

Notes:
- This module operates on already-computed per-unit metrics.
- Missing values for a metric are dropped for that metric only.
- Effect sizes are provided where straightforward:
  - ANOVA: eta-squared (SS_between / SS_total)
  - Kruskal: epsilon-squared (H - k + 1) / (n - k) when possible

All results are returned via Pydantic schemas from core.schemas.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from .schemas import (
    AnalysisWarning,
    AssumptionChecks,
    ComputedMetrics,
    InferentialResult,
    WarningCode,
)


@dataclass(frozen=True)
class StatsOptions:
    """Configuration for statistical analysis."""
    alpha: float = 0.05
    enable_posthoc: bool = True


def _replication_by_treatment(df: pd.DataFrame, *, treatment_col: str) -> Dict[str, int]:
    return df.groupby(treatment_col, dropna=False).size().astype(int).to_dict()


def _min_n(rep_by_trt: Dict[str, int]) -> int:
    return int(min(rep_by_trt.values())) if rep_by_trt else 0


def _shapiro_p(values: np.ndarray) -> Optional[float]:
    # Shapiro requires 3<=n<=5000 (scipy supports >5000 with warning/approx; we keep conservative)
    n = int(values.size)
    if n < 3:
        return None
    try:
        return float(sps.shapiro(values).pvalue)
    except Exception:
        return None


def _levene_p(groups: List[np.ndarray]) -> Optional[float]:
    if len(groups) < 2:
        return None
    # Levene needs at least 2 observations per group to be meaningful
    if any(g.size < 2 for g in groups):
        return None
    try:
        return float(sps.levene(*groups, center="median").pvalue)
    except Exception:
        return None


def _holm_adjust(pvals: List[float]) -> List[float]:
    """
    Holm step-down adjustment.
    Returns adjusted p-values in original order.
    """
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    p_sorted = np.array([pvals[i] for i in order], dtype=float)
    adj_sorted = np.empty(m, dtype=float)
    # Step-down: max over previous adjusted to keep monotonicity
    running_max = 0.0
    for j, p in enumerate(p_sorted):
        adj = (m - j) * p
        running_max = max(running_max, adj)
        adj_sorted[j] = min(1.0, running_max)
    # Re-map to original positions
    adj = np.empty(m, dtype=float)
    for idx, orig_i in enumerate(order):
        adj[orig_i] = adj_sorted[idx]
    return adj.tolist()


def _anova_eta_squared(y: np.ndarray, groups: List[np.ndarray]) -> Optional[float]:
    try:
        grand_mean = float(np.mean(y))
        ss_total = float(np.sum((y - grand_mean) ** 2))
        if ss_total == 0:
            return None
        ss_between = 0.0
        start = 0
        for g in groups:
            n = g.size
            if n == 0:
                continue
            ss_between += float(n * (np.mean(g) - grand_mean) ** 2)
            start += n
        return ss_between / ss_total
    except Exception:
        return None


def _kruskal_epsilon_squared(h_stat: float, n: int, k: int) -> Optional[float]:
    # Epsilon-squared for Kruskal-Wallis
    if n <= k or k < 2:
        return None
    return float((h_stat - k + 1) / (n - k))


def _pairwise_table(
    pairs: List[Tuple[str, str]],
    p_raw: List[float],
    p_adj: List[float],
    *,
    method: str,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for (a, b), pr, pa in zip(pairs, p_raw, p_adj, strict=True):
        rows.append({"group_a": a, "group_b": b, "p_raw": float(pr), "p_adj": float(pa)})
    return {"method": method, "comparisons": rows}


def analyze_metric(
    df: pd.DataFrame,
    *,
    metric: str,
    treatment_col: str = "treatment",
    alpha: float,
    enable_posthoc: bool,
) -> Tuple[InferentialResult, List[AnalysisWarning]]:
    """
    Perform inferential analysis for a single metric.

    Safety:
    - If any treatment has n<2 for this metric after dropping NA, disable.
    """
    warnings: List[AnalysisWarning] = []

    sub = df[[treatment_col, metric]].dropna()
    if sub.empty:
        return (
            InferentialResult(
                metric=metric,
                test=None,
                p_value=None,
                disabled_reason="No data available for metric.",
            ),
            warnings,
        )

    rep = _replication_by_treatment(sub, treatment_col=treatment_col)
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
            InferentialResult(
                metric=metric,
                test=None,
                p_value=None,
                disabled_reason=msg,
            ),
            warnings,
        )

    # Build groups
    treatments = list(rep.keys())
    groups = [sub.loc[sub[treatment_col] == t, metric].to_numpy(dtype=float) for t in treatments]
    y = sub[metric].to_numpy(dtype=float)

    # Assumption checks
    shapiro_ps = [_shapiro_p(g) for g in groups]
    # "Normal" if all available Shapiro p-values > alpha; ignore None
    shapiro_available = [p for p in shapiro_ps if p is not None]
    normal = bool(shapiro_available) and all(p > alpha for p in shapiro_available)

    lev_p = _levene_p(groups)
    homoscedastic = (lev_p is not None) and (lev_p > alpha)

    assumptions = AssumptionChecks(
        metric=metric,
        shapiro_p=float(min(shapiro_available)) if shapiro_available else None,
        levene_p=lev_p,
        notes=(
            "Shapiro performed per group where n>=3; stored value is min p across groups."
            if shapiro_available
            else "Shapiro skipped for groups with n<3."
        ),
    )

    # Omnibus test
    if normal and homoscedastic:
        # One-way ANOVA
        try:
            f_res = sps.f_oneway(*groups)
            p = float(f_res.pvalue)
            effect = _anova_eta_squared(y, groups)
            result = InferentialResult(
                metric=metric,
                test="anova",
                p_value=p,
                effect_size=effect,
                df=None,
                assumptions=assumptions,
                posthoc=None,
            )
        except Exception as e:
            result = InferentialResult(
                metric=metric,
                test="anova",
                p_value=None,
                disabled_reason=f"ANOVA failed: {e}",
                assumptions=assumptions,
            )
            return result, warnings

        # Posthoc: Tukey HSD
        if enable_posthoc:
            try:
                tuk = pairwise_tukeyhsd(endog=sub[metric].to_numpy(float), groups=sub[treatment_col].to_numpy(str), alpha=alpha)
                # Convert summary to structured form
                post = []
                # tuk.summary() is a SimpleTable; use tuk._results_table.data
                header, *rows = tuk._results_table.data  # type: ignore[attr-defined]
                # Expected: group1, group2, meandiff, p-adj, lower, upper, reject
                for r in rows:
                    post.append(
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
                result.posthoc = {"method": "tukey_hsd", "comparisons": post}
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
        # Welch ANOVA via statsmodels one-way anova is non-trivial; use scipy's oneway ANOVA doesn't do Welch.
        # Use scipy.stats.f_oneway as fallback? Not appropriate for Welch.
        # Implement Welch ANOVA using statsmodels (anova_oneway) if available.
        try:
            # statsmodels 0.14+: anova_oneway supports use_var="unequal"
            from statsmodels.stats.oneway import anova_oneway  # type: ignore

            welch = anova_oneway(
                sub[metric].to_numpy(float),
                sub[treatment_col].to_numpy(str),
                use_var="unequal",
                welch_correction=True,
            )
            p = float(welch.pvalue)
            # df info varies by version; best-effort
            df_info = {"df_num": float(getattr(welch, "df_num", np.nan)), "df_denom": float(getattr(welch, "df_denom", np.nan))}
            result = InferentialResult(
                metric=metric,
                test="welch_anova",
                p_value=p,
                effect_size=None,
                df=df_info,
                assumptions=assumptions,
                posthoc=None,
            )
        except Exception as e:
            result = InferentialResult(
                metric=metric,
                test="welch_anova",
                p_value=None,
                disabled_reason=f"Welch ANOVA failed: {e}",
                assumptions=assumptions,
            )
            return result, warnings

        if enable_posthoc:
            # Pairwise Welch t-tests with Holm correction
            pairs: List[Tuple[str, str]] = []
            p_raw: List[float] = []
            for a, b in combinations(treatments, 2):
                ga = sub.loc[sub[treatment_col] == a, metric].to_numpy(float)
                gb = sub.loc[sub[treatment_col] == b, metric].to_numpy(float)
                t_res = sps.ttest_ind(ga, gb, equal_var=False, nan_policy="omit")
                pairs.append((a, b))
                p_raw.append(float(t_res.pvalue))
            p_adj = _holm_adjust(p_raw)
            result.posthoc = _pairwise_table(pairs, p_raw, p_adj, method="welch_ttests_holm")
        return result, warnings

    # Non-normal: Kruskal-Wallis
    try:
        h_res = sps.kruskal(*groups)
        p = float(h_res.pvalue)
        eps2 = _kruskal_epsilon_squared(float(h_res.statistic), int(y.size), len(groups))
        result = InferentialResult(
            metric=metric,
            test="kruskal",
            p_value=p,
            effect_size=eps2,
            df=None,
            assumptions=assumptions,
            posthoc=None,
        )
    except Exception as e:
        result = InferentialResult(
            metric=metric,
            test="kruskal",
            p_value=None,
            disabled_reason=f"Kruskal-Wallis failed: {e}",
            assumptions=assumptions,
        )
        return result, warnings

    if enable_posthoc:
        # Pairwise Mann-Whitney U with Holm correction
        pairs = []
        p_raw = []
        for a, b in combinations(treatments, 2):
            ga = sub.loc[sub[treatment_col] == a, metric].to_numpy(float)
            gb = sub.loc[sub[treatment_col] == b, metric].to_numpy(float)
            # Require replication safety already satisfied (n>=2 per group), but Mann-Whitney can work with n>=1.
            u_res = sps.mannwhitneyu(ga, gb, alternative="two-sided")
            pairs.append((a, b))
            p_raw.append(float(u_res.pvalue))
        p_adj = _holm_adjust(p_raw)
        result.posthoc = _pairwise_table(pairs, p_raw, p_adj, method="mannwhitney_holm")
    return result, warnings


def run_inferential_statistics(
    unit_metrics: List[ComputedMetrics],
    *,
    metrics: List[str],
    options: StatsOptions,
) -> Tuple[List[InferentialResult], Dict[str, int], int, bool, List[AnalysisWarning]]:
    """
    Run inferential statistics across multiple metrics.

    Replication is evaluated per metric after dropping NAs. Additionally,
    we compute replication on the full dataset (unit count per treatment)
    as a summary.

    Returns:
        inferential_results,
        replication_by_treatment (based on unit count),
        min_n_per_treatment (based on unit count),
        inferential_enabled_overall (based on unit count),
        warnings
    """
    df = pd.DataFrame([m.model_dump() for m in unit_metrics])

    rep_all = _replication_by_treatment(df, treatment_col="treatment")
    min_n_all = _min_n(rep_all)
    inferential_enabled_overall = min_n_all >= 2

    warnings: List[AnalysisWarning] = []
    results: List[InferentialResult] = []

    if not inferential_enabled_overall:
        # Global safety warning; per-metric will also be disabled
        warnings.append(
            AnalysisWarning(
                code=WarningCode.inferential_disabled_no_replication,
                message="Inferential statistics disabled due to lack of replication.",
                context={"replication_by_treatment": rep_all, "min_n_per_treatment": min_n_all},
            )
        )

    for metric in metrics:
        r, w = analyze_metric(
            df,
            metric=metric,
            alpha=options.alpha,
            enable_posthoc=options.enable_posthoc,
        )
        results.append(r)
        warnings.extend(w)

    return results, rep_all, min_n_all, inferential_enabled_overall, warnings
