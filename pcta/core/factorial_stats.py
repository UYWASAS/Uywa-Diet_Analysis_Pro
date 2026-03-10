from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pcta.core.schemas import AnalysisWarning, WarningCode


@dataclass(frozen=True)
class FactorialOptions:
    alpha: float = 0.05
    include_interaction: bool = True
    anova_type: int = 2  # 2 (Type II) or 3 (Type III)


def _min_cell_replication(df: pd.DataFrame, a: str, b: str) -> int:
    """
    Min n per A×B cell (after NA drop on Y).
    """
    counts = df.groupby([a, b]).size()
    if counts.empty:
        return 0
    return int(counts.min())


def _coerce_categorical(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = out[c].astype("category")
    return out


def run_factorial_anova_df(
    df: pd.DataFrame,
    *,
    y_col: str,
    factor_a: str,
    factor_b: str,
    block_col: Optional[str] = None,
    options: FactorialOptions,
) -> Tuple[pd.DataFrame, Dict[str, Any], List[AnalysisWarning]]:
    """
    Factorial ANOVA using statsmodels OLS + anova_lm.

    Returns:
      - anova_df: rows for A, B, (A:B), (block if provided), Residual
      - meta: dict with replication info and formula
      - warnings
    """
    warnings: List[AnalysisWarning] = []

    required = [y_col, factor_a, factor_b] + ([block_col] if block_col else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        warnings.append(
            AnalysisWarning(
                code=WarningCode.validation_adjustment,
                message="Columnas requeridas no presentes para ANOVA factorial.",
                context={"missing": missing},
            )
        )
        return pd.DataFrame([]), {"enabled": False, "reason": "missing_columns", "missing": missing}, warnings

    sub = df[required].dropna()
    if sub.empty:
        warnings.append(
            AnalysisWarning(
                code=WarningCode.validation_adjustment,
                message="No hay datos disponibles (todo NA) para ANOVA factorial.",
                context={"y_col": y_col},
            )
        )
        return pd.DataFrame([]), {"enabled": False, "reason": "no_data"}, warnings

    # Replicación mínima por celda
    min_cell_n = _min_cell_replication(sub, factor_a, factor_b)
    if min_cell_n < 2:
        warnings.append(
            AnalysisWarning(
                code=WarningCode.inferential_disabled_no_replication,
                message="Inferencia factorial deshabilitada: se requiere n>=2 por celda (A×B).",
                context={"min_n_per_cell": min_cell_n},
            )
        )
        return (
            pd.DataFrame(
                [
                    {
                        "term": "factorial_anova",
                        "p_value": None,
                        "df": None,
                        "sum_sq": None,
                        "mean_sq": None,
                        "f_value": None,
                        "disabled_reason": "Inferencia factorial deshabilitada: n<2 por celda (A×B).",
                    }
                ]
            ),
            {
                "enabled": False,
                "reason": "no_replication_cells",
                "min_n_per_cell": min_cell_n,
            },
            warnings,
        )

    # statsmodels (import local para evitar hard fail si no está)
    try:
        import statsmodels.api as sm  # noqa: F401
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm
    except Exception as e:
        warnings.append(
            AnalysisWarning(
                code=WarningCode.validation_adjustment,
                message="statsmodels no disponible para ANOVA factorial.",
                context={"error": str(e)},
            )
        )
        return pd.DataFrame([]), {"enabled": False, "reason": "missing_statsmodels"}, warnings

    sub = _coerce_categorical(sub, [factor_a, factor_b] + ([block_col] if block_col else []))

    # Fórmula
    if options.include_interaction:
        core = f"C({factor_a}) * C({factor_b})"
    else:
        core = f"C({factor_a}) + C({factor_b})"

    if block_col:
        formula = f"{y_col} ~ {core} + C({block_col})"
    else:
        formula = f"{y_col} ~ {core}"

    # Ajuste modelo OLS
    model = smf.ols(formula=formula, data=sub).fit()

    # Type II vs III
    typ = int(options.anova_type)
    if typ not in (2, 3):
        typ = 2

    aov = anova_lm(model, typ=typ)

    # Formatear salida
    out = aov.reset_index().rename(
        columns={
            "index": "term",
            "sum_sq": "sum_sq",
            "df": "df",
            "mean_sq": "mean_sq",
            "F": "f_value",
            "PR(>F)": "p_value",
        }
    )

    # Normalizar nombres de terms a algo más legible
    def _pretty_term(t: str) -> str:
        # Ej: 'C(A)' 'C(A):C(B)' 'Residual'
        return t.replace("C(", "").replace(")", "")

    out["term"] = out["term"].astype(str).apply(_pretty_term)

    meta: Dict[str, Any] = {
        "enabled": True,
        "formula": formula,
        "anova_type": typ,
        "min_n_per_cell": min_cell_n,
        "n_rows": int(len(sub)),
    }
    return out, meta, warnings
