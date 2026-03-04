"""
Utility helpers for PCTA.

Includes:
- Column normalization
- Safe numeric parsing
- Date parsing helpers
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import pandas as pd


def normalize_columns(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Normalize column names:
    - lower-case
    - strip whitespace
    - spaces -> underscores
    """
    if df is None:
        return None
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    return out


def coalesce(*vals: Any) -> Any:
    """Return the first value that is not None and not NaN."""
    for v in vals:
        if v is None:
            continue
        if isinstance(v, float) and pd.isna(v):
            continue
        if pd.isna(v):
            continue
        return v
    return None


def safe_int(x: Any) -> int:
    """Parse integer-like values; raises ValueError if not parseable."""
    if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
        raise ValueError("Missing required integer value.")
    if isinstance(x, bool):
        raise ValueError("Boolean is not a valid integer value.")
    try:
        return int(float(x))
    except Exception as e:
        raise ValueError(f"Invalid integer value: {x!r}") from e


def safe_float(x: Any) -> float:
    """Parse float-like values; raises ValueError if not parseable."""
    if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
        raise ValueError("Missing required numeric value.")
    if isinstance(x, bool):
        raise ValueError("Boolean is not a valid numeric value.")
    try:
        return float(x)
    except Exception as e:
        raise ValueError(f"Invalid numeric value: {x!r}") from e


def parse_date_like(x: Any) -> pd.Timestamp:
    """
    Parse a date-like value to pandas Timestamp.
    Accepts datetime/date strings, datetime objects, or Excel serials (handled by pandas).
    """
    if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
        raise ValueError("Missing required date value.")
    try:
        return pd.to_datetime(x)
    except Exception as e:
        raise ValueError(f"Invalid date value: {x!r}") from e
