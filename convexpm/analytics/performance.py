"""Performance summary metrics."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from convexpm.analytics.returns import (
    cagr,
    money_weighted_return,
    monthly_returns,
    rolling_returns,
    time_weighted_return,
    total_return,
    mtd_return,
    ytd_return,
)
from convexpm.transactions import Trade


def performance_summary(nav_history: pd.DataFrame, trades: Iterable[Trade] | None = None) -> pd.Series:
    """Return core V1 performance metrics."""
    monthly = monthly_returns(nav_history)
    rolling_252 = rolling_returns(nav_history, window=252)
    return pd.Series(
        {
            "total_return": total_return(nav_history),
            "cagr": cagr(nav_history),
            "ytd_return": ytd_return(nav_history),
            "mtd_return": mtd_return(nav_history),
            "twr": time_weighted_return(nav_history, trades=trades),
            "xirr": money_weighted_return(nav_history, trades or []),
            "latest_monthly_return": float(monthly.iloc[-1]) if not monthly.empty else 0.0,
            "latest_rolling_252_return": float(rolling_252.iloc[-1]) if not rolling_252.empty else 0.0,
        },
        dtype="object",
    )
