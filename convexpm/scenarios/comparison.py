"""Scenario comparison helpers."""

from __future__ import annotations

import pandas as pd


METRIC_LABELS = {
    "current_nav": "Current NAV",
    "total_return": "Total Return",
    "cagr": "CAGR",
    "ytd_return": "YTD Return",
    "mtd_return": "MTD Return",
    "twr": "TWR",
    "xirr": "XIRR",
    "annualized_volatility": "Volatility",
    "max_drawdown": "Max Drawdown",
    "current_drawdown": "Current Drawdown",
    "sharpe_ratio": "Sharpe",
    "sortino_ratio": "Sortino",
    "historical_var_95": "Historical VaR 95",
    "historical_cvar_95": "Historical CVaR 95",
}


def compare_portfolios(current: object, scenario: object) -> pd.DataFrame:
    """Return a Current vs Scenario metric comparison DataFrame."""
    current_metrics = _metric_bundle(current)
    scenario_metrics = _metric_bundle(scenario)
    rows = []
    for key, label in METRIC_LABELS.items():
        if key in current_metrics or key in scenario_metrics:
            rows.append(
                {
                    "Metric": label,
                    "Current": current_metrics.get(key),
                    "Scenario": scenario_metrics.get(key),
                    "Difference": _difference(current_metrics.get(key), scenario_metrics.get(key)),
                }
            )
    return pd.DataFrame(rows, columns=["Metric", "Current", "Scenario", "Difference"])


def _metric_bundle(portfolio: object) -> dict[str, float | None]:
    performance = portfolio.performance().to_dict()
    risk = portfolio.risk().to_dict()
    return {"current_nav": portfolio.nav(), **performance, **risk}


def _difference(current: object, scenario: object) -> float | None:
    if current is None or scenario is None:
        return None
    try:
        return float(scenario) - float(current)
    except (TypeError, ValueError):
        return None
