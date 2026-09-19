"""Risk analytics based on historical NAV returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from convexpm.analytics.returns import clean_nav_history, nav_returns


def annualized_volatility(nav_history: pd.DataFrame, periods_per_year: int = 252) -> float:
    """Annualized volatility using daily returns by default."""
    returns = nav_returns(nav_history)
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(nav_history: pd.DataFrame) -> float:
    """Worst peak-to-trough drawdown."""
    df = clean_nav_history(nav_history)
    if df.empty:
        return 0.0
    nav = df.set_index("date")["nav"]
    running_max = nav.cummax()
    drawdowns = nav / running_max - 1.0
    return float(drawdowns.min())


def current_drawdown(nav_history: pd.DataFrame) -> float:
    """Current drawdown from the latest historical high."""
    df = clean_nav_history(nav_history)
    if df.empty:
        return 0.0
    nav = df.set_index("date")["nav"]
    running_max = nav.cummax()
    return float(nav.iloc[-1] / running_max.iloc[-1] - 1.0)


def downside_deviation(
    nav_history: pd.DataFrame,
    *,
    target_return: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized downside deviation versus a per-period target return."""
    returns = nav_returns(nav_history)
    if returns.empty:
        return 0.0
    downside = np.minimum(returns - target_return, 0.0)
    return float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(periods_per_year))


def sharpe_ratio(
    nav_history: pd.DataFrame,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio."""
    returns = nav_returns(nav_history)
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(
    nav_history: pd.DataFrame,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sortino ratio."""
    returns = nav_returns(nav_history)
    if returns.empty:
        return 0.0
    target = risk_free_rate / periods_per_year
    downside = downside_deviation(nav_history, target_return=target, periods_per_year=periods_per_year)
    if downside == 0:
        return 0.0
    annualized_excess = (returns.mean() - target) * periods_per_year
    return float(annualized_excess / downside)


def historical_var(nav_history: pd.DataFrame, confidence: float = 0.95) -> float:
    """Historical VaR return threshold.

    The result is usually negative; for example ``-0.02`` means the 5th
    percentile daily return was -2%.
    """
    returns = nav_returns(nav_history)
    if returns.empty:
        return 0.0
    return float(returns.quantile(1.0 - confidence))


def historical_cvar(nav_history: pd.DataFrame, confidence: float = 0.95) -> float:
    """Historical CVaR / expected shortfall."""
    returns = nav_returns(nav_history)
    if returns.empty:
        return 0.0
    threshold = historical_var(nav_history, confidence=confidence)
    tail = returns.loc[returns <= threshold]
    if tail.empty:
        return float(threshold)
    return float(tail.mean())


def beta(nav_history: pd.DataFrame, benchmark_nav: pd.DataFrame) -> float:
    """Beta versus a benchmark NAV history."""
    aligned = _aligned_returns(nav_history, benchmark_nav)
    if aligned.empty:
        return 0.0
    benchmark_variance = aligned["benchmark"].var(ddof=1)
    if benchmark_variance == 0 or np.isnan(benchmark_variance):
        return 0.0
    return float(aligned["portfolio"].cov(aligned["benchmark"]) / benchmark_variance)


def correlation(nav_history: pd.DataFrame, benchmark_nav: pd.DataFrame) -> float:
    """Correlation versus a benchmark NAV history."""
    aligned = _aligned_returns(nav_history, benchmark_nav)
    if len(aligned) < 2:
        return 0.0
    corr = aligned["portfolio"].corr(aligned["benchmark"])
    return 0.0 if np.isnan(corr) else float(corr)


def risk_summary(
    nav_history: pd.DataFrame,
    *,
    risk_free_rate: float = 0.0,
    benchmark_nav: pd.DataFrame | None = None,
    periods_per_year: int = 252,
) -> pd.Series:
    """Return core V1 risk metrics."""
    summary = {
        "annualized_volatility": annualized_volatility(nav_history, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(nav_history),
        "current_drawdown": current_drawdown(nav_history),
        "downside_deviation": downside_deviation(nav_history, periods_per_year=periods_per_year),
        "sharpe_ratio": sharpe_ratio(
            nav_history,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        ),
        "sortino_ratio": sortino_ratio(
            nav_history,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        ),
        "historical_var_95": historical_var(nav_history, confidence=0.95),
        "historical_cvar_95": historical_cvar(nav_history, confidence=0.95),
    }
    if benchmark_nav is not None:
        summary["beta"] = beta(nav_history, benchmark_nav)
        summary["correlation"] = correlation(nav_history, benchmark_nav)
    return pd.Series(summary, dtype=float)


def _aligned_returns(nav_history: pd.DataFrame, benchmark_nav: pd.DataFrame) -> pd.DataFrame:
    portfolio = nav_returns(nav_history).rename("portfolio")
    benchmark = nav_returns(benchmark_nav).rename("benchmark")
    return pd.concat([portfolio, benchmark], axis=1).dropna()
