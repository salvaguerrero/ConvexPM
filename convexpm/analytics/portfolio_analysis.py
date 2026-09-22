"""Combined realized-performance and current-risk portfolio analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from convexpm.analytics.asset_returns import instrument_returns
from convexpm.analytics.current_risk import current_risk
from convexpm.analytics.performance import performance_summary
from convexpm.utils.dates import DateLike


@dataclass(slots=True)
class PortfolioAnalysisResult:
    """Structured result for Portfolio.analyze."""

    portfolio: pd.Series
    instruments: pd.DataFrame
    return_attribution: pd.DataFrame
    risk_contribution: pd.DataFrame
    correlation: pd.DataFrame
    covariance: pd.DataFrame
    weights: pd.Series
    metadata: dict[str, object]


def return_attribution(
    portfolio: object,
    *,
    start: DateLike | None = None,
    end: DateLike | None = None,
) -> pd.DataFrame:
    """Return chain-linked percentage-point contributions from actual holdings."""
    instrument_ids = sorted({trade.instrument_id for trade in portfolio.trades})
    if not instrument_ids:
        return pd.DataFrame(columns=["return_contribution"])

    returns = instrument_returns(
        portfolio.market_data,
        instrument_ids,
        start=start,
        end=end,
    )
    if returns.empty:
        return pd.DataFrame(
            {"return_contribution": [0.0] * len(instrument_ids)},
            index=pd.Index(instrument_ids, name="instrument_id"),
        )

    weight_rows: list[pd.Series] = []
    for return_date in returns.index:
        previous_date = (return_date - pd.Timedelta(days=1)).date()
        values = portfolio.market_values(previous_date)
        row = pd.Series(0.0, index=returns.columns, dtype=float)
        if not values.empty:
            by_id = values.set_index("instrument_id")["weight"].astype(float)
            common = row.index.intersection(by_id.index)
            row.loc[common] = by_id.reindex(common)
        row.name = return_date
        weight_rows.append(row)

    historical_weights = pd.DataFrame(weight_rows, index=returns.index).fillna(0.0)
    daily_contribution = returns * historical_weights
    daily_portfolio_return = daily_contribution.sum(axis=1)

    future_growth = pd.Series(1.0, index=returns.index, dtype=float)
    running = 1.0
    for dt in reversed(returns.index):
        future_growth.loc[dt] = running
        running *= 1.0 + float(daily_portfolio_return.loc[dt])

    linked = daily_contribution.mul(future_growth, axis=0)
    aggregate = linked.sum(axis=0)
    result = aggregate.rename("return_contribution").to_frame()
    result.index.name = "instrument_id"
    return result


def analyze_portfolio(
    portfolio: object,
    *,
    start: DateLike | None = None,
    end: DateLike | None = None,
    benchmark: str | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> PortfolioAnalysisResult:
    """Build the full portfolio analysis view."""
    nav = portfolio.nav_history(start=start, end=end)
    performance = performance_summary(nav, trades=portfolio.trades)
    current = current_risk(
        portfolio,
        start=start,
        end=end,
        benchmark=benchmark,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )
    attribution = return_attribution(portfolio, start=start, end=end)

    instrument_ids = sorted({trade.instrument_id for trade in portfolio.trades})
    asset_returns = instrument_returns(
        portfolio.market_data,
        instrument_ids,
        start=start,
        end=end,
    )
    benchmark_returns = None
    if benchmark is not None:
        benchmark_returns = instrument_returns(
            portfolio.market_data,
            [benchmark],
            start=start,
            end=end,
        )[benchmark]

    instrument_rows = []
    for instrument_id in instrument_ids:
        series = asset_returns[instrument_id]
        stats = _instrument_statistics(
            series,
            benchmark_returns=benchmark_returns,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )
        instrument = portfolio.registry.get(instrument_id) if instrument_id in portfolio.registry else None
        stats.update(
            {
                "instrument_id": instrument_id,
                "name": instrument.name if instrument is not None else instrument_id,
                "current_weight": float(current.weights.get(instrument_id, 0.0)),
                "return_contribution": float(
                    attribution["return_contribution"].get(instrument_id, 0.0)
                ),
                "risk_contribution": float(
                    current.risk_contribution["risk_contribution"].get(instrument_id, 0.0)
                ),
            }
        )
        instrument_rows.append(stats)

    instruments = pd.DataFrame(instrument_rows).set_index("instrument_id")
    portfolio_summary = pd.concat([performance, current.metrics])
    metadata = dict(current.metadata)
    metadata["realized_performance_basis"] = "historical holdings"
    metadata["current_risk_basis"] = "current weights + historical instrument returns"

    return PortfolioAnalysisResult(
        portfolio=portfolio_summary,
        instruments=instruments,
        return_attribution=attribution,
        risk_contribution=current.risk_contribution,
        correlation=current.correlation,
        covariance=current.covariance,
        weights=current.weights,
        metadata=metadata,
    )


def _instrument_statistics(
    returns: pd.Series,
    *,
    benchmark_returns: pd.Series | None,
    risk_free_rate: float,
    periods_per_year: int,
) -> dict[str, float]:
    returns = returns.dropna().astype(float)
    if returns.empty:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "annualized_volatility": 0.0,
            "max_drawdown": 0.0,
            "downside_deviation": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "beta": 0.0,
        }

    total = float((1.0 + returns).prod() - 1.0)
    years = max((returns.index.max() - returns.index.min()).days / 365.25, 0.0)
    cagr_value = (
        float((1.0 + total) ** (1.0 / years) - 1.0)
        if years > 0 and total > -1.0
        else 0.0
    )
    volatility = (
        float(returns.std(ddof=1) * np.sqrt(periods_per_year))
        if len(returns) >= 2
        else 0.0
    )
    nav = (1.0 + returns).cumprod()
    drawdown = nav / nav.cummax() - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    target = risk_free_rate / periods_per_year
    downside = np.minimum(returns - target, 0.0)
    downside_deviation = float(
        np.sqrt(np.mean(np.square(downside))) * np.sqrt(periods_per_year)
    )
    excess = returns - target
    std = excess.std(ddof=1)
    sharpe = (
        0.0
        if len(excess) < 2 or std == 0 or np.isnan(std)
        else float(excess.mean() / std * np.sqrt(periods_per_year))
    )
    sortino = (
        0.0
        if downside_deviation == 0.0
        else float((excess.mean() * periods_per_year) / downside_deviation)
    )

    beta_value = 0.0
    if benchmark_returns is not None:
        aligned = pd.concat(
            [returns.rename("asset"), benchmark_returns.rename("benchmark")],
            axis=1,
        ).dropna()
        if len(aligned) >= 2:
            variance = aligned["benchmark"].var(ddof=1)
            if variance != 0 and not np.isnan(variance):
                beta_value = float(
                    aligned["asset"].cov(aligned["benchmark"]) / variance
                )

    return {
        "total_return": total,
        "cagr": cagr_value,
        "annualized_volatility": volatility,
        "max_drawdown": max_drawdown,
        "downside_deviation": downside_deviation,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "beta": beta_value,
    }
