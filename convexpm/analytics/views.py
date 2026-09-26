"""Simple portfolio performance and risk views for interactive use."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from convexpm.analytics.asset_returns import instrument_returns
from convexpm.analytics.current_risk import current_risk
from convexpm.utils.dates import DateLike


@dataclass(slots=True)
class PerformanceViewResult:
    """User-facing portfolio performance view."""

    summary: pd.Series
    cumulative_return: pd.Series
    assets: pd.DataFrame
    yearly_return_contribution: pd.DataFrame
    metadata: dict[str, object]


@dataclass(slots=True)
class RiskViewResult:
    """User-facing portfolio risk view."""

    summary: pd.Series
    assets: pd.DataFrame
    yearly_risk_contribution: pd.DataFrame
    yearly_portfolio_volatility: pd.Series
    correlation: pd.DataFrame
    metadata: dict[str, object]


def performance_view(
    portfolio: object,
    *,
    start: DateLike | None = None,
    end: DateLike | None = None,
) -> PerformanceViewResult:
    """Build the simple performance view from realized historical holdings.

    Returns are reconstructed from historical portfolio weights and instrument
    returns, so adding capital does not itself create portfolio performance.
    """
    start_ts, end_ts = _portfolio_period(portfolio, start=start, end=end)
    instrument_ids = sorted({trade.instrument_id for trade in portfolio.trades})
    if not instrument_ids:
        raise ValueError("Cannot build a performance view for an empty portfolio")

    daily_contribution = _daily_return_contribution(
        portfolio,
        instrument_ids,
        start=start_ts,
        end=end_ts,
    )
    daily_portfolio_return = daily_contribution.sum(axis=1)
    cumulative = (1.0 + daily_portfolio_return).cumprod() - 1.0
    cumulative.name = "cumulative_return"
    if start_ts not in cumulative.index:
        cumulative = pd.concat(
            [pd.Series([0.0], index=pd.DatetimeIndex([start_ts]), name=cumulative.name), cumulative]
        ).sort_index()

    total = float(cumulative.iloc[-1]) if not cumulative.empty else 0.0
    elapsed_days = max((end_ts - start_ts).days, 0)
    annualized = (
        float((1.0 + total) ** (365.25 / elapsed_days) - 1.0)
        if elapsed_days > 0 and total > -1.0
        else 0.0
    )

    aggregate = _chain_link_contribution(daily_contribution)
    current_weights = portfolio.weights(end_ts.date())
    asset_rows = []
    for instrument_id in instrument_ids:
        returns = instrument_returns(
            portfolio.market_data,
            [instrument_id],
            start=start_ts,
            end=end_ts,
        )[instrument_id]
        asset_return = float((1.0 + returns).prod() - 1.0) if not returns.empty else 0.0
        asset_rows.append(
            {
                "instrument_id": instrument_id,
                "name": _instrument_name(portfolio, instrument_id),
                "current_weight": float(current_weights.get(instrument_id, 0.0)),
                "asset_return": asset_return,
                "return_contribution": float(aggregate.get(instrument_id, 0.0)),
            }
        )
    assets = pd.DataFrame(asset_rows).set_index("instrument_id")

    yearly = _yearly_return_contribution(daily_contribution)
    summary = pd.Series(
        {
            "portfolio_return": total,
            "cagr": annualized,
            "start_nav": float(portfolio.nav(start_ts.date())),
            "end_nav": float(portfolio.nav(end_ts.date())),
        },
        dtype=float,
    )
    metadata = {
        "analysis_start": start_ts.date(),
        "analysis_end": end_ts.date(),
        "performance_basis": "historical holdings + chain-linked asset contributions",
        "external_cash_flows": "excluded from return through historical weights",
        "yearly_matrix": "asset rows, calendar-year columns; values are return contribution",
    }
    return PerformanceViewResult(
        summary=summary,
        cumulative_return=cumulative,
        assets=assets,
        yearly_return_contribution=yearly,
        metadata=metadata,
    )


def risk_view(
    portfolio: object,
    *,
    start: DateLike | None = None,
    end: DateLike | None = None,
    periods_per_year: int = 252,
) -> RiskViewResult:
    """Build the simple current-allocation risk view.

    start and end define the historical risk lookback. Yearly columns use the
    allocation held at each year-end and an expanding history beginning at
    start.
    """
    start_ts, end_ts = _risk_period(portfolio, start=start, end=end)
    current = current_risk(
        portfolio,
        start=start_ts,
        end=end_ts,
        periods_per_year=periods_per_year,
    )
    max_drawdown = _max_drawdown_from_returns(current.synthetic_returns)
    summary = pd.Series(
        {
            "annualized_volatility": float(current.metrics["annualized_volatility"]),
            "max_drawdown": max_drawdown,
        },
        dtype=float,
    )

    annualized_asset_vol = np.sqrt(np.diag(current.covariance) * periods_per_year)
    asset_rows = []
    for instrument_id, volatility in zip(current.covariance.columns, annualized_asset_vol, strict=True):
        asset_rows.append(
            {
                "instrument_id": instrument_id,
                "name": _instrument_name(portfolio, instrument_id),
                "weight": float(current.weights.get(instrument_id, 0.0)),
                "asset_volatility": float(volatility),
                "risk_contribution": float(
                    current.risk_contribution["risk_contribution"].get(instrument_id, 0.0)
                ),
            }
        )
    assets = pd.DataFrame(asset_rows).set_index("instrument_id")

    yearly_contribution, yearly_volatility = _yearly_risk(
        portfolio,
        start=start_ts,
        end=end_ts,
        periods_per_year=periods_per_year,
    )
    metadata = dict(current.metadata)
    metadata.update(
        {
            "risk_basis": "allocation at analysis date + historical instrument returns",
            "yearly_risk_basis": "allocation at each year-end + expanding history from analysis_start",
            "yearly_matrix": "asset rows are percentage contribution to volatility; final row is portfolio volatility",
        }
    )
    return RiskViewResult(
        summary=summary,
        assets=assets,
        yearly_risk_contribution=yearly_contribution,
        yearly_portfolio_volatility=yearly_volatility,
        correlation=current.correlation,
        metadata=metadata,
    )


def _daily_return_contribution(
    portfolio: object,
    instrument_ids: list[str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    returns = instrument_returns(
        portfolio.market_data,
        instrument_ids,
        start=start,
        end=end,
    )
    if returns.empty:
        return pd.DataFrame(columns=instrument_ids, dtype=float)

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
    return returns.mul(historical_weights, axis=1)


def _chain_link_contribution(daily_contribution: pd.DataFrame) -> pd.Series:
    if daily_contribution.empty:
        return pd.Series(0.0, index=daily_contribution.columns, dtype=float)
    portfolio_return = daily_contribution.sum(axis=1)
    future_growth = pd.Series(1.0, index=daily_contribution.index, dtype=float)
    running = 1.0
    for dt in reversed(daily_contribution.index):
        future_growth.loc[dt] = running
        running *= 1.0 + float(portfolio_return.loc[dt])
    return daily_contribution.mul(future_growth, axis=0).sum(axis=0)


def _yearly_return_contribution(daily_contribution: pd.DataFrame) -> pd.DataFrame:
    if daily_contribution.empty:
        return pd.DataFrame()
    columns: dict[str, pd.Series] = {}
    for year, group in daily_contribution.groupby(daily_contribution.index.year, sort=True):
        contribution = _chain_link_contribution(group)
        contribution.loc["Portfolio Return"] = float(contribution.sum())
        columns[str(year)] = contribution
    result = pd.DataFrame(columns).fillna(0.0)
    result.index.name = "asset"
    return result


def _yearly_risk(
    portfolio: object,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    periods_per_year: int,
) -> tuple[pd.DataFrame, pd.Series]:
    contributions: dict[str, pd.Series] = {}
    volatilities: dict[str, float] = {}
    for year in range(start.year, end.year + 1):
        year_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        if year_end < start:
            continue
        try:
            result = current_risk(
                portfolio,
                start=start,
                end=year_end,
                periods_per_year=periods_per_year,
            )
        except (KeyError, ValueError):
            continue
        label = str(year)
        contributions[label] = result.risk_contribution["risk_contribution"].copy()
        volatilities[label] = float(result.metrics["annualized_volatility"])

    matrix = pd.DataFrame(contributions).fillna(0.0)
    volatility = pd.Series(volatilities, name="Portfolio Volatility", dtype=float)
    if not matrix.empty:
        matrix.loc["Portfolio Volatility"] = volatility.reindex(matrix.columns)
        matrix.index.name = "asset"
    return matrix, volatility


def _max_drawdown_from_returns(returns: pd.Series) -> float:
    clean = returns.dropna().astype(float)
    if clean.empty:
        return 0.0
    wealth = (1.0 + clean).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())


def _portfolio_period(
    portfolio: object,
    *,
    start: DateLike | None,
    end: DateLike | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if not portfolio.trades:
        raise ValueError("Cannot analyze an empty portfolio")
    start_ts = pd.Timestamp(start or min(trade.date for trade in portfolio.trades)).normalize()
    end_ts = pd.Timestamp(end or portfolio._default_date()).normalize()
    if start_ts > end_ts:
        raise ValueError("start must be on or before end")
    return start_ts, end_ts


def _risk_period(
    portfolio: object,
    *,
    start: DateLike | None,
    end: DateLike | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if not portfolio.trades:
        raise ValueError("Cannot analyze an empty portfolio")
    end_ts = pd.Timestamp(end or portfolio._default_date()).normalize()
    if start is not None:
        start_ts = pd.Timestamp(start).normalize()
    else:
        current_ids = list(portfolio.weights(end_ts.date()).index)
        first_dates = []
        for instrument_id in current_ids:
            history = portfolio.market_data.get_history(instrument_id, end=end_ts)
            if not history.empty:
                first_dates.append(history["date"].min())
        if not first_dates:
            raise ValueError("Cannot calculate risk without market history")
        start_ts = pd.Timestamp(max(first_dates)).normalize()
    if start_ts > end_ts:
        raise ValueError("start must be on or before end")
    return start_ts, end_ts


def _instrument_name(portfolio: object, instrument_id: str) -> str:
    try:
        return portfolio.registry.get(instrument_id).name
    except KeyError:
        return instrument_id
