"""Simple portfolio performance and risk views for interactive use."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from convexpm.analytics.asset_returns import instrument_returns
from convexpm.analytics.current_risk import current_risk
from convexpm.analytics.returns import cash_settlement_parent_ids, is_external_capital_flow
from convexpm.portfolio.holdings import build_holdings
from convexpm.utils.dates import DateLike


@dataclass(slots=True)
class PerformanceViewResult:
    """User-facing portfolio performance view."""

    summary: pd.Series
    cumulative_return: pd.Series
    assets: pd.DataFrame
    standalone_asset_returns: pd.DataFrame
    asset_weights: pd.DataFrame
    yearly_return_contribution: pd.DataFrame
    metadata: dict[str, object]


@dataclass(slots=True)
class RiskViewResult:
    """User-facing portfolio risk view."""

    summary: pd.Series
    assets: pd.DataFrame
    asset_volatility: pd.DataFrame
    volatility_contribution: pd.DataFrame
    asset_sharpe: pd.DataFrame
    sharpe_contribution: pd.DataFrame
    yearly_risk_contribution: pd.DataFrame
    yearly_portfolio_volatility: pd.Series
    asset_correlation: pd.DataFrame
    weighted_correlation: pd.DataFrame
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

    standalone_asset_returns = _standalone_asset_returns_by_period(
        portfolio,
        instrument_ids,
        start=start_ts,
        end=end_ts,
    )
    aggregate = _chain_link_contribution(daily_contribution)
    current_weights = portfolio.weights(end_ts.date())
    asset_rows = []
    for instrument_id in instrument_ids:
        asset_rows.append(
            {
                "instrument_id": instrument_id,
                "name": _instrument_name(portfolio, instrument_id),
                "current_weight": float(current_weights.get(instrument_id, 0.0)),
                "asset_return": float(standalone_asset_returns.loc[instrument_id, "Total"])
                if instrument_id in standalone_asset_returns.index
                else 0.0,
                "return_contribution": float(aggregate.get(instrument_id, 0.0)),
            }
        )
    assets = pd.DataFrame(asset_rows).set_index("instrument_id")

    asset_weights = _asset_weights_by_period(portfolio, instrument_ids, start=start_ts, end=end_ts)
    yearly = _yearly_return_contribution(daily_contribution)
    cash_flows = _capital_flows(portfolio, start=start_ts, end=end_ts)
    nav_return = _nav_return(portfolio, start=start_ts, end=end_ts)
    nav_cagr = _annualize_return(nav_return, start=start_ts, end=end_ts)
    summary = pd.Series(
        {
            "portfolio_return": total,
            "cagr": annualized,
            "portfolio_return_ex_cash_flows": total,
            "cagr_ex_cash_flows": annualized,
            "portfolio_return_including_cash_flows": nav_return,
            "cagr_including_cash_flows": nav_cagr,
            "capital_inflows": cash_flows["capital_inflows"],
            "capital_outflows": cash_flows["capital_outflows"],
            "net_capital_flow": cash_flows["net_capital_flow"],
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
        "return_contribution_matrix": "asset rows, calendar-year columns plus Total; final row is portfolio return",
        "standalone_asset_return_matrix": "asset rows, calendar-year columns plus Total; values are base-currency returns while held, ignoring portfolio weight",
        "asset_weight_matrix": "asset rows, year-end columns plus Total; final row is portfolio weight",
    }
    return PerformanceViewResult(
        summary=_performance_summary_percent(summary),
        cumulative_return=_percentage_series(cumulative),
        assets=_display_index(
            portfolio,
            _performance_percentage_columns(
                assets,
                ["current_weight", "asset_return", "return_contribution"],
            ),
        ),
        standalone_asset_returns=_display_index(portfolio, _percentage_matrix(standalone_asset_returns)),
        asset_weights=_display_index(portfolio, _percentage_matrix(asset_weights)),
        yearly_return_contribution=_display_index(portfolio, _percentage_matrix(yearly)),
        metadata=metadata,
    )


def risk_view(
    portfolio: object,
    *,
    start: DateLike | None = None,
    end: DateLike | None = None,
    risk_free_rate: float = 0.0,
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
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )
    max_drawdown = _max_drawdown_from_returns(current.synthetic_returns)
    portfolio_sharpe = _sharpe_ratio(
        current.synthetic_returns,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )
    summary = pd.Series(
        {
            "annualized_volatility": float(current.metrics["annualized_volatility"]),
            "sharpe_ratio": portfolio_sharpe,
            "max_drawdown": max_drawdown,
        },
        dtype=float,
    )

    annualized_asset_vol = np.sqrt(np.diag(current.covariance) * periods_per_year)
    current_sharpe_contribution = _sharpe_contribution(
        current.synthetic_returns,
        current.weights,
        current.covariance,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )
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
                "asset_sharpe": _asset_sharpe_from_covariance(
                    current.synthetic_returns.index,
                    portfolio,
                    instrument_id,
                    start=start_ts,
                    end=end_ts,
                    risk_free_rate=risk_free_rate,
                    periods_per_year=periods_per_year,
                ),
                "sharpe_contribution": float(current_sharpe_contribution.get(instrument_id, 0.0)),
            }
        )
    assets = pd.DataFrame(asset_rows).set_index("instrument_id")

    risk_tables = _yearly_risk(
        portfolio,
        current=current,
        start=start_ts,
        end=end_ts,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )
    weighted_correlation = _weighted_correlation(current.covariance, current.weights)
    metadata = dict(current.metadata)
    metadata.update(
        {
            "risk_basis": "allocation at analysis date + historical instrument returns",
            "yearly_risk_basis": "allocation at each year-end + expanding history from analysis_start",
            "volatility_contribution_matrix": "asset rows are percentage contribution to volatility; final row is portfolio volatility",
            "asset_volatility_matrix": "asset rows, yearly columns plus Total; values are annualized standalone volatility",
            "asset_sharpe_matrix": "asset rows, yearly columns plus Total; values are standalone Sharpe ratios",
            "sharpe_contribution_matrix": "asset rows are contribution to portfolio Sharpe; final row is portfolio Sharpe",
            "asset_correlation": "raw return correlation for current portfolio assets",
            "weighted_correlation": "correlation of weighted current-allocation return contributions",
        }
    )
    return RiskViewResult(
        summary=summary,
        assets=_display_index(portfolio, _percentage_columns(assets, ["weight", "risk_contribution"])),
        asset_volatility=_display_index(portfolio, risk_tables["asset_volatility"]),
        volatility_contribution=_display_index(
            portfolio,
            _percentage_rows(risk_tables["volatility_contribution"], exclude=["Portfolio Volatility"]),
        ),
        asset_sharpe=_display_index(portfolio, risk_tables["asset_sharpe"]),
        sharpe_contribution=_display_index(portfolio, risk_tables["sharpe_contribution"]),
        yearly_risk_contribution=_display_index(
            portfolio,
            _percentage_rows(risk_tables["volatility_contribution"], exclude=["Portfolio Volatility"]),
        ),
        yearly_portfolio_volatility=risk_tables["portfolio_volatility"],
        asset_correlation=_display_matrix(portfolio, current.correlation),
        weighted_correlation=_display_matrix(portfolio, weighted_correlation),
        correlation=_display_matrix(portfolio, current.correlation),
        metadata=metadata,
    )


def _percentage_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce") * 100.0
    return result


def _percentage_rows(df: pd.DataFrame, *, exclude: list[str] | None = None) -> pd.DataFrame:
    result = df.copy()
    excluded = set(exclude or [])
    mask = ~result.index.isin(excluded)
    result.loc[mask] = result.loc[mask].apply(pd.to_numeric, errors="coerce") * 100.0
    return result


def _performance_summary_percent(summary: pd.Series) -> pd.Series:
    percent_keys = {
        "portfolio_return",
        "cagr",
        "portfolio_return_ex_cash_flows",
        "cagr_ex_cash_flows",
        "portfolio_return_including_cash_flows",
        "cagr_including_cash_flows",
    }
    result = summary.copy()
    renamed = {}
    for key in percent_keys:
        if key in result.index:
            result.loc[key] = float(result.loc[key]) * 100.0
            renamed[key] = f"{key} (%)"
    return result.rename(index=renamed)


def _percentage_series(series: pd.Series) -> pd.Series:
    result = pd.to_numeric(series, errors="coerce") * 100.0
    result.name = f"{series.name} (%)" if series.name else None
    return result


def _performance_percentage_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    renamed = {}
    for column in columns:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce") * 100.0
            renamed[column] = f"{column} (%)"
    return result.rename(columns=renamed)


def _percentage_matrix(df: pd.DataFrame) -> pd.DataFrame:
    result = df.apply(pd.to_numeric, errors="coerce") * 100.0
    result.columns = [f"{column} (%)" for column in result.columns]
    return result


def _display_index(portfolio: object, df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.index = [_display_label(portfolio, item) for item in result.index]
    result.index.name = "asset"
    return result


def _display_matrix(portfolio: object, df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.index = [_display_label(portfolio, item) for item in result.index]
    result.columns = [_display_label(portfolio, item) for item in result.columns]
    return result


def _display_label(portfolio: object, value: object) -> object:
    if not isinstance(value, str):
        return value
    if value.startswith("Portfolio "):
        return value
    return _instrument_name(portfolio, value)


def _instrument_returns_union(
    portfolio: object,
    instrument_ids: list[str],
    *,
    start: DateLike | None = None,
    end: DateLike | None = None,
) -> pd.DataFrame:
    histories = _price_histories(portfolio)
    series: list[pd.Series] = []
    for instrument_id in instrument_ids:
        returns = _base_currency_returns_for_instrument(
            portfolio,
            histories,
            instrument_id,
            start=start,
            end=end,
        )
        series.append(returns.rename(instrument_id))
    if not series:
        return pd.DataFrame(columns=instrument_ids, dtype=float)
    aligned = pd.concat(series, axis=1, join="outer").fillna(0.0)
    aligned.index = pd.DatetimeIndex(aligned.index).normalize()
    return aligned.sort_index().loc[:, instrument_ids]


def _daily_return_contribution(
    portfolio: object,
    instrument_ids: list[str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    returns = _instrument_returns_union(
        portfolio,
        instrument_ids,
        start=start,
        end=end,
    )
    if returns.empty:
        return pd.DataFrame(columns=instrument_ids, dtype=float)

    histories = _price_histories(portfolio)
    weight_rows: list[pd.Series] = []
    for return_date in returns.index:
        previous_date = return_date - pd.Timedelta(days=1)
        by_id = _weights_on_date(portfolio, instrument_ids, previous_date, histories)
        row = by_id.reindex(returns.columns).fillna(0.0)
        row.name = return_date
        weight_rows.append(row)

    historical_weights = pd.DataFrame(weight_rows, index=returns.index).fillna(0.0)
    return returns.mul(historical_weights, axis=1)


def _price_histories(portfolio: object) -> dict[str, pd.DataFrame]:
    market = portfolio.market_data.read_all()
    return {
        instrument_id: group.sort_values("date").reset_index(drop=True)
        for instrument_id, group in market.groupby("instrument_id", sort=False)
    }


def _base_currency_returns_for_instrument(
    portfolio: object,
    histories: dict[str, pd.DataFrame],
    instrument_id: str,
    *,
    start: DateLike | None = None,
    end: DateLike | None = None,
) -> pd.Series:
    history = histories.get(instrument_id)
    if history is None or history.empty:
        return pd.Series(dtype=float, name=instrument_id)

    frame = history.loc[:, ["date", "value", "currency"]].drop_duplicates(subset=["date"], keep="last").copy()
    frame = frame.sort_values("date")
    base_values = []
    for _, row in frame.iterrows():
        date_value = pd.Timestamp(row["date"]).normalize()
        currency = str(row["currency"]).upper()
        base_values.append(float(row["value"]) * _fx_rate(portfolio, histories, currency, date_value))
    values = pd.Series(base_values, index=pd.DatetimeIndex(frame["date"]).normalize(), dtype=float)
    returns = values.pct_change(fill_method=None)
    if start is not None:
        returns = returns.loc[returns.index >= pd.Timestamp(start).normalize()]
    if end is not None:
        returns = returns.loc[returns.index <= pd.Timestamp(end).normalize()]
    returns.name = instrument_id
    return returns.dropna()


def _weights_on_date(
    portfolio: object,
    instrument_ids: list[str],
    date_value: pd.Timestamp,
    histories: dict[str, pd.DataFrame],
) -> pd.Series:
    holdings = build_holdings(portfolio.trades, date_value.date(), market_data=portfolio.market_data)
    values: dict[str, float] = {}
    for instrument_id in instrument_ids:
        quantity = float(holdings.get(instrument_id, 0.0))
        if abs(quantity) <= 1e-12:
            values[instrument_id] = 0.0
            continue
        price_row = _latest_price_row(histories, instrument_id, date_value)
        if price_row is None:
            values[instrument_id] = 0.0
            continue
        local_value = quantity * float(price_row["value"])
        currency = str(price_row["currency"]).upper()
        values[instrument_id] = local_value * _fx_rate(portfolio, histories, currency, date_value)

    total = sum(values.values())
    if abs(total) <= 1e-12:
        return pd.Series(0.0, index=instrument_ids, dtype=float)
    return pd.Series({instrument_id: value / total for instrument_id, value in values.items()}, dtype=float)


def _latest_price_row(
    histories: dict[str, pd.DataFrame],
    instrument_id: str,
    date_value: pd.Timestamp,
) -> pd.Series | None:
    history = histories.get(instrument_id)
    if history is None or history.empty:
        return None
    dates = pd.DatetimeIndex(history["date"])
    position = dates.searchsorted(date_value, side="right") - 1
    if position < 0:
        return None
    return history.iloc[int(position)]


def _fx_rate(
    portfolio: object,
    histories: dict[str, pd.DataFrame],
    currency: str,
    date_value: pd.Timestamp,
) -> float:
    base_currency = portfolio.base_currency.upper()
    if currency == base_currency:
        return 1.0
    direct = _latest_price_row(histories, f"{currency}{base_currency}", date_value)
    if direct is not None:
        return float(direct["value"])
    inverse = _latest_price_row(histories, f"{base_currency}{currency}", date_value)
    if inverse is not None:
        return 1.0 / float(inverse["value"])
    return 1.0


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


def _yearly_return_contribution(
    daily_contribution: pd.DataFrame,
    *,
    include_portfolio_return: bool = True,
) -> pd.DataFrame:
    if daily_contribution.empty:
        return pd.DataFrame()
    columns: dict[str, pd.Series] = {}
    for year, group in daily_contribution.groupby(daily_contribution.index.year, sort=True):
        contribution = _chain_link_contribution(group)
        if include_portfolio_return:
            contribution.loc["Portfolio Return"] = float(contribution.sum())
        columns[str(year)] = contribution
    total = _chain_link_contribution(daily_contribution)
    if include_portfolio_return:
        total.loc["Portfolio Return"] = float(total.sum())
    columns["Total"] = total
    result = pd.DataFrame(columns).fillna(0.0)
    result.index.name = "asset"
    return result


def _yearly_risk(
    portfolio: object,
    *,
    current: object,
    start: pd.Timestamp,
    end: pd.Timestamp,
    risk_free_rate: float,
    periods_per_year: int,
) -> dict[str, pd.DataFrame | pd.Series]:
    contributions: dict[str, pd.Series] = {}
    volatilities: dict[str, float] = {}
    asset_volatilities: dict[str, pd.Series] = {}
    asset_sharpes: dict[str, pd.Series] = {}
    sharpe_contributions: dict[str, pd.Series] = {}
    portfolio_sharpes: dict[str, float] = {}
    for year in range(start.year, end.year + 1):
        year_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        if year_end < start:
            continue
        try:
            result = current_risk(
                portfolio,
                start=start,
                end=year_end,
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
            )
        except (KeyError, ValueError):
            continue
        label = str(year)
        contributions[label] = result.risk_contribution["risk_contribution"].copy()
        volatilities[label] = float(result.metrics["annualized_volatility"])
        asset_volatilities[label] = _annualized_volatility_by_asset(result.covariance, periods_per_year)
        asset_sharpes[label] = _asset_sharpe_ratios(
            portfolio,
            result.covariance.columns,
            start=start,
            end=year_end,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )
        sharpe_contributions[label] = _sharpe_contribution(
            result.synthetic_returns,
            result.weights,
            result.covariance,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )
        portfolio_sharpes[label] = _sharpe_ratio(
            result.synthetic_returns,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )

    total_label = "Total"
    contributions[total_label] = current.risk_contribution["risk_contribution"].copy()
    volatilities[total_label] = float(current.metrics["annualized_volatility"])
    asset_volatilities[total_label] = _annualized_volatility_by_asset(current.covariance, periods_per_year)
    asset_sharpes[total_label] = _asset_sharpe_ratios(
        portfolio,
        current.covariance.columns,
        start=start,
        end=end,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )
    sharpe_contributions[total_label] = _sharpe_contribution(
        current.synthetic_returns,
        current.weights,
        current.covariance,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )
    portfolio_sharpes[total_label] = _sharpe_ratio(
        current.synthetic_returns,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )

    contribution_matrix = _matrix_with_total_row(
        contributions,
        total_row_name="Portfolio Volatility",
        total_values=volatilities,
    )
    volatility = pd.Series(volatilities, name="Portfolio Volatility", dtype=float)
    asset_volatility = pd.DataFrame(asset_volatilities).fillna(0.0)
    asset_volatility.index.name = "asset"
    asset_sharpe = pd.DataFrame(asset_sharpes).fillna(0.0)
    asset_sharpe.index.name = "asset"
    sharpe_matrix = _matrix_with_total_row(
        sharpe_contributions,
        total_row_name="Portfolio Sharpe",
        total_values=portfolio_sharpes,
    )
    return {
        "volatility_contribution": contribution_matrix,
        "portfolio_volatility": volatility,
        "asset_volatility": asset_volatility,
        "asset_sharpe": asset_sharpe,
        "sharpe_contribution": sharpe_matrix,
    }


def _standalone_asset_returns_by_period(
    portfolio: object,
    instrument_ids: list[str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    returns = _instrument_returns_union(portfolio, instrument_ids, start=start, end=end)
    columns: dict[str, pd.Series] = {}
    for year in range(start.year, end.year + 1):
        period_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        period_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        period_returns = returns.loc[(returns.index >= period_start) & (returns.index <= period_end)]
        columns[str(year)] = _standalone_asset_returns_for_period(
            portfolio,
            instrument_ids,
            period_returns,
        )
    columns["Total"] = _standalone_asset_returns_for_period(portfolio, instrument_ids, returns)
    result = pd.DataFrame(columns).fillna(0.0)
    result.index.name = "asset"
    return result


def _standalone_asset_returns_for_period(
    portfolio: object,
    instrument_ids: list[str],
    returns: pd.DataFrame,
) -> pd.Series:
    if returns.empty:
        return pd.Series(0.0, index=instrument_ids, dtype=float)

    held_returns = {instrument_id: [] for instrument_id in instrument_ids}
    for return_date, row in returns.iterrows():
        previous_date = (pd.Timestamp(return_date) - pd.Timedelta(days=1)).date()
        holdings = build_holdings(portfolio.trades, previous_date, market_data=portfolio.market_data)
        for instrument_id in instrument_ids:
            if abs(float(holdings.get(instrument_id, 0.0))) > 1e-12:
                held_returns[instrument_id].append(float(row.get(instrument_id, 0.0)))

    values = {
        instrument_id: float(pd.Series(asset_returns).add(1.0).prod() - 1.0)
        if asset_returns
        else 0.0
        for instrument_id, asset_returns in held_returns.items()
    }
    return pd.Series(values, dtype=float)


def _asset_weights_by_period(
    portfolio: object,
    instrument_ids: list[str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for year in range(start.year, end.year + 1):
        year_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        if year_end < start:
            continue
        weights = portfolio.weights(year_end.date()).reindex(instrument_ids).fillna(0.0)
        weights.loc["Portfolio Weight"] = float(weights.sum())
        columns[str(year)] = weights
    total = portfolio.weights(end.date()).reindex(instrument_ids).fillna(0.0)
    total.loc["Portfolio Weight"] = float(total.sum())
    columns["Total"] = total
    result = pd.DataFrame(columns).fillna(0.0)
    result.index.name = "asset"
    return result


def _capital_flows(portfolio: object, *, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, float]:
    inflows = 0.0
    outflows = 0.0
    settlement_parent_ids = cash_settlement_parent_ids(portfolio.trades)
    for trade in portfolio.trades:
        trade_date = pd.Timestamp(trade.date).normalize()
        if trade_date < start or trade_date > end:
            continue
        if not is_external_capital_flow(trade, settlement_parent_ids=settlement_parent_ids):
            continue
        amount = _trade_amount_in_base(portfolio, trade)
        if amount is None:
            continue
        if trade.side == "BUY":
            inflows += amount + float(trade.fees)
        else:
            outflows += max(amount - float(trade.fees), 0.0)
    return {
        "capital_inflows": inflows,
        "capital_outflows": outflows,
        "net_capital_flow": inflows - outflows,
    }


def _trade_amount_in_base(portfolio: object, trade: object) -> float | None:
    amount = trade.amount
    if amount is None and trade.quantity is not None and trade.value_per_unit is not None:
        amount = float(trade.quantity) * float(trade.value_per_unit)
    if amount is None:
        try:
            amount = float(trade.quantity) * float(portfolio.market_data.get_value(trade.instrument_id, trade.date))
        except (KeyError, TypeError):
            return None
    currency = (trade.currency or portfolio.base_currency).upper()
    if currency == portfolio.base_currency:
        return float(amount)
    try:
        from convexpm.portfolio.valuation import fx_rate_to_base

        fx_rate, _ = fx_rate_to_base(currency, portfolio.base_currency, portfolio.market_data, trade.date)
    except KeyError:
        return float(amount)
    return float(amount) * float(fx_rate)


def _nav_return(portfolio: object, *, start: pd.Timestamp, end: pd.Timestamp) -> float:
    start_nav = float(portfolio.nav(start.date()))
    end_nav = float(portfolio.nav(end.date()))
    return end_nav / start_nav - 1.0 if start_nav else 0.0


def _annualize_return(value: float, *, start: pd.Timestamp, end: pd.Timestamp) -> float:
    elapsed_days = max((end - start).days, 0)
    return float((1.0 + value) ** (365.25 / elapsed_days) - 1.0) if elapsed_days > 0 and value > -1.0 else 0.0


def _annualized_volatility_by_asset(covariance: pd.DataFrame, periods_per_year: int) -> pd.Series:
    return pd.Series(
        np.sqrt(np.diag(covariance) * periods_per_year),
        index=covariance.columns,
        dtype=float,
    )


def _asset_sharpe_ratios(
    portfolio: object,
    instrument_ids: pd.Index,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    risk_free_rate: float,
    periods_per_year: int,
) -> pd.Series:
    values = {
        instrument_id: _asset_sharpe_from_returns(
            _returns_for_asset(portfolio, instrument_id, start=start, end=end),
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )
        for instrument_id in instrument_ids
    }
    return pd.Series(values, dtype=float)


def _asset_sharpe_from_covariance(
    _aligned_index: pd.Index,
    portfolio: object,
    instrument_id: str,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    risk_free_rate: float,
    periods_per_year: int,
) -> float:
    return _asset_sharpe_from_returns(
        _returns_for_asset(portfolio, instrument_id, start=start, end=end),
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )


def _returns_for_asset(
    portfolio: object,
    instrument_id: str,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    try:
        return instrument_returns(
            portfolio.market_data,
            [instrument_id],
            start=start,
            end=end,
        )[instrument_id]
    except KeyError:
        return pd.Series(dtype=float)


def _asset_sharpe_from_returns(
    returns: pd.Series,
    *,
    risk_free_rate: float,
    periods_per_year: int,
) -> float:
    clean = returns.dropna().astype(float)
    if len(clean) < 2:
        return 0.0
    volatility = clean.std(ddof=1)
    if volatility == 0 or pd.isna(volatility):
        return 0.0
    excess = clean - (risk_free_rate / periods_per_year)
    return float((excess.mean() / volatility) * np.sqrt(periods_per_year))


def _sharpe_ratio(
    returns: pd.Series,
    *,
    risk_free_rate: float,
    periods_per_year: int,
) -> float:
    return _asset_sharpe_from_returns(
        returns,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )


def _sharpe_contribution(
    synthetic_returns: pd.Series,
    weights: pd.Series,
    covariance: pd.DataFrame,
    *,
    risk_free_rate: float,
    periods_per_year: int,
) -> pd.Series:
    portfolio_volatility = float(synthetic_returns.std(ddof=1) * np.sqrt(periods_per_year))
    if portfolio_volatility <= 1e-15 or pd.isna(portfolio_volatility):
        return pd.Series(0.0, index=covariance.columns, dtype=float)
    returns = covariance.attrs.get("returns")
    if isinstance(returns, pd.DataFrame):
        means = returns.mean()
    else:
        means = pd.Series(index=covariance.columns, dtype=float)
    means = means.reindex(covariance.columns).fillna(0.0)
    excess_means = means - (risk_free_rate / periods_per_year)
    contribution = weights.reindex(covariance.columns).fillna(0.0) * excess_means * periods_per_year
    contribution = contribution / portfolio_volatility
    contribution.name = "sharpe_contribution"
    return contribution


def _weighted_correlation(covariance: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    labels = list(covariance.columns)
    w = weights.reindex(labels).fillna(0.0).to_numpy(dtype=float)
    weighted_cov = covariance.loc[labels, labels].to_numpy(dtype=float) * np.outer(w, w)
    diagonal = np.sqrt(np.diag(weighted_cov))
    denominator = np.outer(diagonal, diagonal)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = weighted_cov / denominator
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, np.where(diagonal > 0.0, 1.0, 0.0))
    return pd.DataFrame(corr, index=labels, columns=labels)


def _matrix_with_total_row(
    columns: dict[str, pd.Series],
    *,
    total_row_name: str,
    total_values: dict[str, float],
) -> pd.DataFrame:
    matrix = pd.DataFrame(columns).fillna(0.0)
    if not matrix.empty:
        matrix.loc[total_row_name] = pd.Series(total_values).reindex(matrix.columns)
        matrix.index.name = "asset"
    return matrix


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
