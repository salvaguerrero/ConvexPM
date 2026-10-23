"""Simple portfolio performance and risk views for interactive use."""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd

from convexpm.analytics.asset_returns import instrument_returns
from convexpm.analytics.current_risk import current_risk
from convexpm.analytics.returns import cash_settlement_parent_ids, is_external_capital_flow
from convexpm.portfolio.holdings import build_holdings
from convexpm.portfolio.valuation import value_holdings
from convexpm.utils.dates import DateLike
from convexpm.utils.terminal_table import styled_dataframe, styled_series


ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{10}$")


@dataclass(slots=True)
class PerformanceViewResult:
    """User-facing portfolio performance view."""

    summary: pd.DataFrame
    standalone_asset_returns: pd.DataFrame
    standalone_asset_volatility: pd.DataFrame
    standalone_asset_alpha: pd.DataFrame
    standalone_asset_beta: pd.DataFrame
    standalone_asset_drawdown: pd.DataFrame
    asset_weights: pd.DataFrame
    asset_net_inflows: pd.DataFrame
    asset_valuation: pd.DataFrame
    return_contribution: pd.DataFrame
    volatility_contribution: pd.DataFrame
    drawdown_contribution: pd.DataFrame
    metadata: dict[str, object]
    _portfolio: object
    _asset_returns: pd.DataFrame
    _weighted_returns: pd.DataFrame
    cumulative_return: pd.Series | None = None
    assets: pd.DataFrame | None = None
    yearly_return_contribution: pd.DataFrame | None = None

    def return_correlation(
        self,
        *,
        start: DateLike | None = None,
        end: DateLike | None = None,
        year: int | None = None,
    ) -> pd.DataFrame:
        """Return asset return correlation for the selected performance period."""
        returns = _filter_returns(self._asset_returns, start=start, end=end, year=year)
        return styled_dataframe(
            _compact_matrix_from_ids(self._portfolio, returns.corr().fillna(0.0)),
            "RETURN CORRELATION",
            subtitle=_correlation_date_range(returns),
            max_col_width=14,
            numeric_precision=2,
        )

    def volatility_correlation(
        self,
        *,
        start: DateLike | None = None,
        end: DateLike | None = None,
        year: int | None = None,
    ) -> pd.DataFrame:
        """Return correlation of absolute asset returns as a volatility proxy."""
        returns = _filter_returns(self._asset_returns, start=start, end=end, year=year).abs()
        return styled_dataframe(
            _compact_matrix_from_ids(self._portfolio, returns.corr().fillna(0.0)),
            "VOLATILITY CORRELATION",
            subtitle=_correlation_date_range(returns),
            max_col_width=14,
            numeric_precision=2,
        )


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
    benchmark: str | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
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
    asset_returns = _instrument_returns_union(portfolio, instrument_ids, start=start_ts, end=end_ts)
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
    standalone_asset_volatility = _standalone_asset_metric_by_period(
        portfolio,
        instrument_ids,
        asset_returns,
        start=start_ts,
        end=end_ts,
        metric=lambda returns: _annualized_volatility_from_returns(returns, periods_per_year),
    )
    benchmark_returns = _benchmark_returns(portfolio, benchmark, start=start_ts, end=end_ts)
    standalone_asset_alpha = _standalone_asset_alpha_beta_by_period(
        portfolio,
        instrument_ids,
        asset_returns,
        benchmark_returns,
        start=start_ts,
        end=end_ts,
        metric="alpha",
    )
    standalone_asset_beta = _standalone_asset_alpha_beta_by_period(
        portfolio,
        instrument_ids,
        asset_returns,
        benchmark_returns,
        start=start_ts,
        end=end_ts,
        metric="beta",
    )
    standalone_asset_drawdown = _standalone_asset_metric_by_period(
        portfolio,
        instrument_ids,
        asset_returns,
        start=start_ts,
        end=end_ts,
        metric=_max_drawdown_from_returns,
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
    asset_net_inflows = _asset_net_inflows_by_period(portfolio, instrument_ids, start=start_ts, end=end_ts)
    asset_valuation = _asset_valuation_by_period(portfolio, instrument_ids, start=start_ts, end=end_ts)
    return_contribution = _yearly_return_contribution(daily_contribution)
    volatility_contribution = _volatility_contribution_by_period(
        daily_contribution,
        periods_per_year=periods_per_year,
    )
    drawdown_contribution = _drawdown_contribution_by_period(daily_contribution)
    cash_flows = _capital_flows(portfolio, start=start_ts, end=end_ts)
    nav_return = _nav_return(portfolio, start=start_ts, end=end_ts)
    nav_cagr = _annualize_return(nav_return, start=start_ts, end=end_ts)
    summary = _performance_summary_by_period(
        portfolio,
        daily_portfolio_return,
        benchmark_returns,
        start=start_ts,
        end=end_ts,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )
    metadata = {
        "analysis_start": start_ts.date(),
        "analysis_end": end_ts.date(),
        "benchmark": benchmark,
        "performance_basis": "historical holdings + chain-linked asset contributions",
        "external_cash_flows": "excluded from return through historical weights",
        "return_contribution_matrix": "asset rows, calendar-year columns plus Total; final row is portfolio return",
        "standalone_asset_return_matrix": "asset rows, calendar-year columns plus Total; values are base-currency returns while held, ignoring portfolio weight",
        "asset_weight_matrix": "asset rows, year-end columns plus Total; final row is portfolio weight",
        "asset_net_inflow_matrix": "asset rows, calendar-year columns plus Total; buys add cash invested and sells subtract cash proceeds",
        "asset_valuation_matrix": "asset rows, year-end columns plus Total; values are market value in portfolio base currency",
    }
    return PerformanceViewResult(
        summary=styled_dataframe(_performance_summary_display(summary), "PERFORMANCE SUMMARY"),
        standalone_asset_returns=styled_dataframe(
            _display_index(portfolio, _percentage_matrix(standalone_asset_returns)),
            "STANDALONE ASSET RETURNS",
        ),
        standalone_asset_volatility=styled_dataframe(
            _display_index(portfolio, _percentage_matrix(standalone_asset_volatility)),
            "STANDALONE ASSET VOLATILITY",
        ),
        standalone_asset_alpha=styled_dataframe(
            _display_index(portfolio, _percentage_matrix(standalone_asset_alpha)),
            "STANDALONE ASSET ALPHA",
        ),
        standalone_asset_beta=styled_dataframe(
            _display_index(portfolio, standalone_asset_beta),
            "STANDALONE ASSET BETA",
        ),
        standalone_asset_drawdown=styled_dataframe(
            _display_index(portfolio, _percentage_matrix(standalone_asset_drawdown)),
            "STANDALONE ASSET DRAWDOWN",
        ),
        asset_weights=styled_dataframe(
            _display_index(portfolio, _percentage_matrix(asset_weights)),
            "ASSET WEIGHTS",
        ),
        asset_net_inflows=styled_dataframe(
            _display_index(portfolio, asset_net_inflows),
            "ASSET NET INFLOWS",
        ),
        asset_valuation=styled_dataframe(
            _display_index(portfolio, asset_valuation),
            "ASSET VALUATION",
        ),
        return_contribution=styled_dataframe(
            _display_index(portfolio, _percentage_matrix(return_contribution)),
            "RETURN CONTRIBUTION",
        ),
        volatility_contribution=styled_dataframe(
            _display_index(portfolio, _percentage_matrix(volatility_contribution)),
            "VOLATILITY CONTRIBUTION",
        ),
        drawdown_contribution=styled_dataframe(
            _display_index(portfolio, _percentage_matrix(drawdown_contribution)),
            "DRAWDOWN CONTRIBUTION",
        ),
        metadata=metadata,
        _portfolio=portfolio,
        _asset_returns=asset_returns,
        _weighted_returns=daily_contribution,
        cumulative_return=_percentage_series(cumulative),
        assets=styled_dataframe(
            _display_index(
                portfolio,
                _performance_percentage_columns(
                    assets,
                    ["current_weight", "asset_return", "return_contribution"],
                ),
            ),
            "PERFORMANCE ASSETS",
        ),
        yearly_return_contribution=styled_dataframe(
            _display_index(portfolio, _percentage_matrix(return_contribution)),
            "RETURN CONTRIBUTION",
        ),
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
        summary=styled_series(_risk_summary_display(summary), "RISK SUMMARY"),
        assets=styled_dataframe(
            _display_index(
                portfolio,
                _risk_assets_display(assets),
            ),
            "RISK ASSETS",
        ),
        asset_volatility=styled_dataframe(
            _display_index(portfolio, _percentage_matrix(risk_tables["asset_volatility"])),
            "ASSET VOLATILITY",
        ),
        volatility_contribution=styled_dataframe(
            _display_index(portfolio, _percentage_matrix(risk_tables["volatility_contribution"])),
            "VOLATILITY CONTRIBUTION",
        ),
        asset_sharpe=styled_dataframe(
            _display_index(portfolio, risk_tables["asset_sharpe"]),
            "ASSET SHARPE",
        ),
        sharpe_contribution=styled_dataframe(
            _display_index(portfolio, risk_tables["sharpe_contribution"]),
            "SHARPE CONTRIBUTION",
        ),
        yearly_risk_contribution=styled_dataframe(
            _display_index(portfolio, _percentage_matrix(risk_tables["volatility_contribution"])),
            "YEARLY RISK CONTRIBUTION",
        ),
        yearly_portfolio_volatility=styled_series(
            _percentage_series(risk_tables["portfolio_volatility"]),
            "YEARLY PORTFOLIO VOLATILITY",
        ),
        asset_correlation=styled_dataframe(
            _compact_matrix_from_ids(portfolio, current.correlation),
            "ASSET CORRELATION",
            max_col_width=14,
        ),
        weighted_correlation=styled_dataframe(
            _compact_matrix_from_ids(portfolio, weighted_correlation),
            "WEIGHTED CORRELATION",
            max_col_width=14,
        ),
        correlation=styled_dataframe(
            _compact_matrix_from_ids(portfolio, current.correlation),
            "ASSET CORRELATION",
            max_col_width=14,
        ),
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


def _performance_summary_display(summary: pd.DataFrame) -> pd.DataFrame:
    percent_rows = {
        "portfolio_return",
        "cagr",
        "volatility",
        "max_drawdown",
        "benchmark_return",
        "benchmark_volatility",
        "benchmark_max_drawdown",
        "alpha",
    }
    result = summary.copy()
    renamed = {}
    for row in percent_rows:
        if row in result.index:
            result.loc[row] = pd.to_numeric(result.loc[row], errors="coerce") * 100.0
            renamed[row] = f"{row} (%)"
    result = result.rename(index=renamed)
    ordered_rows = [
        # 1. RETURNS
        "portfolio_return (%)",
        "benchmark_return (%)",
        "cagr (%)",
        "",

        # 2. RISK METRICS
        "sharpe",
        "benchmark_sharpe",
        "volatility (%)",
        "benchmark_volatility (%)",
        "max_drawdown (%)",
        "benchmark_max_drawdown (%)",
        "",

        # 3. RELATIVE PERFORMANCE
        "alpha (%)",
        "beta",
        "",

        # 4. PORTFOLIO VALUATION
        "start_nav",
        "end_nav",
        "delta_nav",
        "capital_gain",
        "",

        # 5. CAPITAL FLOWS
        "capital_inflows",
        "capital_outflows",
        "net_capital_flow",
        "",

        # 6. CASH-FLOW-INCLUSIVE METRICS
        "portfolio_return_including_cash_flows",
        "cagr_including_cash_flows",
    ]
    remaining_rows = [row for row in result.index if row not in set(ordered_rows)]
    return result.reindex([*ordered_rows, *remaining_rows])


def _risk_summary_display(summary: pd.Series) -> pd.Series:
    percent_keys = {
        "annualized_volatility",
        "max_drawdown",
    }
    result = summary.copy()
    renamed = {}
    for key in percent_keys:
        if key in result.index:
            result.loc[key] = float(result.loc[key]) * 100.0
            renamed[key] = f"{key} (%)"
    return result.rename(index=renamed)


def _risk_assets_display(df: pd.DataFrame) -> pd.DataFrame:
    return _performance_percentage_columns(
        df,
        ["weight", "asset_volatility", "risk_contribution"],
    )


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


def _correlation_date_range(returns: pd.DataFrame) -> str:
    """Describe the observation dates actually included in a correlation."""
    if returns.empty:
        return "No observations in the selected date range"
    first, last = returns.index.min(), returns.index.max()
    return f"{first.day}/{first:%b/%Y} - {last.day}/{last:%b/%Y}"


def _filter_returns(
    returns: pd.DataFrame,
    *,
    start: DateLike | None = None,
    end: DateLike | None = None,
    year: int | None = None,
) -> pd.DataFrame:
    result = returns.copy()
    if year is not None:
        result = result.loc[result.index.year == year]
    if start is not None:
        result = result.loc[result.index >= pd.Timestamp(start).normalize()]
    if end is not None:
        result = result.loc[result.index <= pd.Timestamp(end).normalize()]
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


def _compact_matrix_from_ids(portfolio: object, df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    row_labels = _compact_labels(portfolio, list(result.index))
    column_labels = _compact_labels(portfolio, list(result.columns))
    result.index = row_labels
    result.columns = column_labels
    return result


def _compact_labels(portfolio: object, values: list[object]) -> list[object]:
    raw = [_compact_label(portfolio, value) for value in values]
    counts: dict[str, int] = {}
    labels = []
    for label in raw:
        if not isinstance(label, str):
            labels.append(label)
            continue
        count = counts.get(label, 0) + 1
        counts[label] = count
        labels.append(label if count == 1 else f"{label}_{count}")
    return labels


def _compact_label(portfolio: object, value: object) -> object:
    if not isinstance(value, str):
        return value
    if value.startswith("Portfolio "):
        return value
    try:
        instrument = portfolio.registry.get(value)
    except KeyError:
        return value
    if instrument.data_symbol:
        return instrument.data_symbol
    if not ISIN_RE.match(instrument.instrument_id):
        return instrument.instrument_id
    words = [word for word in instrument.name.replace(",", "").split() if word]
    if not words:
        return instrument.instrument_id[:12]
    return " ".join(words[:2])[:14]


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


def _performance_summary_by_period(
    portfolio: object,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    risk_free_rate: float,
    periods_per_year: int,
) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for label, period_start, period_end in _calendar_periods(start, end):
        period_returns = portfolio_returns.loc[
            (portfolio_returns.index >= period_start) & (portfolio_returns.index <= period_end)
        ]
        period_benchmark = _benchmark_period(benchmark_returns, period_start=period_start, period_end=period_end)
        values = _performance_summary_for_period(
            period_returns,
            period_benchmark,
            start=period_start,
            end=period_end,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )
        flows = _capital_flows(portfolio, start=period_start, end=period_end)
        values.update(flows)
        nav_start = pd.Timestamp(year=int(label), month=1, day=1)
        nav_change_flows = _capital_flows_after_start_nav(portfolio, start=nav_start, end=period_end)
        values.update(
            _nav_change_metrics(
                portfolio,
                start=nav_start,
                end=period_end,
                net_capital_flow=nav_change_flows["net_capital_flow"],
            )
        )
        columns[label] = pd.Series(values, dtype=float)

    total_values = _performance_summary_for_period(
        portfolio_returns,
        benchmark_returns,
        start=start,
        end=end,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )
    total_flows = _capital_flows(portfolio, start=start, end=end)
    total_values.update(total_flows)
    nav_return = _nav_return(portfolio, start=start, end=end)
    total_values.update(
        {
            "portfolio_return_including_cash_flows": nav_return,
            "cagr_including_cash_flows": _annualize_return(
                nav_return,
                start=start,
                end=end,
            ),
        }
    )
    total_nav_change_flows = _capital_flows_after_start_nav(portfolio, start=start, end=end)
    total_values.update(
        _nav_change_metrics(
            portfolio,
            start=start,
            end=end,
            net_capital_flow=total_nav_change_flows["net_capital_flow"],
        )
    )
    columns["Total"] = pd.Series(total_values, dtype=float)
    return pd.DataFrame(columns).fillna(np.nan)


def _performance_summary_for_period(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    risk_free_rate: float,
    periods_per_year: int,
) -> dict[str, float]:
    clean = portfolio_returns.dropna().astype(float)
    portfolio_return = _chain_returns(clean)
    benchmark_aligned = _align_benchmark(clean, benchmark_returns)
    benchmark_return = _chain_returns(benchmark_aligned) if benchmark_aligned is not None else np.nan
    benchmark_volatility = (
        _annualized_volatility_from_returns(benchmark_aligned, periods_per_year)
        if benchmark_aligned is not None
        else np.nan
    )
    benchmark_sharpe = (
        _asset_sharpe_from_returns(
            benchmark_aligned,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )
        if benchmark_aligned is not None
        else np.nan
    )
    benchmark_max_drawdown = _max_drawdown_from_returns(benchmark_aligned) if benchmark_aligned is not None else np.nan
    alpha, beta = _alpha_beta(clean, benchmark_aligned)
    return {
        "portfolio_return": portfolio_return,
        "cagr": _annualize_return(portfolio_return, start=start, end=end),
        "volatility": _annualized_volatility_from_returns(clean, periods_per_year),
        "sharpe": _asset_sharpe_from_returns(
            clean,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        ),
        "max_drawdown": _max_drawdown_from_returns(clean),
        "benchmark_return": benchmark_return,
        "benchmark_sharpe": benchmark_sharpe,
        "benchmark_volatility": benchmark_volatility,
        "benchmark_max_drawdown": benchmark_max_drawdown,
        "alpha": alpha,
        "beta": beta,
    }


def _calendar_periods(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    periods = []
    for year in range(start.year, end.year + 1):
        period_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        period_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        if period_start <= period_end:
            periods.append((str(year), period_start, period_end))
    return periods


def _chain_returns(returns: pd.Series) -> float:
    clean = returns.dropna().astype(float)
    if clean.empty:
        return 0.0
    return float((1.0 + clean).prod() - 1.0)


def _annualized_volatility_from_returns(returns: pd.Series, periods_per_year: int) -> float:
    clean = returns.dropna().astype(float)
    if len(clean) < 2:
        return 0.0
    return float(clean.std(ddof=1) * np.sqrt(periods_per_year))


def _benchmark_period(
    benchmark_returns: pd.Series | None,
    *,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> pd.Series | None:
    if benchmark_returns is None:
        return None
    return benchmark_returns.loc[
        (benchmark_returns.index >= period_start) & (benchmark_returns.index <= period_end)
    ]


def _align_benchmark(
    returns: pd.Series,
    benchmark_returns: pd.Series | None,
) -> pd.Series | None:
    if benchmark_returns is None:
        return None
    aligned = pd.concat(
        [returns.rename("portfolio"), benchmark_returns.rename("benchmark")],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    return aligned["benchmark"]


def _alpha_beta(returns: pd.Series, benchmark_returns: pd.Series | None) -> tuple[float, float]:
    if benchmark_returns is None:
        return np.nan, np.nan
    aligned = pd.concat(
        [returns.rename("asset"), benchmark_returns.rename("benchmark")],
        axis=1,
        join="inner",
    ).dropna()
    if len(aligned) < 2:
        return np.nan, np.nan
    benchmark_variance = float(aligned["benchmark"].var(ddof=1))
    if benchmark_variance <= 1e-15 or pd.isna(benchmark_variance):
        return np.nan, np.nan
    beta = float(aligned["asset"].cov(aligned["benchmark"]) / benchmark_variance)
    alpha = _chain_returns(aligned["asset"]) - beta * _chain_returns(aligned["benchmark"])
    return alpha, beta


def _benchmark_returns(
    portfolio: object,
    benchmark: str | None,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series | None:
    benchmark_id = _resolve_instrument_id(portfolio, benchmark) if benchmark else None
    if benchmark_id is None:
        return None
    histories = _price_histories(portfolio)
    returns = _base_currency_returns_for_instrument(
        portfolio,
        histories,
        benchmark_id,
        start=start,
        end=end,
    )
    return returns.rename("benchmark")


def _resolve_instrument_id(portfolio: object, value: str | None) -> str | None:
    if value is None:
        return None
    needle = str(value).strip()
    if not needle:
        return None
    if needle in portfolio.registry:
        return needle
    lowered = needle.lower()
    for instrument in portfolio.registry.all():
        if instrument.instrument_id.lower() == lowered:
            return instrument.instrument_id
        if instrument.name.lower() == lowered:
            return instrument.instrument_id
        if instrument.data_symbol and instrument.data_symbol.lower() == lowered:
            return instrument.instrument_id
    raise KeyError(f"Unknown benchmark instrument: {value}")


def _volatility_contribution_by_period(
    daily_contribution: pd.DataFrame,
    *,
    periods_per_year: int,
) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for label, group in _period_groups(daily_contribution):
        columns[label] = _volatility_contribution_for_period(group, periods_per_year=periods_per_year)
    columns["Total"] = _volatility_contribution_for_period(daily_contribution, periods_per_year=periods_per_year)
    result = pd.DataFrame(columns).fillna(0.0)
    result.index.name = "asset"
    return result


def _volatility_contribution_for_period(
    daily_contribution: pd.DataFrame,
    *,
    periods_per_year: int,
) -> pd.Series:
    if len(daily_contribution) < 2:
        values = pd.Series(0.0, index=daily_contribution.columns, dtype=float)
        values.loc["Portfolio Volatility"] = 0.0
        return values
    covariance = daily_contribution.cov()
    portfolio_returns = daily_contribution.sum(axis=1)
    portfolio_volatility = _annualized_volatility_from_returns(portfolio_returns, periods_per_year)
    if portfolio_volatility <= 1e-15:
        values = pd.Series(0.0, index=daily_contribution.columns, dtype=float)
    else:
        marginal = covariance.sum(axis=1) * periods_per_year / portfolio_volatility
        component = marginal / portfolio_volatility
        values = component.reindex(daily_contribution.columns).fillna(0.0)
    values.loc["Portfolio Volatility"] = portfolio_volatility
    return values


def _drawdown_contribution_by_period(daily_contribution: pd.DataFrame) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for label, group in _period_groups(daily_contribution):
        columns[label] = _drawdown_contribution_for_period(group)
    columns["Total"] = _drawdown_contribution_for_period(daily_contribution)
    result = pd.DataFrame(columns).fillna(0.0)
    result.index.name = "asset"
    return result


def _drawdown_contribution_for_period(daily_contribution: pd.DataFrame) -> pd.Series:
    if daily_contribution.empty:
        values = pd.Series(0.0, index=daily_contribution.columns, dtype=float)
        values.loc["Portfolio Drawdown"] = 0.0
        return values
    portfolio_returns = daily_contribution.sum(axis=1)
    wealth = (1.0 + portfolio_returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    trough = drawdown.idxmin()
    if pd.isna(trough) or float(drawdown.loc[trough]) >= 0.0:
        values = pd.Series(0.0, index=daily_contribution.columns, dtype=float)
        values.loc["Portfolio Drawdown"] = 0.0
        return values
    peak = wealth.loc[:trough].idxmax()
    window = daily_contribution.loc[(daily_contribution.index > peak) & (daily_contribution.index <= trough)]
    values = _chain_link_contribution(window)
    values.loc["Portfolio Drawdown"] = float(drawdown.loc[trough])
    return values


def _period_groups(daily_returns: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [
        (str(year), group)
        for year, group in daily_returns.groupby(daily_returns.index.year, sort=True)
    ]


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


def _standalone_asset_metric_by_period(
    portfolio: object,
    instrument_ids: list[str],
    returns: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    metric: object,
) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for year in range(start.year, end.year + 1):
        period_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        period_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        period_returns = returns.loc[(returns.index >= period_start) & (returns.index <= period_end)]
        columns[str(year)] = _standalone_asset_metric_for_period(
            portfolio,
            instrument_ids,
            period_returns,
            metric=metric,
        )
    columns["Total"] = _standalone_asset_metric_for_period(
        portfolio,
        instrument_ids,
        returns,
        metric=metric,
    )
    result = pd.DataFrame(columns).fillna(0.0)
    result.index.name = "asset"
    return result


def _standalone_asset_metric_for_period(
    portfolio: object,
    instrument_ids: list[str],
    returns: pd.DataFrame,
    *,
    metric: object,
) -> pd.Series:
    values = {}
    for instrument_id, held_returns in _held_asset_returns(portfolio, instrument_ids, returns).items():
        values[instrument_id] = float(metric(pd.Series(held_returns, dtype=float))) if held_returns else 0.0
    return pd.Series(values, dtype=float)


def _standalone_asset_alpha_beta_by_period(
    portfolio: object,
    instrument_ids: list[str],
    returns: pd.DataFrame,
    benchmark_returns: pd.Series | None,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    metric: str,
) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for year in range(start.year, end.year + 1):
        period_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        period_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        period_returns = returns.loc[(returns.index >= period_start) & (returns.index <= period_end)]
        period_benchmark = _benchmark_period(benchmark_returns, period_start=period_start, period_end=period_end)
        columns[str(year)] = _standalone_asset_alpha_beta_for_period(
            portfolio,
            instrument_ids,
            period_returns,
            period_benchmark,
            metric=metric,
        )
    columns["Total"] = _standalone_asset_alpha_beta_for_period(
        portfolio,
        instrument_ids,
        returns,
        benchmark_returns,
        metric=metric,
    )
    result = pd.DataFrame(columns)
    result.index.name = "asset"
    return result


def _standalone_asset_alpha_beta_for_period(
    portfolio: object,
    instrument_ids: list[str],
    returns: pd.DataFrame,
    benchmark_returns: pd.Series | None,
    *,
    metric: str,
) -> pd.Series:
    values = {}
    held = _held_asset_return_series(portfolio, instrument_ids, returns)
    for instrument_id, asset_returns in held.items():
        alpha, beta = _alpha_beta(asset_returns, benchmark_returns)
        values[instrument_id] = alpha if metric == "alpha" else beta
    return pd.Series(values, dtype=float)


def _standalone_asset_returns_for_period(
    portfolio: object,
    instrument_ids: list[str],
    returns: pd.DataFrame,
) -> pd.Series:
    held_returns = _held_asset_returns(portfolio, instrument_ids, returns)
    values = {
        instrument_id: float(pd.Series(asset_returns).add(1.0).prod() - 1.0)
        if asset_returns
        else 0.0
        for instrument_id, asset_returns in held_returns.items()
    }
    return pd.Series(values, dtype=float)


def _held_asset_returns(
    portfolio: object,
    instrument_ids: list[str],
    returns: pd.DataFrame,
) -> dict[str, list[float]]:
    held_returns = {instrument_id: [] for instrument_id in instrument_ids}
    if returns.empty:
        return held_returns
    for return_date, row in returns.iterrows():
        previous_date = (pd.Timestamp(return_date) - pd.Timedelta(days=1)).date()
        holdings = build_holdings(portfolio.trades, previous_date, market_data=portfolio.market_data)
        for instrument_id in instrument_ids:
            if abs(float(holdings.get(instrument_id, 0.0))) > 1e-12:
                held_returns[instrument_id].append(float(row.get(instrument_id, 0.0)))
    return held_returns


def _held_asset_return_series(
    portfolio: object,
    instrument_ids: list[str],
    returns: pd.DataFrame,
) -> dict[str, pd.Series]:
    values = {instrument_id: [] for instrument_id in instrument_ids}
    indexes = {instrument_id: [] for instrument_id in instrument_ids}
    if returns.empty:
        return {instrument_id: pd.Series(dtype=float) for instrument_id in instrument_ids}
    for return_date, row in returns.iterrows():
        previous_date = (pd.Timestamp(return_date) - pd.Timedelta(days=1)).date()
        holdings = build_holdings(portfolio.trades, previous_date, market_data=portfolio.market_data)
        for instrument_id in instrument_ids:
            if abs(float(holdings.get(instrument_id, 0.0))) > 1e-12:
                indexes[instrument_id].append(return_date)
                values[instrument_id].append(float(row.get(instrument_id, 0.0)))
    return {
        instrument_id: pd.Series(values[instrument_id], index=pd.DatetimeIndex(indexes[instrument_id]), dtype=float)
        for instrument_id in instrument_ids
    }


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


def _asset_net_inflows_by_period(
    portfolio: object,
    instrument_ids: list[str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for year in range(start.year, end.year + 1):
        period_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        period_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        if period_end < period_start:
            continue
        columns[str(year)] = _asset_net_inflows_for_period(
            portfolio,
            instrument_ids,
            start=period_start,
            end=period_end,
        )
    columns["Total"] = _asset_net_inflows_for_period(portfolio, instrument_ids, start=start, end=end)
    result = pd.DataFrame(columns).fillna(0.0)
    result.index.name = "asset"
    return result


def _asset_net_inflows_for_period(
    portfolio: object,
    instrument_ids: list[str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    values = pd.Series(0.0, index=pd.Index(instrument_ids, name="asset"), dtype=float)
    for trade in portfolio.trades:
        trade_date = pd.Timestamp(trade.date).normalize()
        if trade_date < start or trade_date > end or trade.instrument_id not in values.index:
            continue
        amount = _trade_amount_in_base(portfolio, trade)
        if amount is None:
            continue
        fees = _trade_fees_in_base(portfolio, trade)
        if trade.side == "BUY":
            values.loc[trade.instrument_id] += amount + fees
        else:
            values.loc[trade.instrument_id] -= max(amount - fees, 0.0)
    values.loc["Portfolio Net Inflow"] = float(values.sum())
    return values


def _asset_valuation_by_period(
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
        columns[str(year)] = _asset_valuation_on_date(portfolio, instrument_ids, year_end)
    columns["Total"] = _asset_valuation_on_date(portfolio, instrument_ids, end)
    result = pd.DataFrame(columns).fillna(0.0)
    result.index.name = "asset"
    return result


def _asset_valuation_on_date(
    portfolio: object,
    instrument_ids: list[str],
    date: pd.Timestamp,
) -> pd.Series:
    holdings = build_holdings(portfolio.trades, date.date(), market_data=portfolio.market_data)
    if holdings:
        valued = value_holdings(
            holdings,
            portfolio.market_data,
            date.date(),
            registry=portfolio.registry,
            base_currency=portfolio.base_currency,
        )
        values = valued.set_index("instrument_id")["market_value"].reindex(instrument_ids).fillna(0.0)
    else:
        values = pd.Series(0.0, index=pd.Index(instrument_ids, name="asset"), dtype=float)
    values.loc["Portfolio Valuation"] = float(values.sum())
    return values


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


def _capital_flows_after_start_nav(portfolio: object, *, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, float]:
    """Return flows after the displayed start NAV snapshot.

    Trade dates have no intraday timestamp, and portfolio.nav(start) includes
    trades dated on start. Same-day flows are therefore already inside
    start_nav and must not be subtracted again when reconciling capital_gain.
    """
    return _capital_flows(portfolio, start=start + pd.Timedelta(days=1), end=end)


def _nav_change_metrics(
    portfolio: object,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    net_capital_flow: float,
) -> dict[str, float]:
    start_nav = float(portfolio.nav(start.date()))
    end_nav = float(portfolio.nav(end.date()))
    delta_nav = end_nav - start_nav
    return {
        "start_nav": start_nav,
        "end_nav": end_nav,
        "delta_nav": delta_nav,
        "capital_gain": delta_nav - float(net_capital_flow),
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


def _trade_fees_in_base(portfolio: object, trade: object) -> float:
    fees = float(getattr(trade, "fees", 0.0) or 0.0)
    if fees == 0.0:
        return 0.0
    currency = (trade.currency or portfolio.base_currency).upper()
    if currency == portfolio.base_currency:
        return fees
    try:
        from convexpm.portfolio.valuation import fx_rate_to_base

        fx_rate, _ = fx_rate_to_base(currency, portfolio.base_currency, portfolio.market_data, trade.date)
    except KeyError:
        return fees
    return fees * float(fx_rate)


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
