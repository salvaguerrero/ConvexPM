"""Portfolio valuation using normalized market data."""

from __future__ import annotations

from datetime import date

import pandas as pd

from convexpm.instruments import InstrumentRegistry
from convexpm.market_data import MarketDataStore
from convexpm.portfolio.holdings import build_holdings
from convexpm.transactions import Trade
from convexpm.utils.dates import DateLike, to_date, to_timestamp


def value_holdings(
    holdings: dict[str, float],
    market_data: MarketDataStore,
    date: DateLike,
    *,
    registry: InstrumentRegistry | None = None,
    base_currency: str = "EUR",
    skip_missing_prices: bool = False,
) -> pd.DataFrame:
    """Value holdings at the latest available value on or before ``date``."""
    valuation_date = to_date(date)
    rows = []
    for instrument_id, quantity in holdings.items():
        try:
            value = market_data.get_value(instrument_id, valuation_date)
            latest_row = market_data.get_history(instrument_id, end=valuation_date).iloc[-1]
        except (KeyError, IndexError):
            if skip_missing_prices:
                continue
            raise
        currency = str(latest_row["currency"]).upper()
        fx_rate, fx_pair = fx_rate_to_base(
            currency,
            base_currency,
            market_data,
            valuation_date,
            skip_missing_prices=skip_missing_prices,
        )
        if fx_rate is None:
            continue
        instrument = registry.get(instrument_id) if registry is not None else None
        local_market_value = float(quantity) * float(value)
        rows.append(
            {
                "date": pd.Timestamp(valuation_date),
                "instrument_id": instrument_id,
                "quantity": float(quantity),
                "value": float(value),
                "local_market_value": local_market_value,
                "fx_rate": fx_rate,
                "fx_pair": fx_pair,
                "market_value": local_market_value * fx_rate,
                "currency": currency,
                "base_currency": base_currency.upper(),
                "asset_class": instrument.asset_class if instrument else None,
                "instrument_type": instrument.instrument_type if instrument else None,
                "name": instrument.name if instrument else instrument_id,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "date",
            "instrument_id",
            "quantity",
            "value",
            "local_market_value",
            "fx_rate",
            "fx_pair",
            "market_value",
            "currency",
            "base_currency",
            "asset_class",
            "instrument_type",
            "name",
        ],
    )


def fx_rate_to_base(
    currency: str,
    base_currency: str,
    market_data: MarketDataStore,
    date: DateLike,
    *,
    skip_missing_prices: bool = False,
) -> tuple[float | None, str | None]:
    """Return a conversion rate from ``currency`` to ``base_currency``."""
    from_currency = currency.upper()
    to_currency = base_currency.upper()
    if from_currency == to_currency:
        return 1.0, None

    valuation_date = to_date(date)
    direct_pair = f"{from_currency}{to_currency}"
    try:
        return market_data.get_value(direct_pair, valuation_date), direct_pair
    except KeyError:
        pass

    inverse_pair = f"{to_currency}{from_currency}"
    try:
        inverse_rate = market_data.get_value(inverse_pair, valuation_date)
    except KeyError:
        if skip_missing_prices:
            return None, None
        raise KeyError(f"No FX market data for {direct_pair} or {inverse_pair}")
    if inverse_rate == 0:
        raise ZeroDivisionError(f"FX rate for {inverse_pair} is zero")
    return 1.0 / inverse_rate, inverse_pair


def nav_on_date(
    trades: list[Trade],
    market_data: MarketDataStore,
    date: DateLike,
    *,
    registry: InstrumentRegistry | None = None,
    base_currency: str = "EUR",
    skip_missing_prices: bool = False,
) -> float:
    """Calculate portfolio NAV on one date."""
    holdings = build_holdings(trades, date, market_data=market_data)
    if not holdings:
        return 0.0
    valued = value_holdings(
        holdings,
        market_data,
        date,
        registry=registry,
        base_currency=base_currency,
        skip_missing_prices=skip_missing_prices,
    )
    return float(valued["market_value"].sum())


def nav_history(
    trades: list[Trade],
    market_data: MarketDataStore,
    *,
    start: DateLike | None = None,
    end: DateLike | None = None,
    registry: InstrumentRegistry | None = None,
    base_currency: str = "EUR",
    skip_missing_prices: bool = False,
) -> pd.DataFrame:
    """Reconstruct a daily NAV history from trades and normalized market data."""
    if not trades:
        return pd.DataFrame(columns=["date", "nav"])

    instrument_ids = {trade.instrument_id for trade in trades}
    all_market_data = market_data.read_all()
    relevant_market_data = all_market_data.loc[all_market_data["instrument_id"].isin(instrument_ids)]
    if relevant_market_data.empty:
        raise ValueError("Cannot build NAV history without market data for traded instruments")

    first_trade_date = min(trade.date for trade in trades)
    start_date = to_date(start) if start is not None else first_trade_date
    end_date = to_date(end) if end is not None else relevant_market_data["date"].max().date()
    if start_date > end_date:
        return pd.DataFrame(columns=["date", "nav"])

    rows = []
    for valuation_date in pd.date_range(to_timestamp(start_date), to_timestamp(end_date), freq="D"):
        day = valuation_date.date()
        rows.append(
            {
                "date": valuation_date.normalize(),
                "nav": nav_on_date(
                    trades,
                    market_data,
                    day,
                    registry=registry,
                    base_currency=base_currency,
                    skip_missing_prices=skip_missing_prices,
                ),
            }
        )
    return pd.DataFrame(rows, columns=["date", "nav"])
