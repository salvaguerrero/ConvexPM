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
) -> pd.DataFrame:
    """Value holdings at the latest available value on or before ``date``."""
    valuation_date = to_date(date)
    rows = []
    for instrument_id, quantity in holdings.items():
        value = market_data.get_value(instrument_id, valuation_date)
        latest_row = market_data.get_history(instrument_id, end=valuation_date).iloc[-1]
        currency = str(latest_row["currency"]).upper()
        if currency != base_currency.upper():
            raise NotImplementedError(
                f"FX conversion is not implemented yet ({instrument_id}: {currency} -> {base_currency})"
            )
        instrument = registry.get(instrument_id) if registry is not None else None
        rows.append(
            {
                "date": pd.Timestamp(valuation_date),
                "instrument_id": instrument_id,
                "quantity": float(quantity),
                "value": float(value),
                "market_value": float(quantity) * float(value),
                "currency": currency,
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
            "market_value",
            "currency",
            "asset_class",
            "instrument_type",
            "name",
        ],
    )


def nav_on_date(
    trades: list[Trade],
    market_data: MarketDataStore,
    date: DateLike,
    *,
    registry: InstrumentRegistry | None = None,
    base_currency: str = "EUR",
) -> float:
    """Calculate portfolio NAV on one date."""
    holdings = build_holdings(trades, date, market_data=market_data)
    if not holdings:
        return 0.0
    valued = value_holdings(holdings, market_data, date, registry=registry, base_currency=base_currency)
    return float(valued["market_value"].sum())


def nav_history(
    trades: list[Trade],
    market_data: MarketDataStore,
    *,
    start: DateLike | None = None,
    end: DateLike | None = None,
    registry: InstrumentRegistry | None = None,
    base_currency: str = "EUR",
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
                ),
            }
        )
    return pd.DataFrame(rows, columns=["date", "nav"])
