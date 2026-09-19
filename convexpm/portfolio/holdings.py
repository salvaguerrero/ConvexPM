"""Internal holdings reconstruction from trades."""

from __future__ import annotations

from collections import defaultdict

from convexpm.market_data import MarketDataStore
from convexpm.transactions import Trade
from convexpm.utils.dates import DateLike, to_date


def trade_quantity(trade: Trade, market_data: MarketDataStore | None = None) -> float:
    """Return the unsigned quantity represented by a trade."""
    if trade.quantity is not None:
        return float(trade.quantity)
    if trade.amount is None:
        raise ValueError(f"Trade {trade.trade_id} has neither quantity nor amount")

    value_per_unit = trade.value_per_unit
    if value_per_unit is None and market_data is not None:
        value_per_unit = market_data.get_value(trade.instrument_id, trade.date)
    if value_per_unit is None:
        raise ValueError(
            f"Trade {trade.trade_id} needs quantity, value_per_unit, or market data at the trade date"
        )
    return float(trade.amount) / float(value_per_unit)


def signed_trade_quantity(trade: Trade, market_data: MarketDataStore | None = None) -> float:
    """Return positive quantity for BUY and negative quantity for SELL."""
    sign = 1.0 if trade.side == "BUY" else -1.0
    return sign * trade_quantity(trade, market_data=market_data)


def build_holdings(
    trades: list[Trade],
    date: DateLike | None = None,
    *,
    market_data: MarketDataStore | None = None,
) -> dict[str, float]:
    """Reconstruct holdings from all trades up to and including ``date``."""
    as_of = to_date(date) if date is not None else max((trade.date for trade in trades), default=None)
    holdings: defaultdict[str, float] = defaultdict(float)

    for trade in sorted(trades, key=lambda item: item.date):
        if as_of is not None and trade.date > as_of:
            continue
        holdings[trade.instrument_id] += signed_trade_quantity(trade, market_data=market_data)

    return {
        instrument_id: quantity
        for instrument_id, quantity in sorted(holdings.items())
        if abs(quantity) > 1e-12
    }
