"""Minimal ConvexPM workflow."""

from datetime import date

import pandas as pd

from convexpm import Instrument, InstrumentRegistry, MarketDataStore, Portfolio, Trade


registry = InstrumentRegistry("data/instruments.parquet", auto_load=False)
registry.add(
    Instrument(
        instrument_id="REP_MC",
        name="Repsol",
        instrument_type="equity",
        asset_class="equity",
        currency="EUR",
        data_source="yahoo",
        data_symbol="REP.MC",
    )
)

trades = [
    Trade(
        instrument_id="REP_MC",
        date=date(2026, 1, 1),
        side="BUY",
        quantity=10,
        amount=150,
        value_per_unit=15,
        currency="EUR",
    )
]

store = MarketDataStore("data/market_data.parquet")
store.upsert(
    pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=3, freq="D"),
            "instrument_id": "REP_MC",
            "value": [15.0, 15.5, 16.0],
            "currency": "EUR",
            "value_type": "close",
            "source": "example",
        }
    )
)

portfolio = Portfolio(
    name="Main",
    base_currency="EUR",
    trades=trades,
    registry=registry,
    market_data=store,
)

print(portfolio.nav_history())
print(portfolio.performance())
