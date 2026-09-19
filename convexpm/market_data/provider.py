"""Market data provider interfaces and update orchestration."""

from __future__ import annotations

from datetime import date
from typing import Mapping, Protocol

import pandas as pd

from convexpm.instruments import Instrument, InstrumentRegistry
from convexpm.market_data.store import MarketDataStore
from convexpm.utils.dates import next_day, to_date


class MarketDataError(RuntimeError):
    """Raised when market data cannot be fetched or normalized."""


class MarketDataProvider(Protocol):
    """Provider that returns normalized historical market values."""

    def fetch_history(
        self,
        instrument: Instrument,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        """Fetch normalized market data for one instrument."""
        ...


class MarketDataUpdater:
    """Fetch only missing history for registered instruments and upsert it."""

    def __init__(
        self,
        registry: InstrumentRegistry,
        store: MarketDataStore,
        providers: Mapping[str, MarketDataProvider],
    ) -> None:
        self.registry = registry
        self.store = store
        self.providers = {key.lower(): provider for key, provider in providers.items()}

    def update_instrument(
        self,
        instrument_id: str,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        """Update one instrument and return the rows that were fetched."""
        instrument = self.registry.get(instrument_id)
        provider = self.providers.get(instrument.data_source)
        if provider is None:
            raise MarketDataError(f"No provider configured for data_source={instrument.data_source!r}")

        latest = self.store.latest(instrument_id)
        fetch_start = start
        if latest is not None:
            latest_date = to_date(latest["date"])
            missing_start = next_day(latest_date)
            fetch_start = max(fetch_start, missing_start) if fetch_start else missing_start

        if end is not None and fetch_start is not None and fetch_start > end:
            return pd.DataFrame(columns=self.store.columns)

        df = provider.fetch_history(instrument, start=fetch_start, end=end)
        if not df.empty:
            self.store.upsert(df)
        return df
