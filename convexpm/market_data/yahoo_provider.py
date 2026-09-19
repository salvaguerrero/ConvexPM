"""Yahoo Finance market data provider."""

from __future__ import annotations

from datetime import date
from typing import Any, Callable

import pandas as pd

from convexpm.instruments import Instrument
from convexpm.market_data.provider import MarketDataError
from convexpm.market_data.store import MARKET_DATA_COLUMNS, MarketDataStore
from convexpm.utils.dates import next_day


class YahooProvider:
    """Fetch daily Yahoo Finance history.

    By default ConvexPM uses Yahoo's daily ``Close`` column with
    ``value_type="close"``. Pass ``use_adjusted=True`` to use ``Adj Close``
    when available and label rows as ``adjusted_close``.
    """

    def __init__(
        self,
        *,
        use_adjusted: bool = False,
        ticker_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.use_adjusted = use_adjusted
        self._ticker_factory = ticker_factory

    def fetch_history(
        self,
        instrument: Instrument,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        """Fetch and normalize daily Yahoo history for one instrument."""
        symbol = instrument.data_symbol or instrument.instrument_id
        ticker = self._build_ticker(symbol)
        try:
            raw = ticker.history(start=start, end=end, auto_adjust=False)
        except TypeError:
            raw = ticker.history(start=start, end=end)
        except Exception as exc:
            raise MarketDataError(f"Yahoo fetch failed for {symbol}: {exc}") from exc

        if raw is None or raw.empty:
            raise MarketDataError(f"Yahoo returned no data for {symbol}")

        frame = raw.copy()
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)

        column = "Adj Close" if self.use_adjusted and "Adj Close" in frame.columns else "Close"
        if column not in frame.columns:
            raise MarketDataError(f"Yahoo response for {symbol} did not contain {column!r}")

        values = pd.to_numeric(frame[column], errors="coerce")
        normalized = pd.DataFrame(
            {
                "date": pd.to_datetime(frame.index).tz_localize(None).normalize(),
                "instrument_id": instrument.instrument_id,
                "value": values.to_numpy(dtype=float),
                "currency": instrument.currency,
                "value_type": "adjusted_close" if column == "Adj Close" else "close",
                "source": "yahoo",
            }
        )
        normalized = normalized.dropna(subset=["value"]).loc[:, MARKET_DATA_COLUMNS]
        if normalized.empty:
            raise MarketDataError(f"Yahoo returned no usable close values for {symbol}")
        return normalized.sort_values("date").reset_index(drop=True)

    def update_instrument(
        self,
        instrument: Instrument,
        store: MarketDataStore,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        """Fetch only missing rows for ``instrument`` and upsert them."""
        latest = store.latest(instrument.instrument_id)
        fetch_start = start
        if latest is not None:
            missing_start = next_day(latest["date"])
            fetch_start = max(fetch_start, missing_start) if fetch_start else missing_start
        if end is not None and fetch_start is not None and fetch_start > end:
            return pd.DataFrame(columns=MARKET_DATA_COLUMNS)
        df = self.fetch_history(instrument, start=fetch_start, end=end)
        store.upsert(df)
        return df

    def _build_ticker(self, symbol: str) -> Any:
        if self._ticker_factory is not None:
            return self._ticker_factory(symbol)
        try:
            import yfinance as yf
        except ImportError as exc:
            raise MarketDataError("Install yfinance to use YahooProvider") from exc
        return yf.Ticker(symbol)
