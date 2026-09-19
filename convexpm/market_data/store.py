"""Local Parquet-backed normalized market data store."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from convexpm.utils.dates import DateLike, to_timestamp

MARKET_DATA_COLUMNS = ["date", "instrument_id", "value", "currency", "value_type", "source"]


class MarketDataStore:
    """Store normalized historical values in a single Parquet file."""

    columns = MARKET_DATA_COLUMNS

    def __init__(self, path: str | Path = "data/market_data.parquet") -> None:
        self.path = Path(path)

    def read_all(self) -> pd.DataFrame:
        """Return all stored market data."""
        if not self.path.exists():
            return pd.DataFrame(columns=MARKET_DATA_COLUMNS)
        return _normalize_market_data(pd.read_parquet(self.path))

    def get_history(
        self,
        instrument_id: str,
        *,
        start: DateLike | None = None,
        end: DateLike | None = None,
    ) -> pd.DataFrame:
        """Return sorted history for one instrument."""
        df = self.read_all()
        if df.empty:
            return df
        mask = df["instrument_id"] == instrument_id
        if start is not None:
            mask &= df["date"] >= to_timestamp(start)
        if end is not None:
            mask &= df["date"] <= to_timestamp(end)
        return df.loc[mask].sort_values("date").reset_index(drop=True)

    def get_value(self, instrument_id: str, on_date: DateLike | None = None, *, exact: bool = False) -> float:
        """Return the value for an instrument.

        By default the latest value on or before ``on_date`` is returned. Set
        ``exact=True`` to require an observation exactly on ``on_date``.
        """
        history = self.get_history(instrument_id)
        if history.empty:
            raise KeyError(f"No market data for instrument_id={instrument_id}")

        if on_date is None:
            row = history.iloc[-1]
            return float(row["value"])

        timestamp = to_timestamp(on_date)
        if exact:
            history = history.loc[history["date"] == timestamp]
        else:
            history = history.loc[history["date"] <= timestamp]
        if history.empty:
            mode = "on" if exact else "on or before"
            raise KeyError(f"No market data for {instrument_id} {mode} {timestamp.date()}")
        return float(history.iloc[-1]["value"])

    def latest(self, instrument_id: str) -> pd.Series | None:
        """Return the latest row for one instrument, or None when absent."""
        history = self.get_history(instrument_id)
        if history.empty:
            return None
        return history.iloc[-1]

    def append(self, df: pd.DataFrame) -> None:
        """Append rows using the same idempotent behavior as upsert."""
        self.upsert(df)

    def upsert(self, df: pd.DataFrame) -> None:
        """Add rows and deduplicate by instrument_id + date, keeping latest."""
        incoming = _normalize_market_data(df)
        if incoming.empty:
            return
        current = self.read_all()
        combined = pd.concat([current, incoming], ignore_index=True)
        combined = (
            _normalize_market_data(combined)
            .drop_duplicates(subset=["instrument_id", "date"], keep="last")
            .sort_values(["instrument_id", "date"])
            .reset_index(drop=True)
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(self.path, index=False)


def normalized_market_data_frame(rows: Iterable[dict[str, object]]) -> pd.DataFrame:
    """Create a normalized market data DataFrame from dictionaries."""
    return _normalize_market_data(pd.DataFrame(list(rows), columns=MARKET_DATA_COLUMNS))


def _normalize_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the market data schema."""
    if df.empty:
        return pd.DataFrame(columns=MARKET_DATA_COLUMNS)

    missing = [column for column in MARKET_DATA_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Market data missing required columns: {missing}")

    normalized = df.loc[:, MARKET_DATA_COLUMNS].copy()
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.normalize()
    normalized["instrument_id"] = normalized["instrument_id"].astype(str)
    normalized["value"] = pd.to_numeric(normalized["value"], errors="raise").astype(float)
    normalized["currency"] = normalized["currency"].astype(str).str.upper()
    normalized["value_type"] = normalized["value_type"].astype(str).str.lower()
    normalized["source"] = normalized["source"].astype(str).str.lower()

    if normalized["date"].isna().any():
        raise ValueError("Market data contains invalid dates")
    if normalized["value"].isna().any():
        raise ValueError("Market data contains invalid values")
    return normalized.sort_values(["instrument_id", "date"]).reset_index(drop=True)
