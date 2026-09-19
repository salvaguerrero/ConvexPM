"""Date normalization helpers used across ConvexPM."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TypeAlias

import pandas as pd

DateLike: TypeAlias = date | datetime | str | pd.Timestamp


def to_date(value: DateLike) -> date:
    """Convert a supported date-like value to a Python date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    timestamp = pd.to_datetime(value)
    if pd.isna(timestamp):
        raise ValueError(f"Cannot parse date value: {value!r}")
    return timestamp.date()


def to_timestamp(value: DateLike) -> pd.Timestamp:
    """Convert a supported date-like value to a normalized pandas Timestamp."""
    return pd.Timestamp(to_date(value))


def next_day(value: DateLike) -> date:
    """Return the calendar day after the given date."""
    return to_date(value) + timedelta(days=1)
