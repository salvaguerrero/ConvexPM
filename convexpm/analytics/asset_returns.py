"""Aligned instrument return histories for portfolio analytics."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from convexpm.market_data import MarketDataStore
from convexpm.utils.dates import DateLike, to_timestamp


def instrument_returns(
    market_data: MarketDataStore,
    instrument_ids: Iterable[str],
    *,
    start: DateLike | None = None,
    end: DateLike | None = None,
) -> pd.DataFrame:
    """Return aligned daily percentage returns for the requested instruments.

    Market history is used independently of portfolio ownership dates. V1 uses
    the intersection of valid return dates and never forward-fills prices.
    """
    ids = list(dict.fromkeys(str(item) for item in instrument_ids))
    if not ids:
        return pd.DataFrame()

    series: list[pd.Series] = []
    for instrument_id in ids:
        history = market_data.get_history(instrument_id, end=end)
        if history.empty:
            raise KeyError(f"No market data for instrument_id={instrument_id}")

        values = (
            history.loc[:, ["date", "value"]]
            .drop_duplicates(subset=["date"], keep="last")
            .sort_values("date")
            .set_index("date")["value"]
            .astype(float)
        )
        returns = values.pct_change(fill_method=None)
        if start is not None:
            returns = returns.loc[returns.index >= to_timestamp(start)]
        if end is not None:
            returns = returns.loc[returns.index <= to_timestamp(end)]
        returns.name = instrument_id
        series.append(returns)

    aligned = pd.concat(series, axis=1, join="inner").dropna(how="any")
    aligned.index = pd.DatetimeIndex(aligned.index).normalize()
    aligned = aligned.sort_index()
    aligned.attrs["observations"] = len(aligned)
    aligned.attrs["missing_data_policy"] = "intersection"
    return aligned
