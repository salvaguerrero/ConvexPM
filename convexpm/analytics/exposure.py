"""Exposure analytics."""

from __future__ import annotations

import pandas as pd

from convexpm.portfolio.valuation import value_holdings
from convexpm.utils.dates import DateLike


def exposure_by(portfolio: object, *, by: str = "asset_class", date: DateLike) -> pd.DataFrame:
    """Group current market value exposure by instrument field or metadata key."""
    holdings = portfolio.holdings(date)
    if not holdings:
        return pd.DataFrame(columns=[by, "market_value", "weight"])

    valued = value_holdings(
        holdings,
        portfolio.market_data,
        date,
        registry=portfolio.registry,
        base_currency=portfolio.base_currency,
    )

    def group_for_instrument(instrument_id: str) -> str:
        if by in valued.columns:
            raw = valued.loc[valued["instrument_id"] == instrument_id, by].iloc[0]
            return str(raw) if pd.notna(raw) and raw else "Unknown"
        instrument = portfolio.registry.get(instrument_id)
        raw = instrument.metadata.get(by)
        return str(raw) if raw else "Unknown"

    valued[by] = [group_for_instrument(instrument_id) for instrument_id in valued["instrument_id"]]
    grouped = valued.groupby(by, dropna=False, as_index=False)["market_value"].sum()
    total = grouped["market_value"].sum()
    grouped["weight"] = grouped["market_value"] / total if total else 0.0
    return grouped.sort_values("market_value", ascending=False).reset_index(drop=True)
