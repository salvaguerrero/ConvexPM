"""Current portfolio market values and allocation weights."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from convexpm.portfolio.valuation import value_holdings
from convexpm.utils.dates import DateLike

if TYPE_CHECKING:
    from convexpm.portfolio.portfolio import Portfolio


def market_values(portfolio: "Portfolio", date: DateLike | None = None) -> pd.DataFrame:
    """Return current holdings valued at the requested date with weights."""
    valuation_date = date or portfolio._default_date()
    holdings = portfolio.holdings(valuation_date)
    if not holdings:
        return pd.DataFrame(
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
                "weight",
            ]
        )

    valued = value_holdings(
        holdings,
        portfolio.market_data,
        valuation_date,
        registry=portfolio.registry,
        base_currency=portfolio.base_currency,
    )
    total = float(valued["market_value"].sum())
    if abs(total) <= 1e-12:
        valued["weight"] = 0.0
    else:
        valued["weight"] = valued["market_value"] / total
    return valued.reset_index(drop=True)


def weights(portfolio: "Portfolio", date: DateLike | None = None) -> pd.Series:
    """Return current allocation weights indexed by instrument ID."""
    valued = market_values(portfolio, date)
    if valued.empty:
        return pd.Series(dtype=float, name="weight")
    result = valued.set_index("instrument_id")["weight"].astype(float)
    result.name = "weight"
    return result
