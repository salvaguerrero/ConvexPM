"""Scenario abstraction."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import pandas as pd

from convexpm.transactions import Trade


@dataclass(slots=True)
class Scenario:
    """A hypothetical portfolio produced by adding trades to a clone."""

    original_portfolio: object
    hypothetical_trades: list[Trade]

    @property
    def portfolio(self) -> object:
        """Return a cloned portfolio with hypothetical trades appended."""
        from convexpm.portfolio.portfolio import Portfolio

        return Portfolio(
            name=f"{self.original_portfolio.name} Scenario",
            base_currency=self.original_portfolio.base_currency,
            trades=deepcopy(self.original_portfolio.trades) + deepcopy(self.hypothetical_trades),
            registry=self.original_portfolio.registry,
            market_data=self.original_portfolio.market_data,
        )

    def compare(self) -> pd.DataFrame:
        """Compare current and scenario performance/risk metrics."""
        from convexpm.scenarios.comparison import compare_portfolios

        return compare_portfolios(self.original_portfolio, self.portfolio)
