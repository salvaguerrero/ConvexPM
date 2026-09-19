"""Portfolio orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from convexpm.analytics.exposure import exposure_by
from convexpm.analytics.performance import performance_summary
from convexpm.analytics.risk import risk_summary
from convexpm.instruments import InstrumentRegistry
from convexpm.market_data import MarketDataStore
from convexpm.portfolio.holdings import build_holdings
from convexpm.portfolio.valuation import nav_history as build_nav_history
from convexpm.portfolio.valuation import nav_on_date
from convexpm.scenarios.scenario import Scenario
from convexpm.transactions import Trade
from convexpm.utils.dates import DateLike, to_date


@dataclass
class Portfolio:
    """A portfolio is trades plus a registry and normalized market data."""

    name: str
    base_currency: str
    trades: list[Trade] = field(default_factory=list)
    registry: InstrumentRegistry | None = None
    market_data: MarketDataStore | None = None

    def __post_init__(self) -> None:
        self.base_currency = self.base_currency.upper()
        self.trades = list(self.trades)
        if self.registry is None:
            self.registry = InstrumentRegistry(auto_load=False)
        if self.market_data is None:
            self.market_data = MarketDataStore()

    def holdings(self, date: DateLike | None = None) -> dict[str, float]:
        """Return derived holdings at ``date``."""
        return build_holdings(
            self.trades,
            date or self._default_date(),
            market_data=self.market_data,
        )

    def nav(self, date: DateLike | None = None) -> float:
        """Return portfolio NAV at ``date``."""
        return nav_on_date(
            self.trades,
            self.market_data,
            date or self._default_date(),
            registry=self.registry,
            base_currency=self.base_currency,
        )

    def nav_history(self, start: DateLike | None = None, end: DateLike | None = None) -> pd.DataFrame:
        """Return daily NAV history."""
        return build_nav_history(
            self.trades,
            self.market_data,
            start=start,
            end=end,
            registry=self.registry,
            base_currency=self.base_currency,
        )

    def performance(self) -> pd.Series:
        """Return performance metrics."""
        return performance_summary(self.nav_history(), trades=self.trades)

    def risk(self, *, risk_free_rate: float = 0.0, benchmark_nav: pd.DataFrame | None = None) -> pd.Series:
        """Return risk metrics."""
        return risk_summary(self.nav_history(), risk_free_rate=risk_free_rate, benchmark_nav=benchmark_nav)

    def exposure(self, by: str = "asset_class", date: DateLike | None = None) -> pd.DataFrame:
        """Return current exposure grouped by an instrument field or metadata key."""
        return exposure_by(self, by=by, date=date or self._default_date())

    def what_if(self, *trades: Trade) -> Scenario:
        """Create a scenario by adding hypothetical trades to a clone."""
        return Scenario(self, list(trades))

    def _default_date(self) -> date:
        candidates: list[date] = []
        market = self.market_data.read_all()
        if not market.empty:
            candidates.append(market["date"].max().date())
        if self.trades:
            candidates.append(max(trade.date for trade in self.trades))
        return max(candidates) if candidates else date.today()
