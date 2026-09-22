"""Portfolio orchestrator."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from convexpm.analytics.current_risk import current_risk as build_current_risk
from convexpm.analytics.exposure import exposure_by
from convexpm.analytics.performance import performance_summary
from convexpm.analytics.portfolio_analysis import analyze_portfolio
from convexpm.analytics.risk import risk_summary
from convexpm.portfolio.allocation import market_values as build_market_values
from convexpm.portfolio.allocation import weights as build_weights
from convexpm.instruments import InstrumentRegistry
from convexpm.market_data import MarketDataStore
from convexpm.portfolio.holdings import build_holdings
from convexpm.portfolio.valuation import nav_history as build_nav_history
from convexpm.portfolio.valuation import nav_on_date
from convexpm.scenarios.scenario import Scenario
from convexpm.transactions import Trade, load_trades, save_trades
from convexpm.utils.dates import DateLike

PORTFOLIO_METADATA_FILENAME = "portfolio.json"
PORTFOLIO_TRADES_FILENAME = "trades.parquet"
DEFAULT_PORTFOLIOS_ROOT = Path("data/portfolios")
PORTFOLIO_FORMAT_VERSION = 1


@dataclass
class Portfolio:
    """A portfolio is trades plus a registry and normalized market data.

    Portfolio-specific state can be persisted under ``data/portfolios/<id>/``.
    Instruments and market data remain shared stores and are not duplicated
    inside each portfolio directory.
    """

    name: str
    base_currency: str
    trades: list[Trade] = field(default_factory=list)
    registry: InstrumentRegistry | None = None
    market_data: MarketDataStore | None = None
    _storage_dir: Path | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.base_currency = self.base_currency.upper()
        self.trades = list(self.trades)
        if self.registry is None:
            self.registry = InstrumentRegistry(auto_load=False)
        if self.market_data is None:
            self.market_data = MarketDataStore()

    def add_trade(self, trade: Trade) -> None:
        """Add a trade to the portfolio ledger.

        Trade IDs are unique within a portfolio. Call :meth:`save` afterwards
        to persist the updated ledger.
        """
        if any(existing.trade_id == trade.trade_id for existing in self.trades):
            raise ValueError(f"Duplicate trade_id in portfolio: {trade.trade_id}")
        self.trades.append(trade)
        self.trades.sort(key=lambda item: (item.date, item.trade_id))

    def save(
        self,
        portfolio_id: str | None = None,
        *,
        root: str | Path = DEFAULT_PORTFOLIOS_ROOT,
    ) -> Path:
        """Persist portfolio metadata and trades to a local portfolio folder.

        The first save requires ``portfolio_id``. Once saved or loaded, the
        portfolio remembers its storage directory, so subsequent calls can use
        ``portfolio.save()``.
        """
        if portfolio_id is not None:
            storage_dir = _portfolio_dir(portfolio_id, root)
            self._storage_dir = storage_dir
        elif self._storage_dir is not None:
            storage_dir = self._storage_dir
        else:
            raise ValueError("portfolio_id is required the first time a portfolio is saved")

        storage_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "version": PORTFOLIO_FORMAT_VERSION,
            "name": self.name,
            "base_currency": self.base_currency,
        }
        metadata_path = storage_dir / PORTFOLIO_METADATA_FILENAME
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        save_trades(self.trades, storage_dir / PORTFOLIO_TRADES_FILENAME)
        return storage_dir

    @classmethod
    def load(
        cls,
        portfolio_id: str,
        *,
        root: str | Path = DEFAULT_PORTFOLIOS_ROOT,
        registry: InstrumentRegistry | None = None,
        market_data: MarketDataStore | None = None,
    ) -> "Portfolio":
        """Load a persisted portfolio from ``data/portfolios/<id>/``.

        By default the shared ``data/instruments.parquet`` registry and
        ``data/market_data.parquet`` store are used.
        """
        storage_dir = _portfolio_dir(portfolio_id, root)
        metadata_path = storage_dir / PORTFOLIO_METADATA_FILENAME
        if not metadata_path.exists():
            raise FileNotFoundError(f"Portfolio metadata not found: {metadata_path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        version = metadata.get("version", 1)
        if version != PORTFOLIO_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported portfolio format version: {version}. "
                f"Expected {PORTFOLIO_FORMAT_VERSION}."
            )

        portfolio = cls(
            name=metadata["name"],
            base_currency=metadata["base_currency"],
            trades=load_trades(storage_dir / PORTFOLIO_TRADES_FILENAME),
            registry=registry if registry is not None else InstrumentRegistry(),
            market_data=market_data if market_data is not None else MarketDataStore(),
        )
        portfolio._storage_dir = storage_dir
        return portfolio

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

    def market_values(self, date: DateLike | None = None) -> pd.DataFrame:
        """Return current holdings valued at date with allocation weights."""
        return build_market_values(self, date=date)

    def weights(self, date: DateLike | None = None) -> pd.Series:
        """Return current allocation weights indexed by instrument ID."""
        return build_weights(self, date=date)

    def performance(
        self,
        start: DateLike | None = None,
        end: DateLike | None = None,
    ) -> pd.Series:
        """Return realized performance from the holdings actually owned through time."""
        return performance_summary(
            self.nav_history(start=start, end=end),
            trades=self.trades,
        )

    def current_risk(
        self,
        start: DateLike | None = None,
        end: DateLike | None = None,
        *,
        benchmark: str | None = None,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ):
        """Return current-allocation risk using historical instrument returns."""
        return build_current_risk(
            self,
            start=start,
            end=end,
            benchmark=benchmark,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )

    def correlation_matrix(
        self,
        start: DateLike | None = None,
        end: DateLike | None = None,
    ) -> pd.DataFrame:
        """Return correlation of current holdings over the selected market history."""
        return self.current_risk(start=start, end=end).correlation

    def risk_contribution(
        self,
        start: DateLike | None = None,
        end: DateLike | None = None,
    ) -> pd.DataFrame:
        """Return marginal, component, and percentage current risk contribution."""
        return self.current_risk(start=start, end=end).risk_contribution

    def analyze(
        self,
        start: DateLike | None = None,
        end: DateLike | None = None,
        *,
        benchmark: str | None = None,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ):
        """Combine realized performance, current risk, and attribution."""
        return analyze_portfolio(
            self,
            start=start,
            end=end,
            benchmark=benchmark,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )

    def risk(
        self,
        *,
        risk_free_rate: float = 0.0,
        benchmark_nav: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Legacy realized-NAV risk metrics.

        Prefer current_risk() for allocation decisions. This method remains for
        scenario comparison code and will not be extended.
        """
        return risk_summary(
            self.nav_history(),
            risk_free_rate=risk_free_rate,
            benchmark_nav=benchmark_nav,
        )

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


def _portfolio_dir(portfolio_id: str, root: str | Path) -> Path:
    """Resolve a safe portfolio directory below ``root``."""
    portfolio_id = portfolio_id.strip()
    if (
        not portfolio_id
        or portfolio_id in {".", ".."}
        or "/" in portfolio_id
        or "\\" in portfolio_id
    ):
        raise ValueError("portfolio_id must be a simple folder name")
    return Path(root) / portfolio_id
