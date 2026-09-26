"""Portfolio orchestrator."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

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
from convexpm.portfolio.valuation import fx_rate_to_base
from convexpm.scenarios.scenario import Scenario
from convexpm.transactions import Trade, TradeLedger, load_trades, save_trades
from convexpm.utils.dates import DateLike
from convexpm.utils.terminal_plot import TerminalPlot, terminal_line_plot

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
    trades: list[Trade] | TradeLedger = field(default_factory=list)
    registry: InstrumentRegistry | None = None
    market_data: MarketDataStore | None = None
    _storage_dir: Path | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.base_currency = self.base_currency.upper()
        if self.registry is None:
            self.registry = InstrumentRegistry(auto_load=False)
        self.trades = (
            self.trades
            if isinstance(self.trades, TradeLedger)
            else TradeLedger(self.trades, registry=self.registry)
        )
        self.trades.registry = self.registry
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

    def add_trades(self, trades: list[Trade], *, duplicate: str = "skip") -> int:
        """Add multiple trades and return the number inserted.

        ``duplicate="skip"`` makes repeated imports idempotent by ignoring
        trades whose stable ``trade_id`` is already present in the portfolio.
        Use ``duplicate="raise"`` to surface duplicates as errors.
        """
        if duplicate not in {"skip", "raise"}:
            raise ValueError("duplicate must be 'skip' or 'raise'")

        existing_ids = {trade.trade_id for trade in self.trades}
        existing_keys = {_trade_identity_key(trade) for trade in self.trades}
        inserted = 0
        for trade in trades:
            trade_key = _trade_identity_key(trade)
            if trade.trade_id in existing_ids or trade_key in existing_keys:
                if duplicate == "raise":
                    raise ValueError(f"Duplicate trade_id in portfolio: {trade.trade_id}")
                continue
            self.trades.append(trade)
            existing_ids.add(trade.trade_id)
            existing_keys.add(trade_key)
            inserted += 1
        self.trades.sort(key=lambda item: (item.date, item.trade_id))
        return inserted

    def delete_trade(self, number_or_trade_id: int | str) -> Trade:
        """Delete a trade by one-based row number or ``trade_id`` and return it."""
        return self.trades.delete(number_or_trade_id)

    def deduplicate_trades(self) -> int:
        """Remove duplicate trades with the same economic fingerprint.

        The first occurrence is kept. This is useful after re-importing
        overlapping broker exports that produced different historical IDs.
        """
        unique: list[Trade] = []
        seen: set[tuple[object, ...]] = set()
        removed = 0
        for trade in self.trades:
            key = _trade_identity_key(trade)
            if key in seen:
                removed += 1
                continue
            seen.add(key)
            unique.append(trade)
        if removed:
            self.trades = TradeLedger(unique, registry=self.registry)
            self.trades.sort(key=lambda item: (item.date, item.trade_id))
        return removed

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

    def nav_history(
        self,
        start: DateLike | None = None,
        end: DateLike | None = None,
        *,
        skip_missing_prices: bool = False,
    ) -> pd.DataFrame:
        """Return daily NAV history."""
        return build_nav_history(
            self.trades,
            self.market_data,
            start=start,
            end=end,
            registry=self.registry,
            base_currency=self.base_currency,
            skip_missing_prices=skip_missing_prices,
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

    def summary(self, date: DateLike | None = None) -> pd.DataFrame:
        """Return a terminal-style holdings and valuation table."""
        valuation_date = date or self._default_date()
        holdings = self.holdings(valuation_date)
        rows = []
        for instrument_id, quantity in holdings.items():
            instrument = None
            try:
                instrument = self.registry.get(instrument_id) if self.registry is not None else None
            except KeyError:
                instrument = None

            latest_row = None
            value = pd.NA
            market_value = pd.NA
            price_date = pd.NaT
            price_source = None
            currency = instrument.currency if instrument else self.base_currency
            status = "ok"
            try:
                history = self.market_data.get_history(instrument_id, end=valuation_date)
                if history.empty:
                    raise KeyError(f"No market data for {instrument_id}")
                latest_row = history.iloc[-1]
                value = float(latest_row["value"])
                local_market_value = float(quantity) * value
                price_date = latest_row["date"]
                price_source = latest_row["source"]
                currency = str(latest_row["currency"]).upper()
                try:
                    fx_rate, fx_pair = fx_rate_to_base(currency, self.base_currency, self.market_data, valuation_date)
                    market_value = local_market_value * fx_rate
                except KeyError:
                    fx_rate = pd.NA
                    fx_pair = None
                    status = f"missing_fx:{currency}->{self.base_currency}"
            except (KeyError, IndexError):
                local_market_value = pd.NA
                fx_rate = pd.NA
                fx_pair = None
                status = "missing_price"

            rows.append(
                {
                    "instrument_id": instrument_id,
                    "name": instrument.name if instrument else instrument_id,
                    "asset_class": instrument.asset_class if instrument else None,
                    "instrument_type": instrument.instrument_type if instrument else None,
                    "quantity": float(quantity),
                    "price": value,
                    "price_date": price_date,
                    "local_market_value": local_market_value,
                    "fx_rate": fx_rate,
                    "fx_pair": fx_pair,
                    "market_value": market_value,
                    "currency": currency,
                    "base_currency": self.base_currency,
                    "source": price_source,
                    "status": status,
                }
            )

        summary = pd.DataFrame(rows)
        if summary.empty:
            return pd.DataFrame(
                columns=[
                    "instrument_id",
                    "name",
                    "asset_class",
                    "quantity",
                    "price",
                    "price_date",
                    "local_market_value",
                    "fx_rate",
                    "fx_pair",
                    "market_value",
                    "weight",
                    "currency",
                    "base_currency",
                    "status",
                ]
            )
        valued = pd.to_numeric(summary["market_value"], errors="coerce")
        total = valued.sum()
        summary["weight"] = valued / total if abs(total) > 1e-12 else pd.NA
        return summary.sort_values("market_value", ascending=False, na_position="last").reset_index(drop=True)

    @property
    def stats(self) -> "PortfolioStats":
        """Interactive portfolio statistics."""
        return PortfolioStats(self)

    def plot(self, *, style: str = "bloomberg", show: bool = True, **kwargs: Any) -> Any:
        """Plot portfolio NAV history. Requires matplotlib."""
        if style in {"terminal", "text", "repl"}:
            return self.plot_text(**kwargs)

        import matplotlib.pyplot as plt

        nav = self.nav_history(skip_missing_prices=True)
        if nav.empty:
            raise ValueError("Cannot plot an empty portfolio NAV history")
        series = nav.set_index("date")["nav"].rename(self.name)
        title = kwargs.pop("title", f"{self.name} NAV")
        if style == "bloomberg":
            kwargs.setdefault("figsize", (12, 5))
            kwargs.setdefault("color", "#ffb000")
            ax = series.plot(title=title, grid=False, **kwargs)
            figure = ax.get_figure()
            figure.patch.set_facecolor("#050505")
            ax.set_facecolor("#050505")
            ax.tick_params(colors="#f2f2f2")
            ax.title.set_color("#f2f2f2")
            ax.yaxis.label.set_color("#f2f2f2")
            ax.xaxis.label.set_color("#f2f2f2")
            for spine in ax.spines.values():
                spine.set_color("#5a5a5a")
            ax.grid(True, color="#333333", linewidth=0.6)
        else:
            ax = series.plot(title=title, grid=kwargs.pop("grid", True), **kwargs)
        ax.set_xlabel("")
        if show:
            plt.show()
        return ax

    def plot_text(self, *, width: int = 80, height: int = 16, title: str | None = None) -> TerminalPlot:
        """Return a terminal-native NAV chart for REPL display."""
        nav = self.nav_history(skip_missing_prices=True)
        series = nav.set_index("date")["nav"].rename(self.name) if not nav.empty else pd.Series(dtype=float)
        return terminal_line_plot(
            series,
            title=title or f"{self.name} NAV",
            width=width,
            height=height,
        )

    def __repr__(self) -> str:
        summary = self.summary()
        header = f"{self.name} ({self.base_currency})"
        if summary.empty:
            return f"{header}\nNo holdings"
        total = pd.to_numeric(summary["market_value"], errors="coerce").sum()
        display_columns = [
            "instrument_id",
            "name",
            "asset_class",
            "quantity",
            "price",
            "market_value",
            "weight",
            "currency",
            "status",
        ]
        return f"{header}\nNAV: {total:,.2f} {self.base_currency}\n{summary.loc[:, display_columns].to_string(index=False)}"

    def _repr_html_(self) -> str:
        summary = self.summary()
        total = pd.to_numeric(summary.get("market_value", pd.Series(dtype=float)), errors="coerce").sum()
        return f"<h3>{self.name} ({self.base_currency}) - NAV {total:,.2f}</h3>" + summary._repr_html_()

    def _default_date(self) -> date:
        candidates: list[date] = []
        market = self.market_data.read_all()
        if not market.empty:
            candidates.append(market["date"].max().date())
        if self.trades:
            candidates.append(max(trade.date for trade in self.trades))
        return max(candidates) if candidates else date.today()


class PortfolioStats:
    """Callable stats accessor for a portfolio."""

    def __init__(self, portfolio: Portfolio) -> None:
        self.portfolio = portfolio

    @property
    def summary(self) -> pd.Series:
        nav = self.portfolio.nav_history(skip_missing_prices=True)
        return _stats_for_nav(nav)

    def by_period(
        self,
        periods: int | None = None,
        *,
        freq: str = "Y",
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> pd.DataFrame:
        nav = self.portfolio.nav_history(skip_missing_prices=True)
        if nav.empty:
            return pd.DataFrame()
        freq = _normalize_stats_freq(freq)
        nav = nav.sort_values("date")
        if periods is not None:
            last_date = nav["date"].max()
            nav = nav.loc[nav["date"] > last_date - _period_offset(periods, freq)]
        grouped = nav.groupby(nav["date"].dt.to_period(freq), sort=True)
        return pd.DataFrame(
            {
                str(period): _stats_for_nav(
                    group,
                    risk_free_rate=risk_free_rate,
                    periods_per_year=periods_per_year,
                )
                for period, group in grouped
            }
        )

    def __call__(
        self,
        periods: int | None = None,
        *,
        freq: str = "Y",
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> pd.DataFrame:
        return self.by_period(
            periods=periods,
            freq=freq,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )

    def __getitem__(self, key: str) -> Any:
        return self.summary[key]

    def __repr__(self) -> str:
        table = self.by_period()
        if table.empty:
            return self.summary.to_string()
        return table.to_string()

    def _repr_html_(self) -> str:
        table = self.by_period()
        if table.empty:
            return self.summary.to_frame("value")._repr_html_()
        return table._repr_html_()


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


def _trade_identity_key(trade: Trade) -> tuple[object, ...]:
    source = str(trade.metadata.get("source") or (trade.tags[0] if trade.tags else "")).lower()
    broker_type = str(
        trade.metadata.get("broker_transaction_type")
        or trade.metadata.get("operation_type")
        or ""
    ).lower()
    return (
        source,
        trade.date,
        trade.account,
        trade.instrument_id,
        trade.side,
        trade.currency,
        broker_type,
        _rounded_trade_number(trade.quantity),
        _rounded_trade_number(trade.amount),
        _rounded_trade_number(trade.value_per_unit),
        _rounded_trade_number(trade.fees),
    )


def _rounded_trade_number(value: float | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 8)


def _stats_for_nav(
    nav: pd.DataFrame,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> pd.Series:
    if nav.empty:
        return pd.Series(dtype=object)
    nav = nav.sort_values("date")
    values = nav["nav"]
    returns = values.pct_change().dropna()
    volatility = returns.std()
    excess_returns = returns - (risk_free_rate / periods_per_year)
    sharpe = (
        (excess_returns.mean() / volatility) * (periods_per_year**0.5)
        if not returns.empty and volatility and pd.notna(volatility)
        else pd.NA
    )
    return pd.Series(
        {
            "observations": len(nav),
            "first_date": nav["date"].min(),
            "last_date": nav["date"].max(),
            "first_nav": values.iloc[0],
            "last_nav": values.iloc[-1],
            "min_nav": values.min(),
            "max_nav": values.max(),
            "total_return": (values.iloc[-1] / values.iloc[0] - 1) if values.iloc[0] else pd.NA,
            "mean_return": returns.mean() if not returns.empty else pd.NA,
            "volatility": volatility if not returns.empty else pd.NA,
            "sharpe": sharpe,
        }
    )


def _normalize_stats_freq(freq: str) -> str:
    normalized = freq.strip().upper()
    aliases = {
        "Y": "Y",
        "A": "Y",
        "ANNUAL": "Y",
        "YEAR": "Y",
        "YEARLY": "Y",
        "YEARS": "Y",
        "Q": "Q",
        "QUARTER": "Q",
        "QUARTERLY": "Q",
        "QUARTERS": "Q",
        "M": "M",
        "MONTH": "M",
        "MONTHLY": "M",
        "MONTHS": "M",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError("freq must be one of 'Y', 'Q', or 'M'") from exc


def _period_offset(periods: int, freq: str) -> pd.DateOffset:
    if periods <= 0:
        raise ValueError("periods must be positive")
    if freq == "Y":
        return pd.DateOffset(years=periods)
    if freq == "Q":
        return pd.DateOffset(months=periods * 3)
    if freq == "M":
        return pd.DateOffset(months=periods)
    raise ValueError("freq must be one of 'Y', 'Q', or 'M'")
