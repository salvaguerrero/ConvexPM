"""Portfolio orchestrator."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from convexpm.analytics.current_risk import current_risk as build_current_risk
from convexpm.analytics.exposure import exposure_by
from convexpm.analytics.performance import performance_summary
from convexpm.analytics.views import performance_view as build_performance_view, risk_view as build_risk_view
from convexpm.analytics.portfolio_analysis import analyze_portfolio
from convexpm.analytics.risk import risk_summary
from convexpm.portfolio.allocation import market_values as build_market_values
from convexpm.portfolio.allocation import weights as build_weights
from convexpm.instruments import Instrument, InstrumentRegistry
from convexpm.market_data import MarketDataStore
from convexpm.portfolio.holdings import build_holdings
from convexpm.portfolio.valuation import nav_history as build_nav_history
from convexpm.portfolio.valuation import nav_on_date
from convexpm.portfolio.valuation import fx_rate_to_base
from convexpm.scenarios.scenario import Scenario
from convexpm.transactions import Trade, TradeLedger, load_trades, save_trades
from convexpm.utils.dates import DateLike
from convexpm.utils.terminal_table import blue_table, terminal_title

PORTFOLIO_METADATA_FILENAME = "portfolio.json"
PORTFOLIO_TRADES_FILENAME = "trades.parquet"
DEFAULT_PORTFOLIOS_ROOT = Path("data/portfolios")
PORTFOLIO_FORMAT_VERSION = 1
ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{10}$")
CASH_INSTRUMENT_SUFFIX = "_CASH"


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
    benchmark: str | None = None
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

    def add_trade(self, trade: Trade, *, cash_ledger: bool = False) -> None:
        """Add a trade to the portfolio ledger.

        Trade IDs are unique within a portfolio. Call :meth:`save` afterwards
        to persist the updated ledger.
        """
        self.add_trades([trade], duplicate="raise", cash_ledger=cash_ledger)

    def add_trades(
        self,
        trades: list[Trade],
        *,
        duplicate: str = "skip",
        cash_ledger: bool = False,
    ) -> int:
        """Add multiple trades and return the number inserted.

        ``duplicate="skip"`` makes repeated imports idempotent by ignoring
        trades whose stable ``trade_id`` is already present in the portfolio.
        Use ``duplicate="raise"`` to surface duplicates as errors.

        Set ``cash_ledger=True`` to add a generated cash leg for every
        non-cash trade with a known amount. Buys reduce cash; sells increase it.
        """
        if duplicate not in {"skip", "raise"}:
            raise ValueError("duplicate must be 'skip' or 'raise'")

        existing_by_id = {trade.trade_id: trade for trade in self.trades}
        existing_by_key = {_trade_identity_key(trade): trade for trade in self.trades}
        inserted = 0
        for trade in self._trades_with_cash_legs(trades) if cash_ledger else trades:
            self._ensure_cash_price_for_trade(trade)
            trade_key = _trade_identity_key(trade)
            existing = existing_by_id.get(trade.trade_id) or existing_by_key.get(trade_key)
            if existing is not None:
                if duplicate == "raise":
                    raise ValueError(f"Duplicate trade_id in portfolio: {trade.trade_id}")
                _upgrade_duplicate_trade(existing, trade)
                continue
            self.trades.append(trade)
            existing_by_id[trade.trade_id] = trade
            existing_by_key[trade_key] = trade
            inserted += 1
        self.trades.sort(key=lambda item: (item.date, item.trade_id))
        return inserted

    def _trades_with_cash_legs(self, trades: list[Trade]) -> list[Trade]:
        expanded: list[Trade] = []
        for trade in trades:
            expanded.append(trade)
            cash_leg = self._cash_leg_for_trade(trade)
            if cash_leg is not None:
                expanded.append(cash_leg)
        return expanded

    def _cash_leg_for_trade(self, trade: Trade) -> Trade | None:
        if _is_cash_instrument_id(trade.instrument_id):
            return None

        amount = _trade_amount_in_trade_currency(trade)
        if amount is None:
            return None

        currency = (trade.currency or self.base_currency).upper()
        cash_amount = amount + float(trade.fees) if trade.side == "BUY" else max(amount - float(trade.fees), 0.0)
        if cash_amount == 0:
            return None

        self._ensure_cash_instrument(currency, trade.date)
        cash_side = "SELL" if trade.side == "BUY" else "BUY"
        cash_instrument_id = _cash_instrument_id(currency)
        metadata = {
            "source": "cash_ledger",
            "parent_trade_id": trade.trade_id,
            "parent_instrument_id": trade.instrument_id,
            "parent_side": trade.side,
        }
        for key in ("concept", "description", "broker_transaction_type", "operation_type", "note"):
            if trade.metadata.get(key):
                metadata[key] = trade.metadata[key]
                break

        return Trade(
            trade_id=f"{trade.trade_id}:cash",
            date=trade.date,
            instrument_id=cash_instrument_id,
            side=cash_side,
            quantity=cash_amount,
            amount=cash_amount,
            value_per_unit=1.0,
            fees=0.0,
            currency=currency,
            account=trade.account,
            tags=[*trade.tags, "cash"] if "cash" not in trade.tags else list(trade.tags),
            metadata=metadata,
        )

    def _ensure_cash_price_for_trade(self, trade: Trade) -> None:
        if _is_cash_instrument_id(trade.instrument_id):
            currency = trade.instrument_id[: -len(CASH_INSTRUMENT_SUFFIX)]
            self._ensure_cash_instrument(currency, trade.date)

    def _ensure_cash_instrument(self, currency: str, trade_date: date) -> None:
        currency = currency.upper()
        instrument_id = _cash_instrument_id(currency)
        if instrument_id not in self.registry:
            self.registry.add(
                Instrument(
                    instrument_id=instrument_id,
                    name=f"{currency} Cash",
                    instrument_type="cash",
                    asset_class="cash",
                    currency=currency,
                    data_source="manual",
                    data_symbol=None,
                    metadata={"source": "cash_ledger"},
                )
            )
        self.market_data.upsert(
            pd.DataFrame(
                {
                    "date": [pd.Timestamp(trade_date)],
                    "instrument_id": [instrument_id],
                    "value": [1.0],
                    "currency": [currency],
                    "value_type": ["cash"],
                    "source": ["cash_ledger"],
                }
            )
        )

    def delete_trade(self, number_or_trade_id: int | str) -> Trade:
        """Delete a trade by one-based row number or ``trade_id`` and return it."""
        return self.trades.delete(number_or_trade_id)

    def deduplicate_trades(self) -> int:
        """Remove duplicate trades with the same economic fingerprint.

        The first occurrence is kept. This is useful after re-importing
        overlapping broker exports that produced different historical IDs.
        """
        unique: list[Trade] = []
        seen: dict[tuple[object, ...], Trade] = {}
        removed = 0
        for trade in self.trades:
            key = _trade_identity_key(trade)
            if key in seen:
                _upgrade_duplicate_trade(seen[key], trade)
                removed += 1
                continue
            seen[key] = trade
            unique.append(trade)
        if removed:
            self.trades = TradeLedger(unique, registry=self.registry)
            self.trades.sort(key=lambda item: (item.date, item.trade_id))
        return removed

    def set_benchmark(self, benchmark: str | None) -> None:
        """Set the default benchmark instrument for performance comparisons."""
        self.benchmark = benchmark

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
            "benchmark": self.benchmark,
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
            benchmark=metadata.get("benchmark"),
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

    def performance_view(
        self,
        start: DateLike | None = None,
        end: DateLike | None = None,
        *,
        benchmark: str | None = None,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ):
        """Return the simple realized-performance view used by the product/REPL."""
        return build_performance_view(
            self,
            start=start,
            end=end,
            benchmark=benchmark if benchmark is not None else self.benchmark,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
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

    def risk_view(
        self,
        start: DateLike | None = None,
        end: DateLike | None = None,
        *,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ):
        """Return the simple current-allocation risk view used by the product/REPL."""
        return build_risk_view(
            self,
            start=start,
            end=end,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )

    def chart(self, **kwargs):
        """Open an interactive NAV chart with contributions and risk bands."""
        from convexpm.plotting import portfolio_chart as build_portfolio_chart

        return build_portfolio_chart(self, **kwargs)

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

    def __repr__(self) -> str:
        summary = self.summary()
        header = terminal_title(self.name, f"BASE {self.base_currency}")
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
        display = summary.loc[:, display_columns].copy()
        fx_rate = pd.to_numeric(summary["fx_rate"], errors="coerce")
        price = pd.to_numeric(summary["price"], errors="coerce")
        display["price"] = price * fx_rate
        display["weight"] = pd.to_numeric(display["weight"], errors="coerce") * 100.0
        display["currency"] = self.base_currency
        return "\n".join(
            [
                header,
                f"\033[1m\033[38;5;39mNAV: {total:,.2f} {self.base_currency}\033[0m",
                blue_table(display, index=False),
            ]
        )

    def _repr_html_(self) -> str:
        summary = self.summary()
        total = pd.to_numeric(summary.get("market_value", pd.Series(dtype=float)), errors="coerce").sum()
        display = summary.copy()
        if "weight" in display:
            display["weight"] = pd.to_numeric(display["weight"], errors="coerce") * 100.0
        return f"<h3>{self.name} ({self.base_currency}) - NAV {total:,.2f}</h3>" + display._repr_html_()

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


def _trade_identity_key(trade: Trade) -> tuple[object, ...]:
    source = str(trade.metadata.get("source") or (trade.tags[0] if trade.tags else "")).lower()
    broker_type = str(
        trade.metadata.get("broker_transaction_type")
        or trade.metadata.get("operation_type")
        or trade.metadata.get("cash_flow_type")
        or ""
    ).lower()
    return (
        source,
        trade.date,
        trade.account,
        _trade_instrument_identity(trade, source),
        trade.side,
        trade.currency,
        broker_type,
        _rounded_trade_number(trade.quantity),
        _rounded_trade_number(trade.amount),
        _rounded_trade_number(trade.value_per_unit),
        _rounded_trade_number(trade.fees),
        _trade_metadata_identity(trade, source),
    )


def _trade_instrument_identity(trade: Trade, source: str) -> str:
    if source == "renta4":
        fallback_id = trade.metadata.get("fallback_instrument_id")
        if isinstance(fallback_id, str) and fallback_id.strip():
            return fallback_id.strip().upper()
        fund_name = trade.metadata.get("fund_name")
        if isinstance(fund_name, str) and fund_name.strip():
            return _renta4_fallback_id(fund_name)
    return trade.instrument_id


def _trade_metadata_identity(trade: Trade, source: str) -> object:
    if source == "renta4_cash":
        return _rounded_trade_number(trade.metadata.get("balance"))
    return None


def _cash_instrument_id(currency: str) -> str:
    return f"{currency.upper()}{CASH_INSTRUMENT_SUFFIX}"


def _is_cash_instrument_id(instrument_id: str) -> bool:
    return instrument_id.upper().endswith(CASH_INSTRUMENT_SUFFIX)


def _trade_amount_in_trade_currency(trade: Trade) -> float | None:
    if trade.amount is not None:
        return float(trade.amount)
    if trade.quantity is not None and trade.value_per_unit is not None:
        return float(trade.quantity) * float(trade.value_per_unit)
    return None


def _upgrade_duplicate_trade(existing: Trade, incoming: Trade) -> None:
    for key, value in incoming.metadata.items():
        if value is not None and key not in existing.metadata:
            existing.metadata[key] = value

    if not _is_renta4_canonical_upgrade(existing, incoming):
        return
    previous_ids = list(existing.metadata.get("previous_instrument_ids") or [])
    if existing.instrument_id not in previous_ids:
        previous_ids.append(existing.instrument_id)
    existing.instrument_id = incoming.instrument_id
    existing.metadata["previous_instrument_ids"] = previous_ids
    existing.metadata["upgraded_instrument_id_from"] = "renta4_isin_resolution"


def _is_renta4_canonical_upgrade(existing: Trade, incoming: Trade) -> bool:
    existing_source = str(existing.metadata.get("source") or (existing.tags[0] if existing.tags else "")).lower()
    incoming_source = str(incoming.metadata.get("source") or (incoming.tags[0] if incoming.tags else "")).lower()
    return (
        existing_source == "renta4"
        and incoming_source == "renta4"
        and existing.instrument_id.startswith("R4_")
        and bool(ISIN_RE.match(incoming.instrument_id))
        and _trade_instrument_identity(existing, existing_source) == _trade_instrument_identity(incoming, incoming_source)
    )


def _renta4_fallback_id(value: str) -> str:
    normalized = _strip_accents(value).upper()
    slug = re.sub(r"[^A-Z0-9]+", "_", normalized).strip("_")
    return f"R4_{slug}"


def _strip_accents(value: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(char)
    )


def _rounded_trade_number(value: float | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 8)


