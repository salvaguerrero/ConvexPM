"""Local Parquet-backed normalized market data store."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from convexpm.instruments import Instrument, InstrumentRegistry
from convexpm.utils.dates import DateLike, to_timestamp
from convexpm.utils.terminal_plot import TerminalPlot, terminal_line_plot

MARKET_DATA_COLUMNS = ["date", "instrument_id", "value", "currency", "value_type", "source"]


class MarketDataStore:
    """Store normalized historical values in a single Parquet file."""

    columns = MARKET_DATA_COLUMNS

    def __init__(
        self,
        path: str | Path = "data/market_data.parquet",
        *,
        registry: InstrumentRegistry | None = None,
    ) -> None:
        self.path = Path(path)
        self.registry = registry

    def attach_registry(self, registry: InstrumentRegistry) -> "MarketDataStore":
        """Attach an instrument registry for rich summaries and lookup."""
        self.registry = registry
        return self

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

    def delete_instrument(self, instrument_id: str, *, missing_ok: bool = True) -> int:
        """Delete all stored market-data rows for an instrument and return the count."""
        current = self.read_all()
        if current.empty:
            if missing_ok:
                return 0
            raise KeyError(f"No market data for instrument_id={instrument_id}")

        mask = current["instrument_id"] == instrument_id
        removed = int(mask.sum())
        if removed == 0:
            if missing_ok:
                return 0
            raise KeyError(f"No market data for instrument_id={instrument_id}")

        remaining = current.loc[~mask].sort_values(["instrument_id", "date"]).reset_index(drop=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        remaining.to_parquet(self.path, index=False)
        return removed

    def summary(self) -> pd.DataFrame:
        """Return one Bloomberg-like row per stored instrument."""
        market_data = self.read_all()
        instrument_df = self._instrument_frame()
        if market_data.empty:
            if instrument_df.empty:
                return pd.DataFrame(
                    columns=[
                        "instrument_id",
                        "name",
                        "ticker",
                        "asset_class",
                        "instrument_type",
                        "currency",
                        "rows",
                        "last_date",
                        "last_price",
                        "source",
                    ]
                )
            summary = instrument_df.copy()
            summary["rows"] = 0
            summary["first_date"] = pd.NaT
            summary["last_date"] = pd.NaT
            summary["last_price"] = pd.NA
            summary["market_currency"] = pd.NA
            summary["value_type"] = pd.NA
            summary["source"] = pd.NA
            return summary

        coverage = (
            market_data.sort_values(["instrument_id", "date"])
            .groupby("instrument_id", as_index=False)
            .agg(
                rows=("value", "size"),
                first_date=("date", "min"),
                last_date=("date", "max"),
                last_price=("value", "last"),
                market_currency=("currency", "last"),
                value_type=("value_type", lambda values: ", ".join(sorted(set(values)))),
                source=("source", lambda values: ", ".join(sorted(set(values)))),
                min_price=("value", "min"),
                max_price=("value", "max"),
            )
        )
        if not instrument_df.empty:
            coverage = coverage.merge(instrument_df, on="instrument_id", how="left")
        else:
            coverage["name"] = coverage["instrument_id"]
            coverage["ticker"] = None
            coverage["asset_class"] = None
            coverage["instrument_type"] = None
            coverage["currency"] = coverage["market_currency"]

        coverage["total_return"] = coverage["instrument_id"].map(
            {
                instrument_id: _total_return(group)
                for instrument_id, group in market_data.groupby("instrument_id")
            }
        )
        columns = [
            "instrument_id",
            "name",
            "ticker",
            "asset_class",
            "instrument_type",
            "currency",
            "rows",
            "first_date",
            "last_date",
            "last_price",
            "market_currency",
            "value_type",
            "source",
            "total_return",
            "min_price",
            "max_price",
        ]
        return coverage.loc[:, columns].sort_values(["asset_class", "instrument_id"], na_position="last").reset_index(drop=True)

    def resolve(self, query: str) -> str:
        """Resolve an instrument ID, ticker, or unique name fragment."""
        query_text = str(query).strip()
        if not query_text:
            raise KeyError("Instrument query is empty")
        market_ids = set(self.read_all()["instrument_id"]) if not self.read_all().empty else set()
        if query_text in market_ids:
            return query_text

        query_lower = query_text.lower()
        candidates: list[tuple[str, str]] = []
        if self.registry is not None:
            for instrument in self.registry.all():
                searchable = [
                    instrument.instrument_id,
                    instrument.name,
                    instrument.data_symbol or "",
                    str(instrument.metadata.get("ibkr_symbol", "")),
                    str(instrument.metadata.get("yahoo_symbol", "")),
                ]
                if any(query_lower == value.lower() for value in searchable if value):
                    return instrument.instrument_id
                if any(query_lower in value.lower() for value in searchable if value):
                    candidates.append((instrument.instrument_id, instrument.name))

        if not candidates:
            matching_market_ids = sorted(instrument_id for instrument_id in market_ids if query_lower in instrument_id.lower())
            candidates = [(instrument_id, instrument_id) for instrument_id in matching_market_ids]

        unique_ids = sorted({instrument_id for instrument_id, _ in candidates})
        if len(unique_ids) == 1:
            return unique_ids[0]
        if len(unique_ids) > 1:
            raise KeyError(f"Ambiguous instrument query {query!r}. Matches: {unique_ids}")
        raise KeyError(f"Unknown instrument query: {query!r}")

    def __getitem__(self, query: str) -> "InstrumentMarketData":
        """Return a rich market-data view for one instrument."""
        return InstrumentMarketData(self, self.resolve(query))

    def __repr__(self) -> str:
        summary = self.summary()
        if summary.empty:
            return "MarketDataStore(empty)"
        display_columns = [
            "instrument_id",
            "name",
            "ticker",
            "asset_class",
            "currency",
            "rows",
            "last_date",
            "last_price",
            "source",
        ]
        return summary.loc[:, display_columns].to_string(index=False)

    def _repr_html_(self) -> str:
        return self.summary()._repr_html_()

    def _instrument_frame(self) -> pd.DataFrame:
        if self.registry is None:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "instrument_id": instrument.instrument_id,
                    "name": instrument.name,
                    "ticker": instrument.data_symbol,
                    "asset_class": instrument.asset_class,
                    "instrument_type": instrument.instrument_type,
                    "currency": instrument.currency,
                    "data_source": instrument.data_source,
                }
                for instrument in self.registry.all()
            ]
        )


class InstrumentMarketData:
    """Interactive market-data view for one instrument."""

    def __init__(self, store: MarketDataStore, instrument_id: str) -> None:
        self.store = store
        self.instrument_id = instrument_id

    @property
    def instrument(self) -> Instrument | None:
        if self.store.registry is None:
            return None
        try:
            return self.store.registry.get(self.instrument_id)
        except KeyError:
            return None

    @property
    def history(self) -> pd.DataFrame:
        """Saved market values for this instrument."""
        return self.store.get_history(self.instrument_id)

    @property
    def metadata(self) -> dict[str, Any]:
        """Instrument metadata from the registry."""
        instrument = self.instrument
        return dict(instrument.metadata) if instrument is not None else {}

    @property
    def info(self) -> pd.Series:
        """Instrument properties as a Series."""
        instrument = self.instrument
        latest = self.store.latest(self.instrument_id)
        return pd.Series(
            {
                "instrument_id": self.instrument_id,
                "name": instrument.name if instrument else self.instrument_id,
                "ticker": instrument.data_symbol if instrument else None,
                "instrument_type": instrument.instrument_type if instrument else None,
                "asset_class": instrument.asset_class if instrument else None,
                "currency": instrument.currency if instrument else (latest["currency"] if latest is not None else None),
                "data_source": instrument.data_source if instrument else None,
                "rows": len(self.history),
                "first_date": self.history["date"].min() if not self.history.empty else pd.NaT,
                "last_date": latest["date"] if latest is not None else pd.NaT,
                "last_price": latest["value"] if latest is not None else pd.NA,
                "value_type": latest["value_type"] if latest is not None else None,
                "market_source": latest["source"] if latest is not None else None,
            }
        )

    @property
    def stats(self) -> "MarketStats":
        """Interactive statistics for this instrument."""
        return MarketStats(self)

    @property
    def prices(self) -> pd.Series:
        """Saved values indexed by date."""
        history = self.history
        if history.empty:
            return pd.Series(dtype=float, name=self.instrument_id)
        return history.set_index("date")["value"].rename(self.instrument_id)

    def plot(self, *, style: str = "bloomberg", show: bool = True, **kwargs: Any) -> Any:
        """Plot saved values. Requires matplotlib."""
        if style in {"terminal", "text", "repl"}:
            return self.plot_text(**kwargs)

        import matplotlib.pyplot as plt

        title = kwargs.pop("title", f"{self.instrument_id} price / NAV")
        if style == "bloomberg":
            kwargs.setdefault("figsize", (12, 5))
            kwargs.setdefault("color", "#ffb000")
            ax = self.prices.plot(title=title, grid=False, **kwargs)
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
            ax = self.prices.plot(title=title, grid=kwargs.pop("grid", True), **kwargs)
        ax.set_xlabel("")
        if show:
            plt.show()
        return ax

    def plot_text(self, *, width: int = 80, height: int = 16, title: str | None = None) -> TerminalPlot:
        """Return a terminal-native text plot for REPL display."""
        return terminal_line_plot(
            self.prices,
            title=title or f"{self.instrument_id} price / NAV",
            width=width,
            height=height,
        )

    def returns(self) -> pd.Series:
        """Saved-value percentage returns."""
        return self.prices.pct_change().dropna()

    def __repr__(self) -> str:
        info = self.info
        lines = [info.to_string()]
        if self.metadata:
            metadata = pd.json_normalize(self.metadata, sep=".").T
            metadata.columns = ["value"] if not metadata.empty else []
            lines.append("\nmetadata")
            lines.append(metadata.to_string())
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        parts = [self.info.to_frame("value")._repr_html_()]
        if self.metadata:
            parts.append(pd.json_normalize(self.metadata, sep=".").T.to_html(header=False))
        return "".join(parts)


class MarketStats:
    """Callable statistics accessor for one instrument."""

    def __init__(self, market_data: InstrumentMarketData) -> None:
        self.market_data = market_data

    @property
    def summary(self) -> pd.Series:
        """Overall statistics across the saved history."""
        history = self.market_data.history
        return _stats_for_history(history)

    def by_period(
        self,
        periods: int | None = None,
        *,
        freq: str = "Y",
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> pd.DataFrame:
        """Return statistics with one column per year, quarter, or month.

        ``periods`` means the trailing number of periods in ``freq``. For
        example, ``periods=10, freq="Y"`` returns the last 10 calendar years,
        while ``periods=12, freq="M"`` returns the last 12 months.
        """
        history = self.market_data.history
        if history.empty:
            return pd.DataFrame()

        freq = _normalize_stats_freq(freq)
        history = history.sort_values("date").copy()
        if periods is not None:
            last_date = history["date"].max()
            offset = _period_offset(periods, freq)
            history = history.loc[history["date"] > last_date - offset]

        grouped = history.groupby(history["date"].dt.to_period(freq), sort=True)
        columns: dict[str, pd.Series] = {}
        for period, group in grouped:
            if group.empty:
                continue
            columns[str(period)] = _stats_for_history(
                group,
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
            )
        return pd.DataFrame(columns)

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


def _total_return(group: pd.DataFrame) -> float | None:
    values = group.sort_values("date")["value"]
    if values.empty or values.iloc[0] == 0:
        return None
    return float(values.iloc[-1] / values.iloc[0] - 1)


def _stats_for_history(
    history: pd.DataFrame,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> pd.Series:
    if history.empty:
        return pd.Series(dtype=object)
    history = history.sort_values("date")
    values = history["value"]
    returns = values.pct_change().dropna()
    excess_returns = returns - (risk_free_rate / periods_per_year)
    volatility = returns.std()
    sharpe = (
        (excess_returns.mean() / volatility) * (periods_per_year**0.5)
        if not returns.empty and volatility and pd.notna(volatility)
        else pd.NA
    )
    return pd.Series(
        {
            "observations": len(history),
            "first_date": history["date"].min(),
            "last_date": history["date"].max(),
            "first_price": values.iloc[0],
            "last_price": values.iloc[-1],
            "min_price": values.min(),
            "max_price": values.max(),
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
