"""Interactive instrument plotting with TradingView Lightweight Charts."""

from __future__ import annotations

import atexit
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from convexpm.instruments import Instrument, InstrumentRegistry
from convexpm.market_data import MarketDataStore
from convexpm.portfolio.holdings import signed_trade_quantity
from convexpm.portfolio.valuation import fx_rate_to_base
from convexpm.utils.dates import DateLike, to_timestamp


DEFAULT_COLORS = (
    "#f1b84b",
    "#4da3ff",
    "#41c287",
    "#ff6f91",
    "#b993ff",
    "#f97316",
    "#2dd4bf",
    "#e879f9",
)


_CHART_CLOSE_WATCHERS: set[int] = set()
_CHART_CLASSES_TO_CLEANUP: set[type[Any]] = set()


def _close_chart(chart: Any, *, timeout: float = 2.0) -> None:
    """Close a lightweight-charts webview and avoid leaving its process alive."""
    if chart is None or not hasattr(chart, "exit"):
        return
    _CHART_CLASSES_TO_CLEANUP.add(type(chart))

    try:
        chart.exit()
    finally:
        _terminate_chart_webview(chart, timeout=timeout)


def _terminate_chart_webview(chart: Any, *, timeout: float = 2.0) -> None:
    chart_cls = type(chart)
    wv = getattr(chart_cls, "WV", None)
    process = getattr(wv, "wv_process", None)
    if wv is None or process is None or getattr(process, "_popen", None) is None:
        return
    if process.is_alive():
        process.terminate()
        process.join(timeout)
    if process.is_alive() and hasattr(process, "kill"):
        process.kill()
        process.join(timeout)
    if not process.is_alive() and hasattr(wv, "_reset"):
        wv._reset()


def _watch_chart_close(chart: Any) -> None:
    chart_cls = type(chart)
    wv = getattr(chart_cls, "WV", None)
    emit_queue = getattr(wv, "emit_queue", None)
    if wv is None or emit_queue is None:
        return
    _CHART_CLASSES_TO_CLEANUP.add(chart_cls)

    watcher_id = id(wv)
    if watcher_id in _CHART_CLOSE_WATCHERS:
        return
    _CHART_CLOSE_WATCHERS.add(watcher_id)

    def watch() -> None:
        try:
            while True:
                process = getattr(wv, "wv_process", None)
                if process is None or getattr(process, "_popen", None) is None:
                    return
                if not process.is_alive():
                    if hasattr(wv, "_reset"):
                        wv._reset()
                    return
                try:
                    response = emit_queue.get(timeout=0.25)
                except Exception:
                    continue
                if response == "exit":
                    _terminate_chart_webview(chart, timeout=1.0)
                    return
        finally:
            _CHART_CLOSE_WATCHERS.discard(watcher_id)

    threading.Thread(target=watch, name="convexpm-chart-close-watch", daemon=True).start()


atexit.register(lambda: _terminate_chart_classes_at_exit())


def _terminate_chart_classes_at_exit() -> None:
    for chart_cls in list(_CHART_CLASSES_TO_CLEANUP):
        wv = getattr(chart_cls, "WV", None)
        process = getattr(wv, "wv_process", None)
        if wv is None or process is None or getattr(process, "_popen", None) is None:
            continue
        if process.is_alive():
            process.terminate()
            process.join(1.0)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()


@dataclass
class _InstrumentSeries:
    instrument: Instrument
    line: Any
    color: str
    curr: str | None


class InstrumentPlot:
    """Interactive multi-instrument chart backed by local ConvexPM data."""

    def __init__(
        self,
        *,
        start: DateLike | None = None,
        end: DateLike | None = None,
        curr: str | None = None,
        registry: InstrumentRegistry | None = None,
        market_data: MarketDataStore | None = None,
        registry_path: str | Path = "data/instruments.parquet",
        market_data_path: str | Path = "data/market_data.parquet",
        chart: Any | None = None,
        show: bool = True,
        block: bool = False,
        width: int = 1200,
        height: int = 720,
        colors: tuple[str, ...] = DEFAULT_COLORS,
    ) -> None:
        self.start = to_timestamp(start).normalize() if start is not None else None
        self.end = to_timestamp(end).normalize() if end is not None else None
        self.curr = curr.upper() if curr else None
        self.registry = registry if registry is not None else InstrumentRegistry(registry_path)
        self.market_data = (
            market_data
            if market_data is not None
            else MarketDataStore(market_data_path, registry=self.registry)
        )
        self.colors = colors
        self._series: dict[str, _InstrumentSeries] = {}
        self._relative = False
        self.chart = chart if chart is not None else self._new_chart(width=width, height=height)
        self._configure_chart()
        if show and chart is None:
            self.chart.show(block=block)
            if not block:
                _watch_chart_close(self.chart)

    @property
    def instruments(self) -> list[str]:
        """Return instrument IDs currently displayed."""
        return list(self._series)

    def add_instrument(self, query: str, *, curr: str | None = None) -> "InstrumentPlot":
        """Add an instrument by ID, name fragment, ticker, or unique query.

        ``curr=None`` follows the plot's selected currency. Use
        ``curr="original"`` to keep the instrument in its native currency.
        """
        instrument = self._resolve_instrument(query)
        if instrument.instrument_id in self._series:
            self._refresh_instrument(instrument.instrument_id, curr=curr)
            return self

        color = self.colors[len(self._series) % len(self.colors)]
        line = self.chart.create_line(
            name=self._series_name(instrument, curr=curr),
            color=color,
            width=2,
            price_line=False,
            price_label=True,
        )
        self._series[instrument.instrument_id] = _InstrumentSeries(
            instrument=instrument,
            line=line,
            color=color,
            curr=curr,
        )
        self._set_line_data(instrument, line, curr=curr)
        self._apply_legend()
        self.chart.fit()
        return self

    def remove_instrument(self, query: str) -> "InstrumentPlot":
        """Remove an instrument from the chart."""
        instrument = self._resolve_plotted_instrument(query)
        item = self._series.pop(instrument.instrument_id, None)
        if item is not None:
            self._hide_line(item.line)
            self._apply_legend()
            self.chart.fit()
        return self

    def relative_chart(self) -> "InstrumentPlot":
        """Show series as percentage performance from the visible baseline."""
        self._relative = True
        self.chart.price_scale(mode="percentage")
        self._apply_legend()
        self.chart.fit()
        return self

    def absolute_chart(self) -> "InstrumentPlot":
        """Show series as absolute prices/NAVs in the selected currency."""
        self._relative = False
        self.chart.price_scale(mode="normal")
        self._apply_legend()
        self.chart.fit()
        return self

    def set_currency(self, curr: str | None) -> "InstrumentPlot":
        """Change the default chart currency and refresh all displayed lines."""
        self.curr = curr.upper() if curr else None
        self._refresh_all()
        return self

    def set_range(self, *, start: DateLike | None = None, end: DateLike | None = None) -> "InstrumentPlot":
        """Change the date range and refresh all displayed lines."""
        self.start = to_timestamp(start).normalize() if start is not None else None
        self.end = to_timestamp(end).normalize() if end is not None else None
        self._refresh_all()
        return self

    def show(self, *, block: bool = True) -> "InstrumentPlot":
        """Show the chart window."""
        self.chart.show(block=block)
        return self

    def close(self) -> None:
        """Close the chart process.

        Call this before creating another chart after closing the window
        manually. The underlying package uses a singleton webview process, so
        this keeps REPL sessions from hitting "cannot start a process twice".
        """
        _close_chart(self.chart)

    def _new_chart(self, *, width: int, height: int) -> Any:
        try:
            from lightweight_charts import Chart
        except ImportError as exc:
            raise ImportError("Install lightweight-charts to use instrument_plot().") from exc
        self._reset_stopped_webview(Chart)
        return Chart(width=width, height=height, title="ConvexPM Instruments")

    def _reset_stopped_webview(self, chart_cls: Any) -> None:
        wv = getattr(chart_cls, "WV", None)
        process = getattr(wv, "wv_process", None)
        if wv is None or process is None:
            return
        if getattr(process, "_popen", None) is not None:
            wv.exit()

    def _configure_chart(self) -> None:
        self.chart.layout(background_color="#101214", text_color="#eef2f3")
        self.chart.grid(vert_enabled=True, horz_enabled=True, color="#303840")
        self.chart.price_scale(mode="normal")
        self._apply_legend()

    def _apply_legend(self) -> None:
        self.chart.legend(
            visible=True,
            ohlc=False,
            percent=self._relative,
            lines=True,
        )

    def _resolve_instrument(self, query: str) -> Instrument:
        query_text = str(query).strip()
        if not query_text:
            raise KeyError("Instrument query is empty")
        try:
            return self.registry.get(query_text)
        except KeyError:
            pass

        query_lower = query_text.lower()
        matches: list[Instrument] = []
        for instrument in self.registry.all():
            values = [
                instrument.instrument_id,
                instrument.name,
                instrument.data_symbol or "",
                str(instrument.metadata.get("ibkr_symbol", "")),
                str(instrument.metadata.get("yahoo_symbol", "")),
            ]
            if any(query_lower == value.lower() for value in values if value):
                return instrument
            if any(query_lower in value.lower() for value in values if value):
                matches.append(instrument)

        unique = {instrument.instrument_id: instrument for instrument in matches}
        if len(unique) == 1:
            return next(iter(unique.values()))
        if len(unique) > 1:
            names = [f"{instrument.instrument_id} ({instrument.name})" for instrument in unique.values()]
            raise KeyError(f"Ambiguous instrument query {query!r}. Matches: {names}")
        raise KeyError(f"Unknown instrument query: {query!r}")

    def _resolve_plotted_instrument(self, query: str) -> Instrument:
        query_text = str(query).strip()
        if not query_text:
            raise KeyError("Instrument query is empty")

        plotted = [item.instrument for item in self._series.values()]
        query_lower = query_text.lower()
        matches = []
        for instrument in plotted:
            values = [
                instrument.instrument_id,
                instrument.name,
                instrument.data_symbol or "",
                str(instrument.metadata.get("ibkr_symbol", "")),
                str(instrument.metadata.get("yahoo_symbol", "")),
            ]
            if any(query_lower == value.lower() for value in values if value):
                return instrument
            if any(query_lower in value.lower() for value in values if value):
                matches.append(instrument)

        unique = {instrument.instrument_id: instrument for instrument in matches}
        if len(unique) == 1:
            return next(iter(unique.values()))
        if len(unique) > 1:
            names = [f"{instrument.instrument_id} ({instrument.name})" for instrument in unique.values()]
            raise KeyError(f"Ambiguous plotted instrument query {query!r}. Matches: {names}")

        return self._resolve_instrument(query)

    def _refresh_all(self) -> None:
        for instrument_id in list(self._series):
            self._refresh_instrument(instrument_id)
        self.chart.fit()

    def _refresh_instrument(self, instrument_id: str, *, curr: str | None = None) -> None:
        item = self._series[instrument_id]
        if curr is not None:
            item.curr = curr
        line_name = self._series_name(item.instrument, curr=item.curr)
        if getattr(item.line, "name", None) == line_name:
            self._set_line_data(item.instrument, item.line, curr=item.curr)
            self.chart.fit()
            return

        self._hide_line(item.line)
        line = self.chart.create_line(
            name=line_name,
            color=item.color,
            width=2,
            price_line=False,
            price_label=True,
        )
        item.line = line
        self._set_line_data(item.instrument, line, curr=item.curr)
        self._apply_legend()
        self.chart.fit()

    def _hide_line(self, line: Any) -> None:
        if hasattr(line, "set"):
            try:
                line.set(None)
            except TypeError:
                line.set(pd.DataFrame())
        if hasattr(line, "hide_data"):
            line.hide_data()

    def _set_line_data(self, instrument: Instrument, line: Any, *, curr: str | None = None) -> None:
        data = self._history_for(instrument, curr=curr)
        if data.empty:
            raise KeyError(f"No saved market data for {instrument.instrument_id}")
        data = data.rename(columns={"value": line.name})
        try:
            line.set(data, format_cols=False)
        except TypeError:
            line.set(data)

    def _history_for(self, instrument: Instrument, *, curr: str | None = None) -> pd.DataFrame:
        history = self.market_data.get_history(instrument.instrument_id, start=self.start, end=self.end)
        if history.empty:
            return pd.DataFrame(columns=["time", "value"])

        target_currency = self._target_currency(instrument, curr=curr)
        history = history.loc[:, ["date", "value", "currency"]].copy()
        history["currency"] = history["currency"].astype(str).str.upper()
        if target_currency:
            history["value"] = [
                self._convert_value(float(row.value), row.currency, target_currency, row.date)
                for row in history.itertuples(index=False)
            ]
            history = history.dropna(subset=["value"])

        return (
            history.assign(time=history["date"].dt.strftime("%Y-%m-%d"))
            .loc[:, ["time", "value"]]
            .sort_values("time")
            .reset_index(drop=True)
        )

    def _convert_value(self, value: float, currency: str, target_currency: str, date: pd.Timestamp) -> float | None:
        if currency == target_currency:
            return value
        try:
            rate, _ = fx_rate_to_base(currency, target_currency, self.market_data, date, skip_missing_prices=True)
        except (KeyError, ZeroDivisionError):
            return None
        if rate is None:
            return None
        return value * rate

    def _series_name(self, instrument: Instrument, *, curr: str | None = None) -> str:
        target_currency = self._target_currency(instrument, curr=curr)
        return f"{instrument.name} ({target_currency})"

    def _target_currency(self, instrument: Instrument, *, curr: str | None = None) -> str:
        if isinstance(curr, str) and curr.strip().lower() == "original":
            return instrument.currency.upper()
        return (curr or self.curr or instrument.currency).upper()


class PortfolioChart:
    """Interactive portfolio chart with contribution, volatility, and drawdown guides."""

    def __init__(
        self,
        portfolio: Any,
        *,
        start: DateLike | None = None,
        end: DateLike | None = None,
        horizon: int = 90,
        chart: Any | None = None,
        show: bool = True,
        block: bool = False,
        width: int = 1200,
        height: int = 720,
        drawdown_tolerance: float = 0.15,
        periods_per_year: int = 252,
        fund_nav: bool = False,
        fund_nav_base: float = 100.0,
    ) -> None:
        self.portfolio = portfolio
        self.start = to_timestamp(start).normalize() if start is not None else None
        self.end = to_timestamp(end).normalize() if end is not None else None
        self.horizon = int(horizon)
        self.drawdown_tolerance = float(drawdown_tolerance)
        self.periods_per_year = int(periods_per_year)
        self.fund_nav = bool(fund_nav)
        self.fund_nav_base = float(fund_nav_base)
        self.chart = chart if chart is not None else self._new_chart(width=width, height=height)
        self._lines: dict[str, Any] = {}
        self._data = pd.DataFrame()
        self._configure_chart()
        self._create_lines()
        self._refresh()
        if show and chart is None:
            self.chart.show(block=block)
            if not block:
                _watch_chart_close(self.chart)

    def show(self, *, block: bool = True) -> "PortfolioChart":
        """Show the chart window."""
        self.chart.show(block=block)
        return self

    def close(self) -> None:
        """Close the chart process."""
        _close_chart(self.chart)

    @property
    def data(self) -> pd.DataFrame:
        """Return the currently plotted data."""
        return self._data.copy()

    def _new_chart(self, *, width: int, height: int) -> Any:
        try:
            from lightweight_charts import Chart
        except ImportError as exc:
            raise ImportError("Install lightweight-charts to use portfolio_chart().") from exc
        InstrumentPlot._reset_stopped_webview(self, Chart)
        suffix = "Fund NAV" if self.fund_nav else "NAV"
        return Chart(width=width, height=height, title=f"{self.portfolio.name} {suffix}")

    def _configure_chart(self) -> None:
        self.chart.layout(background_color="#101214", text_color="#eef2f3")
        self.chart.grid(vert_enabled=True, horz_enabled=True, color="#303840")
        self.chart.price_scale(mode="normal")
        self.chart.legend(visible=True, ohlc=False, percent=False, lines=True)

    def _create_lines(self) -> None:
        main_name = "Fund NAV" if self.fund_nav else "NAV"
        specs = [
            ("nav", main_name, "#f1b84b", 3),
            ("upper_2sigma", "+2σ Vol Envelope", "#7c8a96", 1),
            ("upper_1sigma", "+1σ Vol Envelope", "#9aa7b2", 1),
            ("lower_1sigma", "-1σ Vol Envelope", "#9aa7b2", 1),
            ("lower_2sigma", "-2σ Vol Envelope", "#7c8a96", 1),
            ("drawdown_floor", "Drawdown Tolerance Floor", "#ff6f91", 2),
        ]
        if not self.fund_nav:
            specs.insert(1, ("net_contributions", "Cumulative Net Contributions", "#4da3ff", 2))
        for key, name, color, width in specs:
            self._lines[key] = self.chart.create_line(
                name=name,
                color=color,
                width=width,
                price_line=False,
                price_label=True,
            )

    def _refresh(self) -> None:
        data = self._build_data()
        self._data = data
        for key, line in self._lines.items():
            line_data = data.loc[:, ["time", key]].rename(columns={key: line.name})
            try:
                line.set(line_data, format_cols=False)
            except TypeError:
                line.set(line_data)
        self.chart.fit()

    def _build_data(self) -> pd.DataFrame:
        nav = _portfolio_nav_observations(
            self.portfolio,
            start=self.start,
            end=self.end,
        )
        if nav.empty:
            return pd.DataFrame(
                columns=[
                    "time",
                    "nav",
                    "net_contributions",
                    "upper_2sigma",
                    "upper_1sigma",
                    "lower_1sigma",
                    "lower_2sigma",
                    "drawdown_floor",
                ]
            )
        data = nav.loc[:, ["date", "nav"]].copy().sort_values("date").reset_index(drop=True)
        data["date"] = pd.to_datetime(data["date"]).dt.normalize()
        if self.fund_nav:
            data["nav"] = _portfolio_fund_nav_series(
                self.portfolio,
                nav=data["nav"],
                dates=pd.DatetimeIndex(data["date"]),
                base_value=self.fund_nav_base,
            )
        data["net_contributions"] = _cumulative_net_contributions(
            self.portfolio,
            pd.DatetimeIndex(data["date"]),
        )

        returns = data["nav"].pct_change(fill_method=None)
        rolling_vol = returns.rolling(self.horizon, min_periods=2).std()
        one_sigma = data["nav"] * rolling_vol.fillna(0.0)
        data["upper_1sigma"] = data["nav"] + one_sigma
        data["lower_1sigma"] = data["nav"] - one_sigma
        data["upper_2sigma"] = data["nav"] + 2.0 * one_sigma
        data["lower_2sigma"] = data["nav"] - 2.0 * one_sigma
        high_water = data["nav"].cummax()
        data["drawdown_floor"] = high_water * (1.0 - self.drawdown_tolerance)
        data["time"] = data["date"].dt.strftime("%Y-%m-%d")
        return data.loc[
            :,
            [
                "time",
                "nav",
                "net_contributions",
                "upper_2sigma",
                "upper_1sigma",
                "lower_1sigma",
                "lower_2sigma",
                "drawdown_floor",
            ],
        ]


def _portfolio_fund_nav_series(
    portfolio: Any,
    *,
    nav: pd.Series,
    dates: pd.DatetimeIndex,
    base_value: float,
) -> pd.Series:
    if dates.empty:
        return pd.Series(dtype=float)
    if base_value <= 0:
        raise ValueError("fund_nav_base must be positive")

    flows = _external_flows_by_date(portfolio)
    nav_values = pd.Series(nav, dtype=float).reset_index(drop=True)
    unit_values: list[float] = []
    units = 0.0
    previous_unit_value = float(base_value)

    for idx, dt in enumerate(dates):
        current_nav = float(nav_values.iloc[idx])
        flow = flows.get(pd.Timestamp(dt).normalize(), 0.0)
        if idx == 0 or units <= 1e-12:
            previous_unit_value = float(base_value)
            units = current_nav / previous_unit_value if abs(current_nav) > 1e-12 else 0.0
            unit_values.append(previous_unit_value)
            continue

        ex_flow_nav = current_nav - flow
        ex_flow_unit_value = ex_flow_nav / units
        if abs(ex_flow_unit_value) <= 1e-12:
            unit_values.append(previous_unit_value)
            continue

        units += flow / ex_flow_unit_value
        if units <= 1e-12:
            units = 0.0
            previous_unit_value = ex_flow_unit_value
            unit_values.append(ex_flow_unit_value)
            continue

        previous_unit_value = current_nav / units
        unit_values.append(previous_unit_value)

    return pd.Series(unit_values, index=dates, dtype=float).reset_index(drop=True)


def _cumulative_net_contributions(portfolio: Any, dates: pd.DatetimeIndex) -> pd.Series:
    flows = _external_flows_by_date(portfolio)
    values = []
    running = 0.0
    for dt in dates:
        running += flows.get(pd.Timestamp(dt).normalize(), 0.0)
        values.append(running)
    return pd.Series(values, index=dates, dtype=float).reset_index(drop=True)


def _portfolio_nav_observations(
    portfolio: Any,
    *,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> pd.DataFrame:
    dates = _portfolio_observation_dates(portfolio, start=start, end=end)
    if dates.empty:
        return pd.DataFrame(columns=["date", "nav"])
    return _portfolio_nav_for_dates(portfolio, dates)


def _portfolio_nav_for_dates(portfolio: Any, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Calculate NAV for chart observation dates without repeated store reads."""
    dates = pd.DatetimeIndex(pd.to_datetime(dates).normalize().drop_duplicates().sort_values())
    if dates.empty:
        return pd.DataFrame(columns=["date", "nav"])

    instrument_ids = sorted({trade.instrument_id for trade in portfolio.trades})
    market = portfolio.market_data.read_all()
    market = market.copy()
    market["date"] = pd.to_datetime(market["date"]).dt.normalize()

    values = _forward_filled_market_frame(market, dates, value_column="value")
    currencies = _forward_filled_market_frame(market, dates, value_column="currency")
    holdings = _holdings_frame_for_dates(portfolio, dates, instrument_ids)

    values = values.reindex(columns=instrument_ids)
    currencies = currencies.reindex(columns=instrument_ids)
    local_market_values = holdings * values

    held_without_prices = (holdings.abs() > 1e-12) & values.isna()
    if held_without_prices.any().any():
        row_idx, col_idx = held_without_prices.to_numpy().nonzero()
        missing_date = dates[int(row_idx[0])].date()
        missing_instrument = held_without_prices.columns[int(col_idx[0])]
        raise KeyError(f"No market data for {missing_instrument} on or before {missing_date}")

    fx_rates = _fx_rate_frame_for_dates(
        market,
        dates,
        currencies=currencies,
        holdings=holdings,
        base_currency=portfolio.base_currency,
    )
    nav = (local_market_values * fx_rates).sum(axis=1)
    return pd.DataFrame({"date": dates, "nav": nav.to_numpy(dtype=float)}, columns=["date", "nav"])


def _forward_filled_market_frame(
    market: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    value_column: str,
) -> pd.DataFrame:
    if market.empty:
        return pd.DataFrame(index=dates)
    wide = (
        market.sort_values(["instrument_id", "date"])
        .pivot_table(index="date", columns="instrument_id", values=value_column, aggfunc="last")
        .sort_index()
    )
    index = dates.union(wide.index[wide.index <= dates.max()])
    return wide.reindex(index).ffill().reindex(dates)


def _holdings_frame_for_dates(
    portfolio: Any,
    dates: pd.DatetimeIndex,
    instrument_ids: list[str],
) -> pd.DataFrame:
    if not instrument_ids:
        return pd.DataFrame(index=dates)

    trade_rows = []
    for trade in portfolio.trades:
        trade_date = pd.Timestamp(trade.date).normalize()
        if trade_date > dates.max():
            continue
        trade_rows.append(
            {
                "date": trade_date,
                "instrument_id": trade.instrument_id,
                "quantity": signed_trade_quantity(trade, market_data=portfolio.market_data),
            }
        )

    if not trade_rows:
        return pd.DataFrame(0.0, index=dates, columns=instrument_ids)

    trades = pd.DataFrame(trade_rows)
    daily_trades = trades.pivot_table(
        index="date",
        columns="instrument_id",
        values="quantity",
        aggfunc="sum",
        fill_value=0.0,
    )
    index = dates.union(daily_trades.index)
    return (
        daily_trades.reindex(index, fill_value=0.0)
        .sort_index()
        .cumsum()
        .reindex(dates)
        .reindex(columns=instrument_ids, fill_value=0.0)
    )


def _fx_rate_frame_for_dates(
    market: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    currencies: pd.DataFrame,
    holdings: pd.DataFrame,
    base_currency: str,
) -> pd.DataFrame:
    base_currency = base_currency.upper()
    rates = pd.DataFrame(1.0, index=dates, columns=holdings.columns)
    market_values = _forward_filled_market_frame(market, dates, value_column="value")

    for instrument_id in holdings.columns:
        held_mask = holdings[instrument_id].abs() > 1e-12
        if not held_mask.any():
            continue
        instrument_currencies = currencies[instrument_id].dropna().astype(str).str.upper().unique()
        for currency in instrument_currencies:
            if currency == base_currency:
                continue
            currency_mask = held_mask & (currencies[instrument_id].astype(str).str.upper() == currency)
            if not currency_mask.any():
                continue
            rates.loc[currency_mask, instrument_id] = _fx_rates_for_currency(
                market_values,
                dates,
                currency=currency,
                base_currency=base_currency,
                required_mask=currency_mask,
            ).loc[currency_mask]
    return rates


def _fx_rates_for_currency(
    market_values: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    currency: str,
    base_currency: str,
    required_mask: pd.Series,
) -> pd.Series:
    direct_pair = f"{currency}{base_currency}"
    inverse_pair = f"{base_currency}{currency}"

    if direct_pair in market_values:
        rates = pd.to_numeric(market_values[direct_pair], errors="coerce")
    elif inverse_pair in market_values:
        inverse_rates = pd.to_numeric(market_values[inverse_pair], errors="coerce")
        if (inverse_rates.loc[required_mask] == 0).any():
            raise ZeroDivisionError(f"FX rate for {inverse_pair} is zero")
        rates = 1.0 / inverse_rates
    else:
        first_required_date = dates[required_mask.to_numpy()][0].date()
        raise KeyError(f"No FX market data for {direct_pair} or {inverse_pair} on or before {first_required_date}")

    missing_required = required_mask & rates.isna()
    if missing_required.any():
        first_required_date = dates[missing_required.to_numpy()][0].date()
        raise KeyError(f"No FX market data for {direct_pair} or {inverse_pair} on or before {first_required_date}")
    return rates


def _portfolio_observation_dates(
    portfolio: Any,
    *,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> pd.DatetimeIndex:
    instrument_ids = {trade.instrument_id for trade in portfolio.trades}
    if not instrument_ids:
        return pd.DatetimeIndex([])

    market = portfolio.market_data.read_all()
    market = market.loc[market["instrument_id"].isin(instrument_ids)].copy()
    if market.empty:
        return pd.DatetimeIndex([])
    market["date"] = pd.to_datetime(market["date"]).dt.normalize()

    start_ts = start or pd.Timestamp(min(trade.date for trade in portfolio.trades)).normalize()
    end_ts = end or market["date"].max()
    dates = market.loc[(market["date"] >= start_ts) & (market["date"] <= end_ts), "date"]

    trade_dates = pd.Series(
        [
            pd.Timestamp(trade.date).normalize()
            for trade in portfolio.trades
            if start_ts <= pd.Timestamp(trade.date).normalize() <= end_ts
        ],
        dtype="datetime64[ns]",
    )
    combined = pd.concat([dates, trade_dates], ignore_index=True).drop_duplicates().sort_values()
    return pd.DatetimeIndex(combined)


def _external_flows_by_date(portfolio: Any) -> dict[pd.Timestamp, float]:
    from convexpm.analytics.returns import cash_settlement_parent_ids, is_external_capital_flow

    flows: dict[pd.Timestamp, float] = {}
    settlement_parent_ids = cash_settlement_parent_ids(portfolio.trades)
    for trade in portfolio.trades:
        if not is_external_capital_flow(trade, settlement_parent_ids=settlement_parent_ids):
            continue
        amount = _trade_amount_in_base(portfolio, trade)
        if amount is None:
            continue
        signed = amount + float(trade.fees) if trade.side == "BUY" else -(amount - float(trade.fees))
        timestamp = pd.Timestamp(trade.date).normalize()
        flows[timestamp] = flows.get(timestamp, 0.0) + signed
    return flows


def _trade_amount_in_base(portfolio: Any, trade: Any) -> float | None:
    amount = trade.amount
    if amount is None and trade.quantity is not None and trade.value_per_unit is not None:
        amount = float(trade.quantity) * float(trade.value_per_unit)
    if amount is None and trade.quantity is not None:
        try:
            amount = float(trade.quantity) * float(portfolio.market_data.get_value(trade.instrument_id, trade.date))
        except (KeyError, TypeError):
            return None
    if amount is None:
        return None
    currency = (trade.currency or portfolio.base_currency).upper()
    if currency == portfolio.base_currency:
        return float(amount)
    try:
        rate, _ = fx_rate_to_base(currency, portfolio.base_currency, portfolio.market_data, trade.date)
    except KeyError:
        return float(amount)
    return float(amount) * float(rate)


def instrument_plot(**kwargs: Any) -> InstrumentPlot:
    """Open an interactive chart for saved instrument market data."""
    return InstrumentPlot(**kwargs)


def portfolio_chart(portfolio: Any, **kwargs: Any) -> PortfolioChart:
    """Open an interactive portfolio NAV chart."""
    return PortfolioChart(portfolio, **kwargs)
