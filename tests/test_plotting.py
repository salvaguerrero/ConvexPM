import pandas as pd
import pytest

from datetime import date

from convexpm import Instrument, InstrumentRegistry, MarketDataStore, Portfolio, Trade, instrument_plot, portfolio_chart


class FakeLine:
    def __init__(self, name):
        self.name = name
        self.data = None
        self.deleted = False
        self.hidden = False
        self.format_cols = None

    def set(self, data, format_cols=True):
        self.format_cols = format_cols
        self.data = None if data is None else data.copy()

    def delete(self):
        self.deleted = True

    def hide_data(self):
        self.hidden = True


class FakeChart:
    def __init__(self):
        self.lines = []
        self.price_scale_modes = []
        self.legend_calls = []
        self.fit_count = 0

    def create_line(self, name="", **kwargs):
        line = FakeLine(name)
        self.lines.append(line)
        return line

    def layout(self, **kwargs):
        pass

    def grid(self, **kwargs):
        pass

    def legend(self, **kwargs):
        self.legend_calls.append(kwargs)

    def price_scale(self, **kwargs):
        self.price_scale_modes.append(kwargs.get("mode"))

    def fit(self):
        self.fit_count += 1

    def show(self, block=False):
        pass


def _registry_and_store(tmp_path):
    registry = InstrumentRegistry(tmp_path / "instruments.parquet", auto_load=False)
    registry.add(
        Instrument(
            instrument_id="REP",
            name="Repsol, S.A.",
            instrument_type="equity",
            asset_class="equity",
            currency="EUR",
            data_source="fixture",
            data_symbol="REP.MC",
        )
    )
    registry.add(
        Instrument(
            instrument_id="CVX",
            name="Chevron",
            instrument_type="equity",
            asset_class="equity",
            currency="USD",
            data_source="fixture",
            data_symbol="CVX",
        )
    )
    store = MarketDataStore(tmp_path / "market.parquet", registry=registry)
    store.upsert(
        pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2026-01-01",
                        "2026-01-02",
                        "2026-01-01",
                        "2026-01-02",
                        "2026-01-01",
                        "2026-01-02",
                    ]
                ),
                "instrument_id": ["REP", "REP", "CVX", "CVX", "USDEUR", "USDEUR"],
                "value": [14.0, 15.0, 100.0, 110.0, 0.9, 0.8],
                "currency": ["EUR", "EUR", "USD", "USD", "EUR", "EUR"],
                "value_type": ["close", "close", "close", "close", "fx", "fx"],
                "source": ["fixture"] * 6,
            }
        )
    )
    return registry, store


def test_instrument_plot_adds_converted_and_original_series(tmp_path):
    registry, store = _registry_and_store(tmp_path)
    chart = FakeChart()

    plot = instrument_plot(
        start="2026-01-01",
        curr="EUR",
        registry=registry,
        market_data=store,
        chart=chart,
        show=False,
    )

    plot.add_instrument("Repsol")
    plot.add_instrument("Chevron")
    plot.add_instrument("Chevron", curr="original")

    assert plot.instruments == ["REP", "CVX"]
    assert chart.lines[0].name == "Repsol, S.A. (EUR)"
    assert chart.lines[0].format_cols is False
    assert chart.lines[0].data["time"].tolist() == ["2026-01-01", "2026-01-02"]
    assert chart.lines[0].data["Repsol, S.A. (EUR)"].tolist() == [14.0, 15.0]
    assert chart.lines[1].name == "Chevron (EUR)"
    assert chart.lines[1].deleted is False
    assert chart.lines[1].hidden is True
    assert chart.lines[2].name == "Chevron (USD)"
    assert chart.lines[2].data["Chevron (USD)"].tolist() == [100.0, 110.0]


def test_instrument_plot_readding_same_instrument_updates_existing_line(tmp_path):
    registry, store = _registry_and_store(tmp_path)
    chart = FakeChart()
    plot = instrument_plot(
        start="2026-01-01",
        curr="EUR",
        registry=registry,
        market_data=store,
        chart=chart,
        show=False,
    )

    plot.add_instrument("REP")
    plot.add_instrument("REPSOL")

    assert plot.instruments == ["REP"]
    assert len(chart.lines) == 1
    assert chart.lines[0].deleted is False
    assert chart.lines[0].data["Repsol, S.A. (EUR)"].tolist() == [14.0, 15.0]


def test_instrument_plot_modes_and_remove(tmp_path):
    registry, store = _registry_and_store(tmp_path)
    chart = FakeChart()
    plot = instrument_plot(registry=registry, market_data=store, chart=chart, show=False)

    plot.add_instrument("REP")
    plot.relative_chart()
    plot.absolute_chart()
    plot.remove_instrument("Repsol")

    assert chart.price_scale_modes[-2:] == ["percentage", "normal"]
    assert plot.instruments == []
    assert chart.lines[-1].deleted is False
    assert chart.lines[-1].hidden is True


def test_instrument_plot_remove_matches_id_name_or_ticker(tmp_path):
    registry, store = _registry_and_store(tmp_path)

    by_id = instrument_plot(registry=registry, market_data=store, chart=FakeChart(), show=False)
    by_id.add_instrument("REP")
    by_id.remove_instrument("REP.MC")
    assert by_id.instruments == []

    by_ticker = instrument_plot(registry=registry, market_data=store, chart=FakeChart(), show=False)
    by_ticker.add_instrument("REP.MC")
    by_ticker.remove_instrument("Repsol")
    assert by_ticker.instruments == []

    by_name = instrument_plot(registry=registry, market_data=store, chart=FakeChart(), show=False)
    by_name.add_instrument("Repsol")
    by_name.remove_instrument("REP")
    assert by_name.instruments == []


def test_instrument_plot_ambiguous_query_raises(tmp_path):
    registry, store = _registry_and_store(tmp_path)
    registry.add(
        Instrument(
            instrument_id="REP2",
            name="Repsol Brazil",
            instrument_type="equity",
            asset_class="equity",
            currency="EUR",
            data_source="fixture",
        )
    )
    plot = instrument_plot(registry=registry, market_data=store, chart=FakeChart(), show=False)

    with pytest.raises(KeyError, match="Ambiguous"):
        plot.add_instrument("Repso")


def test_portfolio_chart_plots_nav_contributions_and_risk_bands(tmp_path):
    registry = InstrumentRegistry(tmp_path / "instruments.parquet", auto_load=False)
    registry.add(
        Instrument(
            instrument_id="A",
            name="Asset A",
            instrument_type="equity",
            asset_class="equity",
            currency="EUR",
            data_source="fixture",
        )
    )
    store = MarketDataStore(tmp_path / "market.parquet", registry=registry)
    store.upsert(
        pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]
                ),
                "instrument_id": ["A", "A", "A", "A", "A"],
                "value": [100.0, 110.0, 105.0, 120.0, 115.0],
                "currency": ["EUR"] * 5,
                "value_type": ["close"] * 5,
                "source": ["fixture"] * 5,
            }
        )
    )
    portfolio = Portfolio(name="Chart", base_currency="EUR", registry=registry, market_data=store)
    portfolio.add_trades(
        [
            Trade(
                trade_id="deposit",
                date=date(2026, 1, 1),
                instrument_id="EUR_CASH",
                side="BUY",
                quantity=1_000.0,
                amount=1_000.0,
                value_per_unit=1.0,
                currency="EUR",
                metadata={"source": "manual"},
            ),
            Trade(
                trade_id="asset-buy",
                date=date(2026, 1, 1),
                instrument_id="A",
                side="BUY",
                quantity=1.0,
                amount=100.0,
                value_per_unit=100.0,
                currency="EUR",
                metadata={"source": "manual"},
            ),
        ],
        cash_ledger=True,
    )
    chart = FakeChart()

    plot = portfolio_chart(
        portfolio,
        start="2026-01-01",
        end="2026-01-05",
        horizon=2,
        drawdown_tolerance=0.10,
        chart=chart,
        show=False,
    )

    data = plot.data
    assert chart.lines[0].name == "NAV"
    assert chart.lines[0].data["NAV"].tolist() == [1000.0, 1010.0, 1005.0, 1020.0, 1015.0]
    assert data["net_contributions"].tolist() == [1000.0] * 5
    assert data["drawdown_floor"].tolist() == [900.0, 909.0, 909.0, 918.0, 918.0]
    assert data.loc[2, "upper_1sigma"] > data.loc[2, "nav"]
    assert data.loc[2, "lower_1sigma"] < data.loc[2, "nav"]

    assert not hasattr(plot, "rolling_30")
    assert not hasattr(plot, "rolling_90")
    assert not hasattr(plot, "rolling_365")
    assert not hasattr(plot, "set_horizon")
    assert not hasattr(plot, "set_drawdown_tolerance")


def test_portfolio_chart_can_plot_fund_nav_without_cash_flow_jumps(tmp_path):
    registry = InstrumentRegistry(tmp_path / "instruments.parquet", auto_load=False)
    registry.add(
        Instrument(
            instrument_id="A",
            name="Asset A",
            instrument_type="equity",
            asset_class="equity",
            currency="EUR",
            data_source="fixture",
        )
    )
    store = MarketDataStore(tmp_path / "market.parquet", registry=registry)
    store.upsert(
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
                "instrument_id": ["A", "A", "A"],
                "value": [100.0, 110.0, 105.0],
                "currency": ["EUR"] * 3,
                "value_type": ["close"] * 3,
                "source": ["fixture"] * 3,
            }
        )
    )
    portfolio = Portfolio(name="Chart", base_currency="EUR", registry=registry, market_data=store)
    portfolio.add_trades(
        [
            Trade(
                trade_id="initial-deposit",
                date=date(2026, 1, 1),
                instrument_id="EUR_CASH",
                side="BUY",
                quantity=100.0,
                amount=100.0,
                value_per_unit=1.0,
                currency="EUR",
                metadata={"source": "manual"},
            ),
            Trade(
                trade_id="asset-buy",
                date=date(2026, 1, 1),
                instrument_id="A",
                side="BUY",
                quantity=1.0,
                amount=100.0,
                value_per_unit=100.0,
                currency="EUR",
                metadata={"source": "manual"},
            ),
            Trade(
                trade_id="later-deposit",
                date=date(2026, 1, 3),
                instrument_id="EUR_CASH",
                side="BUY",
                quantity=100.0,
                amount=100.0,
                value_per_unit=1.0,
                currency="EUR",
                metadata={"source": "manual"},
            ),
        ],
        cash_ledger=True,
    )

    plot = portfolio_chart(
        portfolio,
        start="2026-01-01",
        end="2026-01-03",
        chart=FakeChart(),
        show=False,
        fund_nav=True,
    )

    assert plot.chart.lines[0].name == "Fund NAV"
    assert plot.data["nav"].tolist() == pytest.approx([100.0, 110.0, 105.0])


def test_portfolio_chart_is_available_from_portfolio(tmp_path):
    registry, store = _registry_and_store(tmp_path)
    portfolio = Portfolio(
        name="Chart",
        base_currency="EUR",
        registry=registry,
        market_data=store,
        trades=[Trade(date=date(2026, 1, 1), instrument_id="REP", side="BUY", quantity=1.0)],
    )

    plot = portfolio.chart(start="2026-01-01", end="2026-01-02", chart=FakeChart(), show=False)

    assert plot.data["nav"].tolist() == [14.0, 15.0]


def test_portfolio_chart_builds_nav_without_per_date_nav_calls(tmp_path):
    registry, store = _registry_and_store(tmp_path)
    portfolio = Portfolio(
        name="Chart",
        base_currency="EUR",
        registry=registry,
        market_data=store,
        trades=[Trade(date=date(2026, 1, 1), instrument_id="REP", side="BUY", quantity=1.0)],
    )

    def fail_nav(date=None):
        raise AssertionError("portfolio_chart should build NAV in bulk")

    portfolio.nav = fail_nav

    plot = portfolio_chart(start="2026-01-01", end="2026-01-02", portfolio=portfolio, chart=FakeChart(), show=False)

    assert plot.data["nav"].tolist() == [14.0, 15.0]
