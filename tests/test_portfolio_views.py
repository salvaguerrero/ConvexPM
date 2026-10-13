from datetime import date

import pandas as pd
import pytest

from convexpm import Instrument, InstrumentRegistry, MarketDataStore, Portfolio, Trade


def _two_asset_portfolio(tmp_path):
    registry = InstrumentRegistry(tmp_path / "instruments.parquet", auto_load=False)
    for instrument_id in ["A", "B"]:
        registry.add(
            Instrument(
                instrument_id=instrument_id,
                name=f"Asset {instrument_id}",
                instrument_type="equity",
                asset_class="equity",
                currency="EUR",
                data_source="fixture",
            )
        )

    dates = pd.date_range("2025-12-29", periods=7, freq="D")
    prices = {
        "A": [100.0, 102.0, 101.0, 103.0, 104.0, 106.0, 105.0],
        "B": [100.0, 99.0, 101.0, 100.0, 102.0, 101.0, 103.0],
    }
    rows = []
    for instrument_id, values in prices.items():
        for dt, value in zip(dates, values, strict=True):
            rows.append(
                {
                    "date": dt,
                    "instrument_id": instrument_id,
                    "value": value,
                    "currency": "EUR",
                    "value_type": "close",
                    "source": "fixture",
                }
            )
    store = MarketDataStore(tmp_path / "market.parquet")
    store.upsert(pd.DataFrame(rows))

    return Portfolio(
        name="Views",
        base_currency="EUR",
        trades=[
            Trade(date=date(2025, 12, 29), instrument_id="A", side="BUY", quantity=1),
            Trade(date=date(2025, 12, 29), instrument_id="B", side="BUY", quantity=1),
        ],
        registry=registry,
        market_data=store,
    )


def test_performance_view_reconciles_yearly_asset_contributions(tmp_path):
    portfolio = _two_asset_portfolio(tmp_path)

    view = portfolio.performance_view(start="2025-12-29", end="2026-01-04")

    assert list(view.yearly_return_contribution.columns) == ["2025 (%)", "2026 (%)", "Total (%)"]
    assert list(view.assets.columns) == [
        "name",
        "current_weight (%)",
        "asset_return (%)",
        "return_contribution (%)",
    ]
    assert list(view.summary.columns) == ["2025", "2026", "Total"]
    assert view.cumulative_return.iloc[-1] == pytest.approx(view.summary.loc["portfolio_return (%)", "Total"])
    assert view.summary.loc["cagr (%)", "Total"] != view.summary.loc["portfolio_return (%)", "Total"]
    assert view.summary.loc["capital_inflows", "Total"] == pytest.approx(200.0)
    assert view.summary.loc["delta_nav", "Total"] == pytest.approx(
        view.summary.loc["end_nav", "Total"] - view.summary.loc["start_nav", "Total"]
    )
    assert view.summary.loc["capital_gain", "Total"] == pytest.approx(view.summary.loc["delta_nav", "Total"])
    assert view.summary.loc["start_nav", "2025"] == pytest.approx(0.0)
    assert view.summary.loc["end_nav", "2025"] == pytest.approx(portfolio.nav(date(2025, 12, 31)))
    assert view.summary.loc["start_nav", "2026"] == pytest.approx(portfolio.nav(date(2026, 1, 1)))
    assert view.summary.loc["end_nav", "2026"] == pytest.approx(portfolio.nav(date(2026, 1, 4)))
    assert view.summary.loc["delta_nav", "2026"] == pytest.approx(
        view.summary.loc["end_nav", "2026"] - view.summary.loc["start_nav", "2026"]
    )
    assert view.summary.loc["capital_gain", "2026"] == pytest.approx(view.summary.loc["delta_nav", "2026"])
    assert list(view.summary.index[:23]) == [
        "portfolio_return (%)",
        "benchmark_return (%)",
        "cagr (%)",
        "",
        "sharpe",
        "benchmark_sharpe",
        "volatility (%)",
        "benchmark_volatility (%)",
        "max_drawdown (%)",
        "benchmark_max_drawdown (%)",
        "",
        "alpha (%)",
        "beta",
        "",
        "start_nav",
        "end_nav",
        "delta_nav",
        "capital_gain",
        "",
        "capital_inflows",
        "capital_outflows",
        "net_capital_flow",
        "",
    ]
    assert "\n\n" in repr(view.summary)
    assert {
        "volatility (%)",
        "sharpe",
        "max_drawdown (%)",
        "benchmark_return (%)",
        "benchmark_sharpe",
        "benchmark_volatility (%)",
        "benchmark_max_drawdown (%)",
        "alpha (%)",
        "beta",
    }.issubset(set(view.summary.index))
    assert view.cumulative_return.name == "cumulative_return (%)"
    assert "Total (%)" in view.standalone_asset_returns.columns
    assert "Total (%)" in view.standalone_asset_volatility.columns
    assert "Total (%)" in view.standalone_asset_drawdown.columns
    assert "Total" in view.standalone_asset_beta.columns
    assert "Total (%)" in view.asset_weights.columns
    assert list(view.asset_net_inflows.columns) == ["2025", "2026", "Total"]
    assert list(view.asset_valuation.columns) == ["2025", "2026", "Total"]
    assert "Portfolio Weight" in view.asset_weights.index
    assert "Portfolio Net Inflow" in view.asset_net_inflows.index
    assert "Portfolio Valuation" in view.asset_valuation.index
    assert view.asset_net_inflows.loc["Asset A", "2025"] == pytest.approx(100.0)
    assert view.asset_net_inflows.loc["Asset B", "2025"] == pytest.approx(100.0)
    assert view.asset_net_inflows.loc["Asset A", "2026"] == pytest.approx(0.0)
    assert view.asset_net_inflows.loc["Portfolio Net Inflow", "Total"] == pytest.approx(200.0)
    assert view.asset_valuation.loc["Asset A", "2025"] == pytest.approx(101.0)
    assert view.asset_valuation.loc["Asset B", "2025"] == pytest.approx(101.0)
    assert view.asset_valuation.loc["Asset A", "Total"] == pytest.approx(105.0)
    assert view.asset_valuation.loc["Asset B", "Total"] == pytest.approx(103.0)
    assert view.asset_valuation.loc["Portfolio Valuation", "Total"] == pytest.approx(208.0)
    assert "Asset A" in view.standalone_asset_returns.index
    assert "Portfolio Volatility" in view.volatility_contribution.index
    assert "Portfolio Drawdown" in view.drawdown_contribution.index
    assert view.return_correlation().shape == (2, 2)
    assert view.volatility_correlation(year=2026).shape == (2, 2)
    for year in view.return_contribution.columns:
        asset_total = view.return_contribution.loc[["Asset A", "Asset B"], year].sum()
        assert asset_total == pytest.approx(
            view.return_contribution.loc["Portfolio Return", year]
        )


def test_performance_view_counts_only_external_capital_flows_for_funded_trades(tmp_path):
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
    store = MarketDataStore(tmp_path / "market.parquet")
    store.upsert(
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                "instrument_id": ["A", "A"],
                "value": [100.0, 110.0],
                "currency": ["EUR", "EUR"],
                "value_type": ["close", "close"],
                "source": ["fixture", "fixture"],
            }
        )
    )
    portfolio = Portfolio(name="Funded", base_currency="EUR", registry=registry, market_data=store)
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

    view = portfolio.performance_view(start="2026-01-01", end="2026-01-02")

    assert view.summary.loc["capital_inflows", "Total"] == pytest.approx(1_000.0)
    assert view.summary.loc["capital_outflows", "Total"] == pytest.approx(0.0)


def test_performance_view_capital_gain_excludes_flows_already_in_start_nav(tmp_path):
    portfolio = Portfolio(
        name="Cash",
        base_currency="EUR",
        registry=InstrumentRegistry(tmp_path / "instruments.parquet", auto_load=False),
        market_data=MarketDataStore(tmp_path / "market.parquet"),
    )
    portfolio.add_trades(
        [
            Trade(
                trade_id="start-deposit",
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
                trade_id="later-deposit",
                date=date(2026, 1, 2),
                instrument_id="EUR_CASH",
                side="BUY",
                quantity=100.0,
                amount=100.0,
                value_per_unit=1.0,
                currency="EUR",
                metadata={"source": "manual"},
            ),
        ]
    )

    view = portfolio.performance_view(start="2026-01-01", end="2026-01-02")

    assert view.summary.loc["capital_inflows", "Total"] == pytest.approx(200.0)
    assert view.summary.loc["start_nav", "Total"] == pytest.approx(100.0)
    assert view.summary.loc["end_nav", "Total"] == pytest.approx(200.0)
    assert view.summary.loc["delta_nav", "Total"] == pytest.approx(100.0)
    assert view.summary.loc["capital_gain", "Total"] == pytest.approx(0.0)


def test_performance_view_uses_imported_renta4_cash_ledger_for_external_flows(tmp_path):
    registry = InstrumentRegistry(tmp_path / "instruments.parquet", auto_load=False)
    registry.add(
        Instrument(
            instrument_id="FONC",
            name="Renta 4 Foncuenta Ahorro, F.i.",
            instrument_type="fund",
            asset_class="cash",
            currency="EUR",
            data_source="fixture",
        )
    )
    registry.add(
        Instrument(
            instrument_id="WORLD_ENERGY",
            name="Bgf World Energy A2 Eur Hedged",
            instrument_type="fund",
            asset_class="fund",
            currency="EUR",
            data_source="fixture",
        )
    )
    store = MarketDataStore(tmp_path / "market.parquet", registry=registry)
    store.upsert(
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-01", "2026-08-24", "2026-09-22", "2026-09-24"] * 2),
                "instrument_id": ["FONC"] * 4 + ["WORLD_ENERGY"] * 4,
                "value": [10.0, 10.87, 10.88, 10.88, 8.5, 8.6, 8.7, 8.82],
                "currency": ["EUR"] * 8,
                "value_type": ["close"] * 8,
                "source": ["fixture"] * 8,
            }
        )
    )
    portfolio = Portfolio(name="Renta 4", base_currency="EUR", registry=registry, market_data=store)
    portfolio.add_trades(
        [
            Trade(
                trade_id="initial-foncuenta",
                date=date(2026, 1, 1),
                instrument_id="FONC",
                side="BUY",
                quantity=500.0,
                amount=5_000.0,
                value_per_unit=10.0,
                currency="EUR",
                account="renta4",
                tags=["renta4"],
                metadata={
                    "source": "renta4",
                    "operation_type": "SUSCRIPCIÓN",
                    "cash_ledger_source": "renta4_saldo_en_divisa",
                },
            ),
            Trade(
                trade_id="initial-transfer",
                date=date(2026, 1, 1),
                instrument_id="EUR_CASH",
                side="BUY",
                quantity=5_000.0,
                amount=5_000.0,
                value_per_unit=1.0,
                currency="EUR",
                account="renta4",
                tags=["renta4", "cash"],
                metadata={"source": "renta4_cash", "cash_flow_type": "transfer_in"},
            ),
            Trade(
                trade_id="initial-cash-to-foncuenta",
                date=date(2026, 1, 1),
                instrument_id="EUR_CASH",
                side="SELL",
                quantity=5_000.0,
                amount=5_000.0,
                value_per_unit=1.0,
                currency="EUR",
                account="renta4",
                tags=["renta4", "cash"],
                metadata={"source": "renta4_cash", "cash_flow_type": "fund_buy"},
            ),
            Trade(
                trade_id="cash-in-foncuenta",
                date=date(2026, 8, 24),
                instrument_id="FONC",
                side="BUY",
                quantity=275.93,
                amount=3_000.0,
                value_per_unit=10.87,
                currency="EUR",
                account="renta4",
                tags=["renta4"],
                metadata={
                    "source": "renta4",
                    "operation_type": "SUSCRIPCIÓN",
                    "cash_ledger_source": "renta4_saldo_en_divisa",
                },
            ),
            Trade(
                trade_id="cash-in-transfer",
                date=date(2026, 8, 24),
                instrument_id="EUR_CASH",
                side="BUY",
                quantity=3_000.0,
                amount=3_000.0,
                value_per_unit=1.0,
                currency="EUR",
                account="renta4",
                tags=["renta4", "cash"],
                metadata={"source": "renta4_cash", "cash_flow_type": "transfer_in"},
            ),
            Trade(
                trade_id="cash-in-cash-to-foncuenta",
                date=date(2026, 8, 24),
                instrument_id="EUR_CASH",
                side="SELL",
                quantity=3_000.0,
                amount=3_000.0,
                value_per_unit=1.0,
                currency="EUR",
                account="renta4",
                tags=["renta4", "cash"],
                metadata={"source": "renta4_cash", "cash_flow_type": "fund_buy"},
            ),
            Trade(
                trade_id="sell-foncuenta-for-switch",
                date=date(2026, 9, 22),
                instrument_id="FONC",
                side="SELL",
                quantity=459.47,
                amount=5_000.0,
                value_per_unit=10.88,
                currency="EUR",
                account="renta4",
                tags=["renta4"],
                metadata={
                    "source": "renta4",
                    "operation_type": "REEMBOLSO",
                    "cash_ledger_source": "renta4_saldo_en_divisa",
                },
            ),
            Trade(
                trade_id="switch-foncuenta-to-cash",
                date=date(2026, 9, 22),
                instrument_id="EUR_CASH",
                side="BUY",
                quantity=5_000.0,
                amount=5_000.0,
                value_per_unit=1.0,
                currency="EUR",
                account="renta4",
                tags=["renta4", "cash"],
                metadata={"source": "renta4_cash", "cash_flow_type": "fund_sell"},
            ),
            Trade(
                trade_id="buy-world-energy-from-switch",
                date=date(2026, 9, 24),
                instrument_id="WORLD_ENERGY",
                side="BUY",
                quantity=566.89,
                amount=5_000.0,
                value_per_unit=8.82,
                currency="EUR",
                account="renta4",
                tags=["renta4"],
                metadata={
                    "source": "renta4",
                    "operation_type": "SUSCRIPCIÓN",
                    "cash_ledger_source": "renta4_saldo_en_divisa",
                },
            ),
            Trade(
                trade_id="switch-cash-to-world-energy",
                date=date(2026, 9, 24),
                instrument_id="EUR_CASH",
                side="SELL",
                quantity=5_000.0,
                amount=5_000.0,
                value_per_unit=1.0,
                currency="EUR",
                account="renta4",
                tags=["renta4", "cash"],
                metadata={"source": "renta4_cash", "cash_flow_type": "fund_buy"},
            ),
        ]
    )

    view = portfolio.performance_view(start="2026-01-01", end="2026-09-24")

    assert view.summary.loc["capital_inflows", "Total"] == pytest.approx(8_000.0)
    assert view.summary.loc["capital_outflows", "Total"] == pytest.approx(0.0)
    assert view.summary.loc["net_capital_flow", "Total"] == pytest.approx(8_000.0)


def test_performance_view_returns_are_in_portfolio_base_currency(tmp_path):
    registry = InstrumentRegistry(tmp_path / "instruments.parquet", auto_load=False)
    registry.add(
        Instrument(
            instrument_id="USD_ASSET",
            name="USD Asset",
            instrument_type="fund",
            asset_class="fund",
            currency="USD",
            data_source="fixture",
        )
    )
    registry.add(
        Instrument(
            instrument_id="USDEUR",
            name="USD/EUR",
            instrument_type="fx",
            asset_class="fx",
            currency="EUR",
            data_source="fixture",
        )
    )
    store = MarketDataStore(tmp_path / "market.parquet")
    store.upsert(
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-02"]),
                "instrument_id": ["USD_ASSET", "USD_ASSET", "USDEUR", "USDEUR"],
                "value": [100.0, 110.0, 1.0, 0.5],
                "currency": ["USD", "USD", "EUR", "EUR"],
                "value_type": ["close", "close", "fx", "fx"],
                "source": ["fixture", "fixture", "fixture", "fixture"],
            }
        )
    )
    portfolio = Portfolio(
        name="FX",
        base_currency="EUR",
        registry=registry,
        market_data=store,
        trades=[
            Trade(
                date=date(2026, 1, 1),
                instrument_id="USD_ASSET",
                side="BUY",
                quantity=1,
                amount=100,
                value_per_unit=100,
                currency="USD",
            )
        ],
    )

    view = portfolio.performance_view(start="2026-01-01", end="2026-01-02")

    assert view.summary.loc["portfolio_return (%)", "Total"] == pytest.approx(-45.0)
    assert view.yearly_return_contribution.loc["USD Asset", "2026 (%)"] == pytest.approx(-45.0)
    assert view.standalone_asset_returns.loc["USD Asset", "2026 (%)"] == pytest.approx(-45.0)


def test_performance_view_supports_benchmark_alpha_beta(tmp_path):
    portfolio = _two_asset_portfolio(tmp_path)

    view = portfolio.performance_view(start="2025-12-29", end="2026-01-04", benchmark="A")

    assert view.summary.loc["benchmark_return (%)", "Total"] == pytest.approx(
        view.standalone_asset_returns.loc["Asset A", "Total (%)"]
    )
    assert pd.notna(view.summary.loc["benchmark_sharpe", "Total"])
    assert pd.notna(view.summary.loc["benchmark_volatility (%)", "Total"])
    assert pd.notna(view.summary.loc["benchmark_max_drawdown (%)", "Total"])
    assert view.standalone_asset_beta.loc["Asset A", "Total"] == pytest.approx(1.0)
    assert view.standalone_asset_alpha.loc["Asset A", "Total (%)"] == pytest.approx(0.0)


def test_portfolio_default_benchmark_is_persisted_and_used(tmp_path):
    portfolio = _two_asset_portfolio(tmp_path)
    portfolio.set_benchmark("A")
    portfolio.save("bench", root=tmp_path / "portfolios")

    loaded = Portfolio.load(
        "bench",
        root=tmp_path / "portfolios",
        registry=portfolio.registry,
        market_data=portfolio.market_data,
    )
    view = loaded.performance_view(start="2025-12-29", end="2026-01-04")

    assert loaded.benchmark == "A"
    assert view.summary.loc["benchmark_return (%)", "Total"] == pytest.approx(
        view.standalone_asset_returns.loc["Asset A", "Total (%)"]
    )


def test_performance_correlation_uses_compact_axis_labels(tmp_path):
    portfolio = _two_asset_portfolio(tmp_path)
    portfolio.registry.get("A").data_symbol = "AAA"
    portfolio.registry.get("B").data_symbol = "BBB"

    view = portfolio.performance_view(start="2025-12-29", end="2026-01-04")
    corr = view.return_correlation()

    assert list(corr.columns) == ["AAA", "BBB"]
    assert list(corr.index) == ["AAA", "BBB"]


def test_risk_view_exposes_current_and_yearly_risk_by_asset(tmp_path):
    portfolio = _two_asset_portfolio(tmp_path)

    view = portfolio.risk_view(start="2025-12-29", end="2026-01-04")

    assert set(view.summary.index) == {"annualized_volatility (%)", "sharpe_ratio", "max_drawdown (%)"}
    assert list(view.assets.columns) == [
        "name",
        "weight (%)",
        "asset_volatility (%)",
        "risk_contribution (%)",
        "asset_sharpe",
        "sharpe_contribution",
    ]
    assert view.assets["risk_contribution (%)"].sum() == pytest.approx(100.0)
    assert list(view.yearly_risk_contribution.columns) == ["2025 (%)", "2026 (%)", "Total (%)"]
    assert "Portfolio Volatility" in view.yearly_risk_contribution.index
    assert view.yearly_risk_contribution.loc[["Asset A", "Asset B"], "2026 (%)"].sum() == pytest.approx(100.0)
    assert view.yearly_portfolio_volatility["Total"] == pytest.approx(
        view.summary["annualized_volatility (%)"]
    )
    assert "Total (%)" in view.asset_volatility.columns
    assert "Total" in view.asset_sharpe.columns
    assert "Portfolio Sharpe" in view.sharpe_contribution.index
    assert "Asset A" in view.asset_volatility.index
    assert view.asset_correlation.equals(view.correlation)
    assert view.weighted_correlation.shape == view.correlation.shape
    assert view.summary["max_drawdown (%)"] <= 0.0
