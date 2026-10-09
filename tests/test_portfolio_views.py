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
    assert view.cumulative_return.iloc[-1] == pytest.approx(view.summary["portfolio_return (%)"])
    assert view.summary["cagr (%)"] != view.summary["portfolio_return (%)"]
    assert view.summary["portfolio_return_ex_cash_flows (%)"] == pytest.approx(view.summary["portfolio_return (%)"])
    assert view.summary["capital_inflows"] == pytest.approx(200.0)
    assert view.cumulative_return.name == "cumulative_return (%)"
    assert "Total (%)" in view.standalone_asset_returns.columns
    assert "Total (%)" in view.asset_weights.columns
    assert "Portfolio Weight" in view.asset_weights.index
    assert "Asset A" in view.standalone_asset_returns.index
    assert not hasattr(view, "asset_returns")
    assert not hasattr(view, "return_contribution")
    for year in view.yearly_return_contribution.columns:
        asset_total = view.yearly_return_contribution.loc[["Asset A", "Asset B"], year].sum()
        assert asset_total == pytest.approx(
            view.yearly_return_contribution.loc["Portfolio Return", year]
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

    assert view.summary["capital_inflows"] == pytest.approx(1_000.0)
    assert view.summary["capital_outflows"] == pytest.approx(0.0)


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

    assert view.summary["portfolio_return (%)"] == pytest.approx(-45.0)
    assert view.yearly_return_contribution.loc["USD Asset", "2026 (%)"] == pytest.approx(-45.0)
    assert view.standalone_asset_returns.loc["USD Asset", "2026 (%)"] == pytest.approx(-45.0)


def test_risk_view_exposes_current_and_yearly_risk_by_asset(tmp_path):
    portfolio = _two_asset_portfolio(tmp_path)

    view = portfolio.risk_view(start="2025-12-29", end="2026-01-04")

    assert set(view.summary.index) == {"annualized_volatility", "sharpe_ratio", "max_drawdown"}
    assert list(view.assets.columns) == [
        "name",
        "weight",
        "asset_volatility",
        "risk_contribution",
        "asset_sharpe",
        "sharpe_contribution",
    ]
    assert view.assets["risk_contribution"].sum() == pytest.approx(100.0)
    assert list(view.yearly_risk_contribution.columns) == ["2025", "2026", "Total"]
    assert "Portfolio Volatility" in view.yearly_risk_contribution.index
    assert view.yearly_risk_contribution.loc[["Asset A", "Asset B"], "2026"].sum() == pytest.approx(100.0)
    assert view.yearly_portfolio_volatility["Total"] == pytest.approx(
        view.summary["annualized_volatility"]
    )
    assert "Total" in view.asset_volatility.columns
    assert "Total" in view.asset_sharpe.columns
    assert "Portfolio Sharpe" in view.sharpe_contribution.index
    assert "Asset A" in view.asset_volatility.index
    assert view.asset_correlation.equals(view.correlation)
    assert view.weighted_correlation.shape == view.correlation.shape
    assert view.summary["max_drawdown"] <= 0.0
