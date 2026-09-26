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

    assert list(view.yearly_return_contribution.columns) == ["2025", "2026"]
    assert list(view.assets.columns) == [
        "name",
        "current_weight",
        "asset_return",
        "return_contribution",
    ]
    assert view.cumulative_return.iloc[-1] == pytest.approx(view.summary["portfolio_return"])
    assert view.summary["cagr"] != view.summary["portfolio_return"]
    for year in view.yearly_return_contribution.columns:
        asset_total = view.yearly_return_contribution.loc[["A", "B"], year].sum()
        assert asset_total == pytest.approx(
            view.yearly_return_contribution.loc["Portfolio Return", year]
        )


def test_risk_view_exposes_current_and_yearly_risk_by_asset(tmp_path):
    portfolio = _two_asset_portfolio(tmp_path)

    view = portfolio.risk_view(start="2025-12-29", end="2026-01-04")

    assert set(view.summary.index) == {"annualized_volatility", "max_drawdown"}
    assert list(view.assets.columns) == [
        "name",
        "weight",
        "asset_volatility",
        "risk_contribution",
    ]
    assert view.assets["risk_contribution"].sum() == pytest.approx(1.0)
    assert list(view.yearly_risk_contribution.columns) == ["2025", "2026"]
    assert "Portfolio Volatility" in view.yearly_risk_contribution.index
    assert view.yearly_risk_contribution.loc[["A", "B"], "2026"].sum() == pytest.approx(1.0)
    assert view.yearly_portfolio_volatility["2026"] == pytest.approx(
        view.summary["annualized_volatility"]
    )
    assert view.summary["max_drawdown"] <= 0.0
