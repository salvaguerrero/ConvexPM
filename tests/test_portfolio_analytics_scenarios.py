from datetime import date

import numpy as np
import pandas as pd
import pytest

from convexpm import Instrument, InstrumentRegistry, MarketDataStore, Portfolio, Trade
from convexpm.analytics.returns import cagr, total_return
from convexpm.analytics.risk import annualized_volatility, historical_cvar, historical_var, max_drawdown
from convexpm.portfolio import build_holdings


def test_holdings_reconstruction_buy_and_sell():
    trades = [
        Trade(trade_id="t1", date=date(2026, 1, 1), instrument_id="REP_MC", side="BUY", quantity=10),
        Trade(trade_id="t2", date=date(2026, 1, 3), instrument_id="REP_MC", side="BUY", quantity=5),
        Trade(trade_id="t3", date=date(2026, 1, 6), instrument_id="REP_MC", side="SELL", quantity=3),
    ]

    assert build_holdings(trades, date(2026, 1, 7)) == {"REP_MC": 12.0}


def test_nav_history(tmp_path):
    portfolio = _sample_portfolio(tmp_path)

    history = portfolio.nav_history()

    assert history["nav"].tolist() == [100.0, 110.0, 120.0, 120.0]
    assert portfolio.nav(date(2026, 1, 4)) == pytest.approx(120.0)


def test_total_return_and_cagr():
    nav = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2026-01-01"]),
            "nav": [100.0, 110.0],
        }
    )

    assert total_return(nav) == pytest.approx(0.10)
    assert cagr(nav) == pytest.approx((1.10 ** (365.25 / 365)) - 1.0)


def test_max_drawdown_and_volatility_and_var_cvar():
    nav = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=5, freq="D"),
            "nav": [100.0, 110.0, 99.0, 101.0, 90.9],
        }
    )
    returns = pd.Series([0.10, -0.10, 0.0202020202, -0.10])

    assert max_drawdown(nav) == pytest.approx(90.9 / 110.0 - 1.0)
    assert annualized_volatility(nav) == pytest.approx(returns.std(ddof=1) * np.sqrt(252))
    assert historical_var(nav, confidence=0.75) == pytest.approx(np.quantile(returns, 0.25))
    assert historical_cvar(nav, confidence=0.75) == pytest.approx(-0.10)


def test_exposure_by_asset_class_and_currency(tmp_path):
    portfolio = _sample_portfolio(tmp_path)

    by_asset = portfolio.exposure(by="asset_class")
    by_currency = portfolio.exposure(by="currency")

    assert by_asset.loc[0, "asset_class"] == "equity"
    assert by_asset.loc[0, "market_value"] == pytest.approx(120.0)
    assert by_asset.loc[0, "weight"] == pytest.approx(1.0)
    assert by_currency.loc[0, "currency"] == "EUR"


def test_scenario_cloning_original_unchanged(tmp_path):
    portfolio = _sample_portfolio(tmp_path)
    scenario = portfolio.what_if(
        Trade(
            trade_id="hypo",
            date=date(2026, 1, 2),
            instrument_id="REP_MC",
            side="BUY",
            quantity=1,
            amount=11,
            value_per_unit=11,
        )
    )

    scenario_portfolio = scenario.portfolio

    assert portfolio.holdings(date(2026, 1, 4)) == {"REP_MC": 10.0}
    assert scenario_portfolio.holdings(date(2026, 1, 4)) == {"REP_MC": 11.0}
    assert len(portfolio.trades) == 1
    assert len(scenario_portfolio.trades) == 2


def test_scenario_compare(tmp_path):
    portfolio = _sample_portfolio(tmp_path)
    scenario = portfolio.what_if(
        Trade(
            trade_id="hypo",
            date=date(2026, 1, 2),
            instrument_id="REP_MC",
            side="BUY",
            quantity=1,
            amount=11,
            value_per_unit=11,
        )
    )

    comparison = scenario.compare()

    assert "Metric" in comparison.columns
    assert "Current" in comparison.columns
    assert "Scenario" in comparison.columns
    assert comparison.loc[comparison["Metric"] == "Current NAV", "Scenario"].iloc[0] == pytest.approx(132.0)


def test_current_scenario_after_latest_market_date_uses_latest_value(tmp_path):
    portfolio = _sample_portfolio(tmp_path)
    scenario = portfolio.what_if(
        Trade(
            trade_id="future_hypo",
            date=date(2026, 1, 5),
            instrument_id="REP_MC",
            side="BUY",
            quantity=1,
        )
    )

    assert scenario.portfolio.holdings() == {"REP_MC": 11.0}
    assert scenario.portfolio.nav() == pytest.approx(132.0)



def test_portfolio_save_and_load(tmp_path):
    portfolio = _sample_portfolio(tmp_path)
    portfolios_root = tmp_path / "portfolios"

    saved_dir = portfolio.save("personal", root=portfolios_root)

    assert saved_dir == portfolios_root / "personal"
    assert (saved_dir / "portfolio.json").exists()
    assert (saved_dir / "trades.parquet").exists()

    loaded = Portfolio.load(
        "personal",
        root=portfolios_root,
        registry=portfolio.registry,
        market_data=portfolio.market_data,
    )

    assert loaded.name == portfolio.name
    assert loaded.base_currency == portfolio.base_currency
    assert [trade.to_dict() for trade in loaded.trades] == [
        trade.to_dict() for trade in portfolio.trades
    ]
    assert loaded.holdings(date(2026, 1, 4)) == {"REP_MC": 10.0}
    assert loaded.nav(date(2026, 1, 4)) == pytest.approx(120.0)


def test_loaded_portfolio_can_add_trade_and_save_without_id(tmp_path):
    portfolio = _sample_portfolio(tmp_path)
    portfolios_root = tmp_path / "portfolios"
    portfolio.save("personal", root=portfolios_root)

    loaded = Portfolio.load(
        "personal",
        root=portfolios_root,
        registry=portfolio.registry,
        market_data=portfolio.market_data,
    )
    loaded.add_trade(
        Trade(
            trade_id="t2",
            date=date(2026, 1, 2),
            instrument_id="REP_MC",
            side="BUY",
            quantity=2,
            amount=22,
            value_per_unit=11,
            currency="EUR",
        )
    )
    loaded.save()

    reloaded = Portfolio.load(
        "personal",
        root=portfolios_root,
        registry=portfolio.registry,
        market_data=portfolio.market_data,
    )

    assert [trade.trade_id for trade in reloaded.trades] == ["t1", "t2"]
    assert reloaded.holdings(date(2026, 1, 4)) == {"REP_MC": 12.0}


def test_add_trade_rejects_duplicate_trade_id(tmp_path):
    portfolio = _sample_portfolio(tmp_path)

    with pytest.raises(ValueError, match="Duplicate trade_id"):
        portfolio.add_trade(
            Trade(
                trade_id="t1",
                date=date(2026, 1, 2),
                instrument_id="REP_MC",
                side="BUY",
                quantity=1,
            )
        )


def _sample_portfolio(tmp_path):
    registry = InstrumentRegistry(tmp_path / "instruments.parquet", auto_load=False)
    registry.add(
        Instrument(
            instrument_id="REP_MC",
            name="Repsol",
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
                "date": pd.date_range("2026-01-01", periods=4, freq="D"),
                "instrument_id": "REP_MC",
                "value": [10.0, 11.0, 12.0, 12.0],
                "currency": "EUR",
                "value_type": "close",
                "source": "fixture",
            }
        )
    )
    trades = [
        Trade(
            trade_id="t1",
            date=date(2026, 1, 1),
            instrument_id="REP_MC",
            side="BUY",
            quantity=10,
            amount=100,
            value_per_unit=10,
            currency="EUR",
        )
    ]
    return Portfolio(
        name="Test",
        base_currency="EUR",
        trades=trades,
        registry=registry,
        market_data=store,
    )
