from datetime import date

import numpy as np
import pandas as pd
import pytest

from convexpm import Instrument, InstrumentRegistry, MarketDataStore, Portfolio, Trade
from convexpm.analytics.asset_returns import instrument_returns


def _portfolio(tmp_path, prices, trades):
    registry = InstrumentRegistry(tmp_path / "instruments.parquet", auto_load=False)
    store = MarketDataStore(tmp_path / "market.parquet")
    rows = []
    for instrument_id, values in prices.items():
        registry.add(
            Instrument(
                instrument_id=instrument_id,
                name=instrument_id,
                instrument_type="equity",
                asset_class="equity",
                currency="EUR",
                data_source="fixture",
            )
        )
        for dt, value in zip(pd.date_range("2026-01-01", periods=len(values), freq="D"), values):
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
    store.upsert(pd.DataFrame(rows))
    return Portfolio(
        name="Analysis",
        base_currency="EUR",
        trades=trades,
        registry=registry,
        market_data=store,
    )


def test_recent_purchase_uses_full_history_for_current_risk_but_not_return_attribution(tmp_path):
    portfolio = _portfolio(
        tmp_path,
        {
            "A": [100, 101, 102, 103, 104, 105],
            "B": [50, 55, 50, 55, 50, 55],
        },
        [
            Trade(date=date(2026, 1, 1), instrument_id="A", side="BUY", quantity=1),
            Trade(date=date(2026, 1, 6), instrument_id="B", side="BUY", quantity=1),
        ],
    )

    risk = portfolio.current_risk(start="2026-01-01", end="2026-01-06")
    analysis = portfolio.analyze(start="2026-01-01", end="2026-01-06")

    assert risk.metadata["observations"] == 5
    assert "B" in risk.correlation.columns
    assert analysis.return_attribution.loc["B", "return_contribution"] == pytest.approx(0.0)


def test_100_percent_single_asset_volatility_matches_asset_history(tmp_path):
    portfolio = _portfolio(
        tmp_path,
        {"A": [100, 110, 99, 108, 102]},
        [Trade(date=date(2026, 1, 1), instrument_id="A", side="BUY", quantity=1)],
    )

    returns = instrument_returns(
        portfolio.market_data,
        ["A"],
        start="2026-01-01",
        end="2026-01-05",
    )["A"]
    expected = returns.std(ddof=1) * np.sqrt(252)

    risk = portfolio.current_risk(start="2026-01-01", end="2026-01-05")
    assert risk.metrics["annualized_volatility"] == pytest.approx(expected)


def test_perfectly_correlated_assets_have_no_diversification_benefit(tmp_path):
    portfolio = _portfolio(
        tmp_path,
        {
            "A": [100, 110, 99, 108, 102],
            "B": [200, 220, 198, 216, 204],
        },
        [
            Trade(date=date(2026, 1, 1), instrument_id="A", side="BUY", quantity=1),
            Trade(date=date(2026, 1, 1), instrument_id="B", side="BUY", quantity=0.5),
        ],
    )

    risk = portfolio.current_risk(start="2026-01-01", end="2026-01-05")
    asset_vol = instrument_returns(
        portfolio.market_data,
        ["A"],
        start="2026-01-01",
        end="2026-01-05",
    )["A"].std(ddof=1) * np.sqrt(252)

    assert risk.correlation.loc["A", "B"] == pytest.approx(1.0)
    assert risk.metrics["annualized_volatility"] == pytest.approx(asset_vol)


def test_negative_correlation_reduces_portfolio_volatility(tmp_path):
    portfolio = _portfolio(
        tmp_path,
        {
            "A": [100, 110, 99, 108.9, 98.01],
            "B": [100, 90, 99, 89.1, 98.01],
        },
        [
            Trade(date=date(2026, 1, 1), instrument_id="A", side="BUY", quantity=1),
            Trade(date=date(2026, 1, 1), instrument_id="B", side="BUY", quantity=1),
        ],
    )

    risk = portfolio.current_risk(start="2026-01-01", end="2026-01-05")
    returns = instrument_returns(
        portfolio.market_data,
        ["A", "B"],
        start="2026-01-01",
        end="2026-01-05",
    )
    individual_vol = returns.std(ddof=1) * np.sqrt(252)

    assert risk.correlation.loc["A", "B"] < 0
    assert risk.metrics["annualized_volatility"] < individual_vol.min()


def test_weights_correlation_and_risk_contributions_reconcile(tmp_path):
    portfolio = _portfolio(
        tmp_path,
        {
            "A": [100, 102, 101, 104, 103, 106],
            "B": [80, 79, 82, 81, 84, 83],
        },
        [
            Trade(date=date(2026, 1, 1), instrument_id="A", side="BUY", quantity=1),
            Trade(date=date(2026, 1, 1), instrument_id="B", side="BUY", quantity=1),
        ],
    )

    risk = portfolio.current_risk(start="2026-01-01", end="2026-01-06")

    assert risk.weights.sum() == pytest.approx(1.0)
    assert np.allclose(risk.correlation, risk.correlation.T)
    assert np.allclose(np.diag(risk.correlation), 1.0)
    assert risk.risk_contribution["component_contribution"].sum() == pytest.approx(
        risk.metrics["annualized_volatility"]
    )
    assert risk.risk_contribution["risk_contribution"].sum() == pytest.approx(1.0)


def test_single_asset_return_attribution_reconciles_with_realized_return(tmp_path):
    portfolio = _portfolio(
        tmp_path,
        {"A": [100, 105, 102, 110]},
        [Trade(date=date(2026, 1, 1), instrument_id="A", side="BUY", quantity=1)],
    )

    analysis = portfolio.analyze(start="2026-01-01", end="2026-01-04")
    expected = 110 / 100 - 1

    assert analysis.return_attribution.loc["A", "return_contribution"] == pytest.approx(expected)
    assert analysis.instruments.loc["A", "current_weight"] == pytest.approx(1.0)
    assert analysis.metadata["missing_data_policy"] == "intersection"
