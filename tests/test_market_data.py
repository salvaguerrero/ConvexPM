from datetime import date

import pandas as pd
import pytest

from convexpm import Instrument, InstrumentRegistry, MarketDataStore
from convexpm.market_data import MarketDataUpdater, Renta4CSVProvider, YahooProvider


class FakeTicker:
    def history(self, start=None, end=None, auto_adjust=False):
        return pd.DataFrame(
            {
                "Close": [10.0, 10.5],
                "Adj Close": [9.8, 10.3],
            },
            index=pd.to_datetime(["2026-01-01", "2026-01-02"]),
        )


class RecordingProvider:
    def __init__(self):
        self.calls = []

    def fetch_history(self, instrument, start=None, end=None):
        self.calls.append({"instrument_id": instrument.instrument_id, "start": start, "end": end})
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-01", "2026-01-03"]),
                "instrument_id": [instrument.instrument_id, instrument.instrument_id],
                "value": [9.0, 13.0],
                "currency": [instrument.currency, instrument.currency],
                "value_type": ["close", "close"],
                "source": ["recording", "recording"],
            }
        )


def test_yahoo_normalized_schema():
    instrument = Instrument(
        instrument_id="REP_MC",
        name="Repsol",
        instrument_type="equity",
        asset_class="equity",
        currency="EUR",
        data_source="yahoo",
        data_symbol="REP.MC",
    )
    provider = YahooProvider(ticker_factory=lambda symbol: FakeTicker())

    df = provider.fetch_history(instrument, start=date(2026, 1, 1), end=date(2026, 1, 3))

    assert list(df.columns) == ["date", "instrument_id", "value", "currency", "value_type", "source"]
    assert df["instrument_id"].tolist() == ["REP_MC", "REP_MC"]
    assert df["value"].tolist() == [10.0, 10.5]
    assert set(df["value_type"]) == {"close"}
    assert set(df["source"]) == {"yahoo"}


def test_renta4_parser_with_one_fund():
    text = """
;ES0173311103 - R4 Multigestion Numantia Patr.global

;12/06/2015;10;
;13/06/2015;10,25;
"""
    market_df, instruments = Renta4CSVProvider().parse_text(text)

    assert len(instruments) == 1
    assert instruments[0].instrument_id == "ES0173311103"
    assert instruments[0].name == "R4 Multigestion Numantia Patr.global"
    assert market_df["value"].tolist() == [10.0, 10.25]
    assert set(market_df["value_type"]) == {"nav"}


def test_renta4_parser_with_multiple_funds():
    text = """
;ES0173311103 - Fund One
;12/06/2015;10;
;ES0123456789 - Fund Two
;12/06/2015;20;
;13/06/2015;21;
"""
    market_df, instruments = Renta4CSVProvider().parse_text(text)

    assert [instrument.instrument_id for instrument in instruments] == ["ES0123456789", "ES0173311103"]
    assert len(market_df) == 3
    assert market_df.groupby("instrument_id")["value"].sum().to_dict() == {
        "ES0123456789": 41.0,
        "ES0173311103": 10.0,
    }


def test_market_data_deduplication(tmp_path):
    store = MarketDataStore(tmp_path / "market.parquet")
    store.upsert(
        pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-01", "2026-01-02"],
                "instrument_id": ["REP_MC", "REP_MC", "REP_MC"],
                "value": [10.0, 11.0, 12.0],
                "currency": ["EUR", "EUR", "EUR"],
                "value_type": ["close", "close", "close"],
                "source": ["fixture", "fixture", "fixture"],
            }
        )
    )

    history = store.get_history("REP_MC")

    assert len(history) == 2
    assert store.get_value("REP_MC", date(2026, 1, 1)) == pytest.approx(11.0)
    assert store.latest("REP_MC")["value"] == pytest.approx(12.0)


def test_market_data_delete_instrument(tmp_path):
    store = MarketDataStore(tmp_path / "market.parquet")
    store.upsert(
        pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-02", "2026-01-01"],
                "instrument_id": ["REP_MC", "REP_MC", "AAPL"],
                "value": [10.0, 12.0, 200.0],
                "currency": ["EUR", "EUR", "USD"],
                "value_type": ["close", "close", "close"],
                "source": ["fixture", "fixture", "fixture"],
            }
        )
    )

    removed = store.delete_instrument("REP_MC")

    assert removed == 2
    assert store.get_history("REP_MC").empty
    assert store.get_value("AAPL", date(2026, 1, 1)) == pytest.approx(200.0)


def test_market_data_store_interactive_summary_and_lookup(tmp_path):
    registry_path = tmp_path / "instruments.parquet"
    registry = InstrumentRegistry(registry_path, auto_load=False)
    registry.add(
        Instrument(
            instrument_id="REP",
            name="Repsol",
            instrument_type="equity",
            asset_class="equity",
            currency="EUR",
            data_source="yahoo",
            data_symbol="REP.MC",
            metadata={"country": "ES"},
        )
    )
    store = MarketDataStore(tmp_path / "market.parquet", registry=registry)
    store.upsert(
        pd.DataFrame(
            {
                "date": ["2025-01-01", "2025-01-02", "2026-01-01", "2026-01-02", "2026-01-03"],
                "instrument_id": ["REP", "REP", "REP", "REP", "REP"],
                "value": [8.0, 9.0, 10.0, 11.0, 12.0],
                "currency": ["EUR", "EUR", "EUR", "EUR", "EUR"],
                "value_type": ["close", "close", "close", "close", "close"],
                "source": ["fixture", "fixture", "fixture", "fixture", "fixture"],
            }
        )
    )

    summary = store.summary()
    repsol = store["REPSOL"]

    assert summary.loc[0, "name"] == "Repsol"
    assert summary.loc[0, "last_price"] == pytest.approx(12.0)
    assert store["REP.MC"].instrument_id == "REP"
    assert repsol.info["ticker"] == "REP.MC"
    assert repsol.metadata == {"country": "ES"}
    assert repsol.stats["total_return"] == pytest.approx(0.5)
    assert pd.notna(repsol.stats["sharpe"])
    assert list(repsol.stats().columns) == ["2025", "2026"]
    assert list(repsol.stats(1, freq="Y").columns) == ["2026"]
    assert "2026-01" in repsol.stats(freq="M").columns
    assert repsol.prices.iloc[-1] == pytest.approx(12.0)
    import matplotlib

    matplotlib.use("Agg")
    assert repsol.plot(show=False).get_title() == "REP price / NAV"
    assert "REP price / NAV" in repr(repsol.plot_text(width=20, height=5))
    assert "REP price / NAV" in repr(repsol.plot(style="terminal", width=20, height=5))


def test_market_data_updater_can_force_backfill(tmp_path):
    registry = InstrumentRegistry(tmp_path / "instruments.parquet", auto_load=False)
    registry.add(
        Instrument(
            instrument_id="REP",
            name="Repsol",
            instrument_type="equity",
            asset_class="equity",
            currency="EUR",
            data_source="recording",
            data_symbol="REP.MC",
        )
    )
    store = MarketDataStore(tmp_path / "market.parquet")
    store.upsert(
        pd.DataFrame(
            {
                "date": ["2026-01-02"],
                "instrument_id": ["REP"],
                "value": [12.0],
                "currency": ["EUR"],
                "value_type": ["close"],
                "source": ["fixture"],
            }
        )
    )
    provider = RecordingProvider()
    updater = MarketDataUpdater(registry, store, {"recording": provider})

    updater.update_instrument("REP", start=date(2020, 1, 1))
    updater.update_instrument("REP", start=date(2020, 1, 1), force=True)

    assert provider.calls[0]["start"] == date(2026, 1, 3)
    assert provider.calls[1]["start"] == date(2020, 1, 1)
    assert store.get_value("REP", date(2025, 1, 1), exact=True) == pytest.approx(9.0)
