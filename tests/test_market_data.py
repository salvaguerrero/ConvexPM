from datetime import date

import pandas as pd
import pytest

from convexpm import Instrument, MarketDataStore
from convexpm.market_data import Renta4CSVProvider, YahooProvider


class FakeTicker:
    def history(self, start=None, end=None, auto_adjust=False):
        return pd.DataFrame(
            {
                "Close": [10.0, 10.5],
                "Adj Close": [9.8, 10.3],
            },
            index=pd.to_datetime(["2026-01-01", "2026-01-02"]),
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
