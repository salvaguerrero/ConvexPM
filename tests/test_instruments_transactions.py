from datetime import date

from convexpm import Instrument, InstrumentRegistry, Trade
from convexpm.transactions import load_trades, save_trades


def test_adding_loading_instruments(tmp_path):
    path = tmp_path / "instruments.parquet"
    registry = InstrumentRegistry(path, auto_load=False)
    registry.add(
        Instrument(
            instrument_id="REP_MC",
            name="Repsol",
            instrument_type="equity",
            asset_class="equity",
            currency="EUR",
            data_source="yahoo",
            data_symbol="REP.MC",
            metadata={"country": "ES"},
        )
    )
    registry.save()

    loaded = InstrumentRegistry(path)

    assert len(loaded) == 1
    assert loaded.get("REP_MC").data_symbol == "REP.MC"
    assert loaded.get("REP_MC").metadata["country"] == "ES"


def test_trade_serialization(tmp_path):
    path = tmp_path / "trades.parquet"
    trades = [
        Trade(
            trade_id="t1",
            date=date(2026, 1, 2),
            instrument_id="REP_MC",
            side="BUY",
            quantity=10,
            amount=150,
            value_per_unit=15,
            fees=1,
            currency="EUR",
            tags=["core"],
            metadata={"note": "fixture"},
        )
    ]

    save_trades(trades, path)
    loaded = load_trades(path)

    assert len(loaded) == 1
    assert loaded[0].trade_id == "t1"
    assert loaded[0].date == date(2026, 1, 2)
    assert loaded[0].tags == ["core"]
    assert loaded[0].metadata == {"note": "fixture"}
