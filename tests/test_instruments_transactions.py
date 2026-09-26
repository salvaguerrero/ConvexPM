from datetime import date

from convexpm import Instrument, InstrumentRegistry, Portfolio, Trade
from convexpm.transactions import load_trades, save_trades


class FakeYahooTicker:
    def get_info(self):
        return {
            "symbol": "AAPL",
            "longName": "Apple Inc.",
            "quoteType": "EQUITY",
            "currency": "USD",
            "exchange": "NMS",
            "market": "us_market",
            "country": "United States",
            "sector": "Technology",
            "industry": "Consumer Electronics",
        }


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


def test_remove_instrument_and_persist(tmp_path):
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
        )
    )

    removed = registry.remove("REP_MC")
    registry.save()
    loaded = InstrumentRegistry(path)

    assert removed.name == "Repsol"
    assert "REP_MC" not in loaded


def test_registry_update_alias(tmp_path):
    registry = InstrumentRegistry(tmp_path / "instruments.parquet", auto_load=False)
    registry.update(
        [
            Instrument(
                instrument_id="REP_MC",
                name="Repsol",
                instrument_type="equity",
                asset_class="equity",
                currency="EUR",
                data_source="yahoo",
                data_symbol="REP.MC",
            )
        ]
    )

    assert registry.get("REP_MC").name == "Repsol"


def test_registry_displays_as_dataframe(tmp_path):
    registry = InstrumentRegistry(tmp_path / "instruments.parquet", auto_load=False)
    registry.add(
        Instrument(
            instrument_id="AAPL",
            name="Apple",
            instrument_type="equity",
            asset_class="equity",
            currency="USD",
            data_source="yahoo",
            data_symbol="AAPL",
            metadata={"country": "United States", "sector": "Technology"},
        )
    )

    df = registry.to_dataframe()
    display = repr(registry)

    assert df.loc[0, "instrument_id"] == "AAPL"
    assert df.loc[0, "name"] == "Apple"
    assert df.loc[0, "sector"] == "Technology"
    assert "instrument_id" in display
    assert "Apple" in display


def test_instrument_from_yahoo_metadata():
    instrument = Instrument.from_yahoo("AAPL", ticker_factory=lambda symbol: FakeYahooTicker())

    assert instrument.instrument_id == "AAPL"
    assert instrument.name == "Apple Inc."
    assert instrument.instrument_type == "equity"
    assert instrument.asset_class == "equity"
    assert instrument.currency == "USD"
    assert instrument.data_source == "yahoo"
    assert instrument.data_symbol == "AAPL"
    assert instrument.metadata["sector"] == "Technology"


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


def test_portfolio_add_trades_is_idempotent():
    trade = Trade(
        trade_id="stable-import-id",
        date=date(2026, 1, 2),
        instrument_id="REP_MC",
        side="BUY",
        quantity=10,
    )
    portfolio = Portfolio(name="Main", base_currency="EUR")

    assert portfolio.add_trades([trade]) == 1
    assert portfolio.add_trades([trade]) == 0
    assert len(portfolio.trades) == 1


def test_portfolio_add_trades_skips_same_trade_with_different_id():
    existing = Trade(
        trade_id="old-id",
        date=date(2026, 1, 2),
        instrument_id="REP",
        side="BUY",
        quantity=10,
        amount=100,
        value_per_unit=10,
        fees=1,
        currency="EUR",
        account="ibkr",
        tags=["ibkr"],
        metadata={"source": "ibkr", "source_rows": [10]},
    )
    same_trade = Trade(
        trade_id="new-id",
        date=date(2026, 1, 2),
        instrument_id="REP",
        side="BUY",
        quantity=10,
        amount=100,
        value_per_unit=10,
        fees=1,
        currency="EUR",
        account="ibkr",
        tags=["ibkr"],
        metadata={"source": "ibkr", "source_rows": [99]},
    )
    portfolio = Portfolio(name="Main", base_currency="EUR", trades=[existing])

    assert portfolio.add_trades([same_trade]) == 0
    assert len(portfolio.trades) == 1


def test_portfolio_deduplicate_trades_removes_existing_duplicates():
    first = Trade(
        trade_id="old-id",
        date=date(2026, 1, 2),
        instrument_id="REP",
        side="BUY",
        quantity=10,
        amount=100,
        value_per_unit=10,
        fees=1,
        currency="EUR",
        tags=["ibkr"],
    )
    duplicate = Trade(
        trade_id="new-id",
        date=date(2026, 1, 2),
        instrument_id="REP",
        side="BUY",
        quantity=10,
        amount=100,
        value_per_unit=10,
        fees=1,
        currency="EUR",
        tags=["ibkr"],
    )
    portfolio = Portfolio(name="Main", base_currency="EUR", trades=[first, duplicate])

    assert portfolio.deduplicate_trades() == 1
    assert [trade.trade_id for trade in portfolio.trades] == ["old-id"]


def test_portfolio_trades_ledger_detail_and_delete():
    registry = InstrumentRegistry(auto_load=False)
    registry.add(
        Instrument(
            instrument_id="AAPL",
            name="Apple",
            instrument_type="equity",
            asset_class="equity",
            currency="USD",
            data_source="fixture",
        )
    )
    portfolio = Portfolio(name="Main", base_currency="EUR", registry=registry)
    trades = [
        Trade(
            trade_id="t1",
            date=date(2026, 1, 1),
            instrument_id="AAPL",
            side="BUY",
            quantity=2,
            amount=300,
            value_per_unit=150,
            fees=1,
        ),
        Trade(
            trade_id="t2",
            date=date(2026, 1, 2),
            instrument_id="REP",
            side="BUY",
            quantity=2,
        ),
    ]
    portfolio.add_trades(trades)

    assert portfolio.trades.to_dataframe()["trade_id"].tolist() == ["t1", "t2"]
    assert portfolio.trades.to_dataframe()["instrument_name"].tolist() == ["Apple", None]
    assert portfolio.trades.to_dataframe()["price"].tolist()[0] == 150
    assert portfolio.trades.to_dataframe()["price_without_fee"].tolist()[0] == 150
    assert portfolio.trades[1]["trade_id"] == "t1"
    assert portfolio.trades[1]["instrument_name"] == "Apple"
    assert portfolio.trades[1]["price"] == 150
    assert portfolio.trades[1]["price_without_fee"] == 150
    assert portfolio.trades["instrument_id"].tolist() == ["AAPL", "REP"]
    assert portfolio.trades["instrument_name"].tolist() == ["Apple", None]
    assert portfolio.trades["price"].tolist()[0] == 150
    assert portfolio.trades["price_without_fee"].tolist()[0] == 150
    assert portfolio.trades.to_dataframe().index.tolist() == ["t1", "t2"]
    display = repr(portfolio.trades)
    assert "trade_id" not in display
    assert "instrument_id" not in display
    assert " price" in display

    deleted = portfolio.delete_trade(1)

    assert deleted.trade_id == "t1"
    assert [trade.trade_id for trade in portfolio.trades] == ["t2"]

    portfolio.add_trade(trades[0])
    assert portfolio.trades.delete("t2").instrument_id == "REP"
    assert [trade.trade_id for trade in portfolio.trades] == ["t1"]


def test_trade_filter_index_can_be_used_for_deletion():
    portfolio = Portfolio(name="Main", base_currency="EUR")
    portfolio.add_trades(
        [
            Trade(trade_id="keep", date=date(2026, 1, 1), instrument_id="AAPL", side="BUY", quantity=1),
            Trade(trade_id="drop-1", date=date(2026, 1, 2), instrument_id="CASH_FUND", side="BUY", quantity=1),
            Trade(trade_id="drop-2", date=date(2026, 1, 3), instrument_id="CASH_FUND", side="BUY", quantity=1),
        ]
    )

    mask = portfolio.trades["instrument_id"] == "CASH_FUND"
    for trade_no in portfolio.trades[mask].index.tolist():
        portfolio.trades.delete(trade_no)

    assert [trade.trade_id for trade in portfolio.trades] == ["keep"]


def test_trade_display_uses_renta4_fund_name_metadata_when_registry_is_missing():
    portfolio = Portfolio(name="Main", base_currency="EUR")
    portfolio.add_trade(
        Trade(
            trade_id="r4",
            date=date(2026, 1, 2),
            instrument_id="R4_RENTA_4_FONCUENTA_AHORRO_FI",
            side="BUY",
            quantity=1,
            metadata={
                "source": "renta4",
                "fund_name": "RENTA 4 FONCUENTA AHORRO, FI",
            },
        )
    )

    assert portfolio.trades.to_dataframe().loc["r4", "instrument_name"] == "RENTA 4 FONCUENTA AHORRO, FI"
    assert portfolio.trades[1]["instrument_name"] == "RENTA 4 FONCUENTA AHORRO, FI"
