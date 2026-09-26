from datetime import date

import pandas as pd
import pytest

from convexpm.importers import IBKRTransactionParser, Renta4ImportParser
from convexpm.importers.renta4 import _fallback_instruments_from_transactions


def test_ibkr_parser_groups_trades_and_allocates_transaction_fees():
    text = """Statement,Header,Field Name,Field Value
Summary,Header,Field Name,Field Value
Summary,Data,Base Currency,EUR
Transaction History,Header,Date,Account,Description,Transaction Type,Symbol,Quantity,Price,Price Currency,Gross Amount ,Commission,Net Amount
Transaction History,Data,2026-04-16,U***32596,REPSOL SA,Buy,REP,18.0,20.65,EUR,-371.7,-,-371.7
Transaction History,Data,2026-04-16,U***32596,Spanish Daily Trade Charge Tax REP 20,Transaction Fee,REP,20.0,-,-,-0.83,-,-0.83
Transaction History,Data,2026-04-16,U***32596,REPSOL SA,Buy,REP,2.0,20.65,EUR,-41.3,-3.0,-44.3
Transaction History,Data,2026-07-08,U***32596,REP Cash Dividend,Dividend,REP,-,-,-,16.53,-,16.53
"""

    result = IBKRTransactionParser().parse_text(text)

    assert result.summary["Base Currency"] == "EUR"
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.instrument_id == "REP"
    assert trade.side == "BUY"
    assert trade.date == date(2026, 4, 16)
    assert trade.quantity == pytest.approx(20)
    assert trade.amount == pytest.approx(413)
    assert trade.value_per_unit == pytest.approx(20.65)
    assert trade.fees == pytest.approx(3.83)
    assert trade.metadata["commission"] == pytest.approx(-3)
    assert trade.metadata["transaction_costs"] == pytest.approx(0.83)
    assert result.cash_transactions["transaction_type"].tolist() == ["Transaction Fee", "Dividend"]


def test_ibkr_trade_id_does_not_depend_on_statement_row_number():
    first = """Statement,Header,Field Name,Field Value
Summary,Header,Field Name,Field Value
Summary,Data,Base Currency,EUR
Transaction History,Header,Date,Account,Description,Transaction Type,Symbol,Quantity,Price,Price Currency,Gross Amount ,Commission,Net Amount
Transaction History,Data,2026-04-16,U***32596,REPSOL SA,Buy,REP,2.0,20.65,EUR,-41.3,-3.0,-44.3
"""
    second = """Statement,Header,Field Name,Field Value
Summary,Header,Field Name,Field Value
Summary,Data,Base Currency,EUR
Transaction History,Header,Date,Account,Description,Transaction Type,Symbol,Quantity,Price,Price Currency,Gross Amount ,Commission,Net Amount
Transaction History,Data,2026-01-01,U***32596,Dividend,Dividend,REP,-,-,-,1.0,-,1.0
Transaction History,Data,2026-04-16,U***32596,REPSOL SA,Buy,REP,2.0,20.65,EUR,-41.3,-3.0,-44.3
"""

    trade_id = IBKRTransactionParser().parse_text(first).trades[0].trade_id
    same_trade_id = IBKRTransactionParser().parse_text(second).trades[0].trade_id

    assert same_trade_id == trade_id


def test_renta4_transaction_parser_maps_fund_names_to_isin_and_costs():
    df = pd.DataFrame(
        [
            ["client", None, None, None, None, None, None, "Código Bolsa", "OW1877"],
            [None, None, None, None, None, None, None, None, None],
            [
                "Fecha",
                "Tipo operación",
                "Participaciones",
                "Importe bruto Div.",
                "Importe bruto",
                "Comisión.",
                "Retención",
                "Importe NETO",
                "Estado",
            ],
            ["R4 MULTIGESTION NUMANTIA PATR. GLOBAL", None, None, None, None, None, None, None, None],
            ["26/03/2026", "SUSCRIPCIÓN", 37.397614, 1000, 1000, 1.5, 0, 998.5, "Validada"],
            ["RENTA 4 FONCUENTA AHORRO, FI", None, None, None, None, None, None, None, None],
            ["10/07/2026", "REEMBOLSO", 18.436779, 200, 200, 0, 0.91, 199.09, "Validada"],
        ]
    )

    trades, raw = Renta4ImportParser().parse_transactions_dataframe(
        df,
        instrument_name_map={
            "r4 multigestion numantia patr global": "ES0173311103",
        },
    )

    assert raw["instrument_id"].tolist() == ["ES0173311103", "R4_RENTA_4_FONCUENTA_AHORRO_FI"]
    assert len(trades) == 2
    assert trades[0].instrument_id == "ES0173311103"
    assert trades[0].side == "BUY"
    assert trades[0].fees == pytest.approx(1.5)
    assert trades[1].side == "SELL"
    assert trades[1].fees == pytest.approx(0.91)


def test_renta4_fallback_instruments_keep_transaction_fund_names():
    raw = pd.DataFrame(
        [
            {
                "instrument_id": "R4_RENTA_4_FONCUENTA_AHORRO_FI",
                "fund_name": "RENTA 4 FONCUENTA AHORRO, FI",
            },
            {
                "instrument_id": "R4_BNP_USD_MONEY_MARKET_C_USD_ACC",
                "fund_name": 'BNP USD MONEY MARKET "C" (USD) ACC',
            }
        ]
    )

    instruments = _fallback_instruments_from_transactions(raw, known_ids=set())
    by_id = {instrument.instrument_id: instrument for instrument in instruments}

    assert by_id["R4_RENTA_4_FONCUENTA_AHORRO_FI"].name == "RENTA 4 FONCUENTA AHORRO, FI"
    assert by_id["R4_BNP_USD_MONEY_MARKET_C_USD_ACC"].name == 'BNP USD MONEY MARKET "C" (USD) ACC'
    assert by_id["R4_BNP_USD_MONEY_MARKET_C_USD_ACC"].currency == "USD"
    assert by_id["R4_BNP_USD_MONEY_MARKET_C_USD_ACC"].metadata["created_from"] == "renta4_trade"


def test_morningstar_text_parser_extracts_report_metadata():
    text = """Informe a 22 sep. 2026
Vanguard U.S. 500 Stock Index Fund Investor USD Accumulation
Morningstar Medalist Rating™
Œ
Benchmark Morningstar
Morningstar US Large-Mid Cap
Market NR USD
Usado a lo largo del informe
Benchmark del fondo
S&P 500 NR USD
Rating Morningstar™
QQQQ
Categoría Morningstar™
RV USA Cap. Grande Blend
Medidas de riesgo
Sharpe 3a 1,11
Volatilidad a 3a. 13,04
Rentab. acum. % Fondo Ref. Cat
3 meses 3,67 3,62 2,83
Datos acumulados a 21/09/2026
Distribución de activos
Acciones 100,00
Obligaciones 0,00
Style Box™ de Morningstar
Principales Posiciones
Nombre del activo Sector %
NVIDIA Corp a 8,09
% de activos en las 10 mayores posiciones 37,85
Desglose por regiones % Fondo
América 99,59
See disclosures for more details
"""

    report = Renta4ImportParser().parse_morningstar_text(text)

    assert report.name == "Vanguard U.S. 500 Stock Index Fund Investor USD Accumulation"
    assert report.report_date == date(2026, 9, 22)
    assert report.rating == "QQQQ"
    assert report.category == "RV USA Cap. Grande Blend"
    assert report.benchmark == "Morningstar US Large-Mid Cap Market NR USD"
    assert report.risk_measures["Sharpe 3a"] == pytest.approx(1.11)
    assert report.trailing_returns["3 meses"]["fund"] == pytest.approx(3.67)
    assert report.asset_allocation["Acciones"] == pytest.approx(100)
    assert report.top_holdings == [{"name": "NVIDIA Corp", "weight": pytest.approx(8.09)}]
    assert report.regions["América"] == pytest.approx(99.59)
