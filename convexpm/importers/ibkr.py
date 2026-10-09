"""Interactive Brokers transaction statement parser."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from convexpm.instruments import Instrument
from convexpm.transactions import Trade


@dataclass(slots=True)
class IBKRImportResult:
    """Normalized IBKR import output."""

    trades: list[Trade]
    cash_trades: list[Trade]
    cash_transactions: pd.DataFrame
    raw_transactions: pd.DataFrame
    summary: dict[str, Any]

    def instruments(
        self,
        *,
        yahoo_symbols: dict[str, str] | None = None,
        known_ids: set[str] | None = None,
        fetch_yahoo_metadata: bool = False,
        ticker_factory: Any | None = None,
    ) -> list[Instrument]:
        """Infer missing tradable instruments from IBKR trades."""
        yahoo_symbols = yahoo_symbols or {}
        known_ids = known_ids or set()
        instruments: list[Instrument] = []
        trade_rows = self.raw_transactions.loc[
            self.raw_transactions["transaction_type"].isin(["Buy", "Sell"])
        ]
        for symbol in sorted({trade.instrument_id for trade in self.trades} - known_ids):
            rows = trade_rows.loc[trade_rows["symbol"] == symbol]
            if rows.empty:
                continue
            first_row = rows.iloc[0]
            data_symbol = yahoo_symbols.get(symbol, symbol)
            name = str(first_row["description"]).strip() or symbol
            currency = str(first_row.get("price_currency") or first_row.get("base_currency") or "EUR").upper()
            metadata = {
                "source": "ibkr_transactions",
                "ibkr_symbol": symbol,
                "yahoo_symbol": data_symbol,
            }
            if fetch_yahoo_metadata:
                try:
                    ticker = ticker_factory(data_symbol) if ticker_factory is not None else _yahoo_ticker(data_symbol)
                    yahoo_info = ticker.get_info() if hasattr(ticker, "get_info") else getattr(ticker, "info", {})
                    if isinstance(yahoo_info, dict):
                        name = yahoo_info.get("longName") or yahoo_info.get("shortName") or name
                        currency = yahoo_info.get("currency") or currency
                        metadata["yahoo_quote_type"] = yahoo_info.get("quoteType")
                        metadata["yahoo_exchange"] = yahoo_info.get("exchange")
                except Exception as exc:
                    metadata["yahoo_error"] = str(exc)

            instruments.append(
                Instrument(
                    instrument_id=symbol,
                    name=name,
                    instrument_type="equity",
                    asset_class="equity",
                    currency=currency,
                    data_source="yahoo",
                    data_symbol=data_symbol,
                    metadata={key: value for key, value in metadata.items() if value is not None},
                )
            )
        return instruments


class IBKRTransactionParser:
    """Parse IBKR activity statement transaction-history CSV exports."""

    source = "ibkr"

    def parse(self, path: str | Path) -> IBKRImportResult:
        """Parse an IBKR CSV file."""
        text = Path(path).read_text(encoding="utf-8-sig")
        return self.parse_text(text)

    def parse_text(self, text: str) -> IBKRImportResult:
        """Parse IBKR CSV text."""
        summary: dict[str, Any] = {}
        transaction_header: list[str] | None = None
        transaction_rows: list[dict[str, str]] = []

        for row_number, row in enumerate(csv.reader(StringIO(text)), start=1):
            if len(row) < 2:
                continue
            section, row_type = row[0], row[1]
            if section == "Summary" and row_type == "Data" and len(row) >= 4:
                summary[row[2].strip()] = _parse_scalar(row[3])
                continue
            if section == "Transaction History" and row_type == "Header":
                transaction_header = [item.strip() for item in row[2:]]
                continue
            if section == "Transaction History" and row_type == "Data" and transaction_header:
                values = row[2 : 2 + len(transaction_header)]
                entry = dict(zip(transaction_header, values, strict=False))
                entry["_row_number"] = str(row_number)
                transaction_rows.append(entry)

        raw = pd.DataFrame(transaction_rows)
        if raw.empty:
            return IBKRImportResult(
                trades=[],
                cash_trades=[],
                cash_transactions=pd.DataFrame(),
                raw_transactions=raw,
                summary=summary,
            )

        normalized = _normalize_transactions(raw, base_currency=str(summary.get("Base Currency") or "EUR"))
        trades = _build_trades(normalized, base_currency=str(summary.get("Base Currency") or "EUR"))
        cash = normalized.loc[~normalized["transaction_type"].isin(["Buy", "Sell"])].reset_index(drop=True)
        cash_trades = _build_cash_trades(cash, trade_rows=normalized, base_currency=str(summary.get("Base Currency") or "EUR"))
        return IBKRImportResult(
            trades=trades,
            cash_trades=cash_trades,
            cash_transactions=cash,
            raw_transactions=normalized,
            summary=summary,
        )


def _normalize_transactions(raw: pd.DataFrame, *, base_currency: str) -> pd.DataFrame:
    renamed = raw.rename(
        columns={
            "Date": "date",
            "Account": "account",
            "Description": "description",
            "Transaction Type": "transaction_type",
            "Symbol": "symbol",
            "Quantity": "quantity",
            "Price": "price",
            "Price Currency": "price_currency",
            "Gross Amount ": "gross_amount",
            "Gross Amount": "gross_amount",
            "Commission": "commission",
            "Net Amount": "net_amount",
        }
    ).copy()
    for column in ["quantity", "price", "gross_amount", "commission", "net_amount"]:
        normalized = renamed[column] if column in renamed else pd.Series([None] * len(renamed))
        renamed[column] = normalized.map(_parse_number)

    renamed["date"] = pd.to_datetime(renamed["date"], errors="coerce")
    renamed["symbol"] = renamed["symbol"].replace("-", None)
    renamed["price_currency"] = renamed["price_currency"].replace("-", None)
    renamed["base_currency"] = base_currency.upper()
    renamed["fees"] = renamed.apply(_row_fee, axis=1)
    return renamed.sort_values(["date", "_row_number"]).reset_index(drop=True)


def _build_trades(df: pd.DataFrame, *, base_currency: str) -> list[Trade]:
    trade_rows = df.loc[df["transaction_type"].isin(["Buy", "Sell"])].copy()
    if trade_rows.empty:
        return []

    fee_rows = df.loc[df["transaction_type"].eq("Transaction Fee")]
    fee_by_key = {
        key: abs(float(group["net_amount"].fillna(group["gross_amount"]).sum()))
        for key, group in fee_rows.groupby(["date", "account", "symbol"], dropna=False)
    }

    trades: list[Trade] = []
    group_columns = ["date", "account", "symbol", "transaction_type", "price_currency"]
    for key, group in trade_rows.groupby(group_columns, dropna=False, sort=True):
        trade_date, account, symbol, transaction_type, price_currency = key
        quantity = float(group["quantity"].abs().sum())
        gross_amount = float(group["gross_amount"].sum())
        commission = float(group["commission"].fillna(0.0).sum())
        net_amount = float(group["net_amount"].sum())
        transaction_costs = fee_by_key.get((trade_date, account, symbol), 0.0)
        fees = abs(commission) + transaction_costs
        amount = abs(gross_amount)
        value_per_unit = amount / quantity if quantity else None
        side = "BUY" if transaction_type == "Buy" else "SELL"
        row_numbers = [int(value) for value in group["_row_number"].tolist()]
        trade_id = _stable_trade_id(
            self_source="ibkr",
            values=[
                str(trade_date.date()),
                str(account),
                str(symbol),
                side,
                str(price_currency),
                _stable_number(quantity),
                _stable_number(amount),
                _stable_number(fees),
            ],
        )
        metadata = {
            "source": "ibkr",
            "broker_transaction_type": transaction_type,
            "description": " | ".join(sorted({str(item) for item in group["description"].dropna()})),
            "gross_amount": gross_amount,
            "commission": commission,
            "transaction_costs": transaction_costs,
            "net_amount": net_amount,
            "price_currency": None if pd.isna(price_currency) else price_currency,
            "execution_price": _weighted_price(group),
            "source_rows": row_numbers,
        }
        trades.append(
            Trade(
                trade_id=trade_id,
                date=trade_date.date(),
                instrument_id=str(symbol),
                side=side,
                quantity=quantity,
                amount=amount,
                value_per_unit=value_per_unit,
                fees=fees,
                currency=base_currency,
                account=None if pd.isna(account) else str(account),
                tags=["ibkr"],
                metadata=metadata,
            )
        )

    return sorted(trades, key=lambda trade: (trade.date, trade.instrument_id, trade.trade_id))


def _build_cash_trades(cash: pd.DataFrame, *, trade_rows: pd.DataFrame, base_currency: str) -> list[Trade]:
    if cash.empty:
        return []

    allocated_fee_keys = set(
        trade_rows.loc[trade_rows["transaction_type"].isin(["Buy", "Sell"]), ["date", "account", "symbol"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    cash_trades: list[Trade] = []
    for _, row in cash.iterrows():
        transaction_type = str(row.get("transaction_type") or "").strip()
        if _is_allocated_transaction_fee(row, allocated_fee_keys):
            continue

        net_amount = _cash_row_amount(row)
        if net_amount == 0:
            continue

        currency = str(row.get("base_currency") or base_currency).upper()
        side = "BUY" if net_amount > 0 else "SELL"
        amount = abs(net_amount)
        trade_date = row["date"]
        symbol = row.get("symbol")
        trade_id = _stable_trade_id(
            self_source="ibkr-cash",
            values=[
                str(trade_date.date()),
                str(row.get("account")),
                transaction_type,
                "" if pd.isna(symbol) else str(symbol),
                _stable_number(net_amount),
                str(row.get("_row_number")),
            ],
        )
        cash_trades.append(
            Trade(
                trade_id=trade_id,
                date=trade_date.date(),
                instrument_id=f"{currency}_CASH",
                side=side,
                quantity=amount,
                amount=amount,
                value_per_unit=1.0,
                fees=0.0,
                currency=currency,
                account=None if pd.isna(row.get("account")) else str(row.get("account")),
                tags=["ibkr", "cash"],
                metadata={
                    "source": "ibkr_cash",
                    "broker_transaction_type": transaction_type,
                    "description": None if pd.isna(row.get("description")) else str(row.get("description")),
                    "symbol": None if pd.isna(symbol) else str(symbol),
                    "gross_amount": row.get("gross_amount"),
                    "net_amount": row.get("net_amount"),
                    "source_row": int(row["_row_number"]),
                },
            )
        )
    return sorted(cash_trades, key=lambda trade: (trade.date, trade.trade_id))


def _is_allocated_transaction_fee(row: pd.Series, allocated_fee_keys: set[tuple[object, ...]]) -> bool:
    if row.get("transaction_type") != "Transaction Fee":
        return False
    key = (row.get("date"), row.get("account"), row.get("symbol"))
    return key in allocated_fee_keys


def _cash_row_amount(row: pd.Series) -> float:
    net_amount = row.get("net_amount")
    if net_amount is not None and not pd.isna(net_amount):
        return float(net_amount)
    gross_amount = row.get("gross_amount")
    if gross_amount is not None and not pd.isna(gross_amount):
        return float(gross_amount)
    return 0.0


def _weighted_price(group: pd.DataFrame) -> float | None:
    priced = group.dropna(subset=["price", "quantity"])
    if priced.empty:
        return None
    quantity = priced["quantity"].abs().sum()
    if quantity == 0:
        return None
    return float((priced["price"] * priced["quantity"].abs()).sum() / quantity)


def _row_fee(row: pd.Series) -> float:
    commission = _value_or_zero(row.get("commission"))
    if row.get("transaction_type") == "Transaction Fee":
        return abs(_value_or_zero(row.get("net_amount")) or _value_or_zero(row.get("gross_amount")))
    return abs(commission)


def _parse_number(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text == "-":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_scalar(value: str) -> object:
    parsed = _parse_number(value)
    return parsed if parsed is not None else value


def _value_or_zero(value: object) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def _stable_trade_id(*, self_source: str, values: list[str]) -> str:
    digest = hashlib.sha1("|".join(values).encode("utf-8")).hexdigest()[:12]
    return f"{self_source}-{digest}"


def _stable_number(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.8f}"


def _yahoo_ticker(symbol: str) -> Any:
    import yfinance as yf

    return yf.Ticker(symbol)
