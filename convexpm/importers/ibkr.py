"""Interactive Brokers transaction statement parser."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from convexpm.transactions import Trade


@dataclass(slots=True)
class IBKRImportResult:
    """Normalized IBKR import output."""

    trades: list[Trade]
    cash_transactions: pd.DataFrame
    raw_transactions: pd.DataFrame
    summary: dict[str, Any]


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
                cash_transactions=pd.DataFrame(),
                raw_transactions=raw,
                summary=summary,
            )

        normalized = _normalize_transactions(raw, base_currency=str(summary.get("Base Currency") or "EUR"))
        trades = _build_trades(normalized, base_currency=str(summary.get("Base Currency") or "EUR"))
        cash = normalized.loc[~normalized["transaction_type"].isin(["Buy", "Sell"])].reset_index(drop=True)
        return IBKRImportResult(
            trades=trades,
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
