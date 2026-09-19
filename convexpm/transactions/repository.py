"""Parquet persistence helpers for trades."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from convexpm.transactions.trade import Trade
from convexpm.utils.dates import to_date

TRADE_COLUMNS = [
    "trade_id",
    "date",
    "instrument_id",
    "side",
    "quantity",
    "amount",
    "value_per_unit",
    "fees",
    "currency",
    "account",
    "tags_json",
    "metadata_json",
]


def trades_to_dataframe(trades: Iterable[Trade]) -> pd.DataFrame:
    """Convert trades to the stable Parquet schema."""
    rows = []
    for trade in trades:
        rows.append(
            {
                "trade_id": trade.trade_id,
                "date": pd.Timestamp(trade.date),
                "instrument_id": trade.instrument_id,
                "side": trade.side,
                "quantity": trade.quantity,
                "amount": trade.amount,
                "value_per_unit": trade.value_per_unit,
                "fees": trade.fees,
                "currency": trade.currency,
                "account": trade.account,
                "tags_json": json.dumps(trade.tags),
                "metadata_json": json.dumps(trade.metadata, sort_keys=True),
            }
        )
    return pd.DataFrame(rows, columns=TRADE_COLUMNS)


def save_trades(trades: Iterable[Trade], path: str | Path = "data/trades.parquet") -> None:
    """Persist trades to Parquet."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trades_to_dataframe(trades).to_parquet(output_path, index=False)


def load_trades(path: str | Path = "data/trades.parquet") -> list[Trade]:
    """Load trades from Parquet. Missing files return an empty list."""
    input_path = Path(path)
    if not input_path.exists():
        return []
    df = pd.read_parquet(input_path)
    trades: list[Trade] = []
    for _, row in df.iterrows():
        tags_raw = row.get("tags_json")
        metadata_raw = row.get("metadata_json")
        quantity = row.get("quantity")
        amount = row.get("amount")
        value_per_unit = row.get("value_per_unit")
        trades.append(
            Trade(
                trade_id=row["trade_id"],
                date=to_date(row["date"]),
                instrument_id=row["instrument_id"],
                side=row["side"],
                quantity=float(quantity) if pd.notna(quantity) else None,
                amount=float(amount) if pd.notna(amount) else None,
                value_per_unit=float(value_per_unit) if pd.notna(value_per_unit) else None,
                fees=float(row.get("fees") or 0.0),
                currency=row.get("currency") if pd.notna(row.get("currency")) else None,
                account=row.get("account") if pd.notna(row.get("account")) else None,
                tags=json.loads(tags_raw) if isinstance(tags_raw, str) and tags_raw else [],
                metadata=json.loads(metadata_raw) if isinstance(metadata_raw, str) and metadata_raw else {},
            )
        )
    return sorted(trades, key=lambda trade: (trade.date, trade.trade_id))
