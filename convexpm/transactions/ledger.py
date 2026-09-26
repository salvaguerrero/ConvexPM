"""Interactive trade ledger."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd

from convexpm.transactions.repository import trades_to_dataframe
from convexpm.transactions.trade import Trade

if TYPE_CHECKING:
    from convexpm.instruments import InstrumentRegistry


class TradeLedger(Sequence[Trade]):
    """List-like trade ledger that displays as a DataFrame in the REPL."""

    def __init__(
        self,
        trades: Iterable[Trade] = (),
        *,
        registry: "InstrumentRegistry | None" = None,
    ) -> None:
        self._trades = list(trades)
        self.registry = registry

    def __iter__(self) -> Iterator[Trade]:
        return iter(self._trades)

    def __len__(self) -> int:
        return len(self._trades)

    def __bool__(self) -> bool:
        return bool(self._trades)

    def __getitem__(self, key: int | slice | str) -> Any:
        if isinstance(key, int):
            return self.detail(key)
        if isinstance(key, slice):
            return TradeLedger(self._trades[key], registry=self.registry)
        return self.to_dataframe()[key]

    def __add__(self, other: Iterable[Trade]) -> list[Trade]:
        return list(self._trades) + list(other)

    def __radd__(self, other: Iterable[Trade]) -> list[Trade]:
        return list(other) + list(self._trades)

    def append(self, trade: Trade) -> None:
        self._trades.append(trade)

    def extend(self, trades: Iterable[Trade]) -> None:
        self._trades.extend(trades)

    def sort(self, *, key: Callable[[Trade], Any] | None = None, reverse: bool = False) -> None:
        self._trades.sort(key=key, reverse=reverse)

    def to_list(self) -> list[Trade]:
        """Return a plain list of trades."""
        return list(self._trades)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the ledger as a DataFrame."""
        df = trades_to_dataframe(self._trades)
        if df.empty:
            return df
        position = df.columns.get_loc("instrument_id") + 1
        names = pd.Series(
            [self._trade_instrument_name(trade) for trade in self._trades],
            dtype=object,
        )
        df.insert(position, "instrument_name", names)
        df.insert(
            df.columns.get_loc("amount") + 1,
            "price",
            df.apply(_amount_per_quantity, axis=1),
        )
        df.insert(
            df.columns.get_loc("price") + 1,
            "price_without_fee",
            df.apply(_price_without_fee, axis=1),
        )
        df = df.set_index("trade_id", drop=False)
        df.index.name = "trade_id"
        return df

    def detail(self, number: int) -> pd.Series:
        """Return detailed information for trade number ``number``.

        Trade numbers are one-based for REPL ergonomics, so ``ledger[1]`` is
        the first displayed row.
        """
        if number == 0:
            raise IndexError("Trade numbers are one-based. Use 1 for the first trade.")
        index = number - 1 if number > 0 else number
        trade = self._trades[index]
        data = trade.to_dict()
        data["instrument_name"] = self._trade_instrument_name(trade)
        data["price"] = _amount_per_quantity(data)
        data["price_without_fee"] = _price_without_fee(data)
        metadata = data.pop("metadata", {})
        detail = pd.Series(data)
        if metadata:
            metadata_detail = pd.json_normalize(metadata, sep=".").T.iloc[:, 0]
            metadata_detail.index = [f"metadata.{item}" for item in metadata_detail.index]
            detail = pd.concat([detail, metadata_detail])
        return detail

    def delete(self, number_or_trade_id: int | str) -> Trade:
        """Delete a trade by one-based row number or ``trade_id`` and return it."""
        if isinstance(number_or_trade_id, int):
            if number_or_trade_id == 0:
                raise IndexError("Trade numbers are one-based. Use 1 for the first trade.")
            index = number_or_trade_id - 1 if number_or_trade_id > 0 else number_or_trade_id
            return self._trades.pop(index)

        for index, trade in enumerate(self._trades):
            if trade.trade_id == number_or_trade_id:
                return self._trades.pop(index)
        raise KeyError(f"Unknown trade_id: {number_or_trade_id}")

    def __repr__(self) -> str:
        df = self.to_dataframe()
        if df.empty:
            return "TradeLedger(empty)"
        display = df.loc[
            :,
            [
                "date",
                "instrument_name",
                "side",
                "quantity",
                "amount",
                "price",
                "price_without_fee",
                "fees",
                "currency",
                "account",
            ],
        ].copy()
        display.index = range(1, len(display) + 1)
        display.index.name = "n"
        return display.to_string()

    def _repr_html_(self) -> str:
        df = self.to_dataframe()
        if df.empty:
            return "<p>TradeLedger(empty)</p>"
        display = df.copy()
        display.index = range(1, len(display) + 1)
        display.index.name = "n"
        return display._repr_html_()

    def _instrument_name(self, instrument_id: str) -> str | None:
        if self.registry is None:
            return None
        try:
            return self.registry.get(instrument_id).name
        except KeyError:
            return None

    def _trade_instrument_name(self, trade: Trade) -> str | None:
        registry_name = self._instrument_name(trade.instrument_id)
        if registry_name:
            return registry_name
        metadata = trade.metadata or {}
        fallback = metadata.get("fund_name") or metadata.get("description")
        return str(fallback) if fallback else None


def _amount_per_quantity(row: Any) -> float | None:
    amount = row.get("amount")
    quantity = row.get("quantity")
    if pd.isna(amount) or pd.isna(quantity) or float(quantity) == 0:
        return None
    return float(amount) / float(quantity)


def _price_without_fee(row: Any) -> float | None:
    value_per_unit = row.get("value_per_unit")
    if not pd.isna(value_per_unit):
        return float(value_per_unit)
    return _amount_per_quantity(row)
