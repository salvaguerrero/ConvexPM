"""Trade model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from convexpm.utils.dates import DateLike, to_date

SUPPORTED_SIDES = {"BUY", "SELL"}


@dataclass(slots=True)
class Trade:
    """A BUY or SELL transaction.

    ``quantity`` can be omitted when ``amount`` is known. If ``value_per_unit``
    is also absent, portfolio valuation can derive quantity from market data at
    the trade date.
    """

    instrument_id: str
    side: str
    date: DateLike
    quantity: float | None = None
    amount: float | None = None
    value_per_unit: float | None = None
    fees: float = 0.0
    currency: str | None = None
    account: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    trade_id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        self.instrument_id = self.instrument_id.strip()
        self.side = self.side.strip().upper()
        self.date = to_date(self.date)
        self.currency = self.currency.strip().upper() if self.currency else None
        self.account = self.account.strip() if self.account else None

        if not self.instrument_id:
            raise ValueError("instrument_id is required")
        if self.side not in SUPPORTED_SIDES:
            raise ValueError(f"side must be one of {sorted(SUPPORTED_SIDES)}")
        if self.quantity is None and self.amount is None:
            raise ValueError("Trade requires either quantity or amount")
        if self.quantity is not None and self.quantity < 0:
            raise ValueError("quantity must be non-negative")
        if self.amount is not None and self.amount < 0:
            raise ValueError("amount must be non-negative")
        if self.value_per_unit is not None and self.value_per_unit <= 0:
            raise ValueError("value_per_unit must be positive")
        if self.fees < 0:
            raise ValueError("fees must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation."""
        return {
            "trade_id": self.trade_id,
            "date": self.date,
            "instrument_id": self.instrument_id,
            "side": self.side,
            "quantity": self.quantity,
            "amount": self.amount,
            "value_per_unit": self.value_per_unit,
            "fees": self.fees,
            "currency": self.currency,
            "account": self.account,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Trade":
        """Build a trade from a dictionary."""
        return cls(
            trade_id=data.get("trade_id") or str(uuid4()),
            date=data["date"],
            instrument_id=data["instrument_id"],
            side=data["side"],
            quantity=data.get("quantity"),
            amount=data.get("amount"),
            value_per_unit=data.get("value_per_unit"),
            fees=float(data.get("fees") or 0.0),
            currency=data.get("currency"),
            account=data.get("account"),
            tags=list(data.get("tags") or []),
            metadata=dict(data.get("metadata") or {}),
        )
