"""Instrument model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Instrument:
    """Metadata describing an investable instrument.

    Mutual funds should use their ISIN as ``instrument_id`` when available.
    ``data_symbol`` is optional because some providers use the instrument ID
    directly.
    """

    instrument_id: str
    name: str
    instrument_type: str
    asset_class: str | None
    currency: str
    data_source: str
    data_symbol: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.instrument_id = self.instrument_id.strip()
        self.name = self.name.strip()
        self.instrument_type = self.instrument_type.strip().lower()
        self.asset_class = self.asset_class.strip().lower() if self.asset_class else None
        self.currency = self.currency.strip().upper()
        self.data_source = self.data_source.strip().lower()
        if self.data_symbol is not None:
            self.data_symbol = self.data_symbol.strip() or None

        if not self.instrument_id:
            raise ValueError("instrument_id is required")
        if not self.name:
            raise ValueError("name is required")
        if not self.instrument_type:
            raise ValueError("instrument_type is required")
        if not self.currency:
            raise ValueError("currency is required")
        if not self.data_source:
            raise ValueError("data_source is required")

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation."""
        return {
            "instrument_id": self.instrument_id,
            "name": self.name,
            "instrument_type": self.instrument_type,
            "asset_class": self.asset_class,
            "currency": self.currency,
            "data_source": self.data_source,
            "data_symbol": self.data_symbol,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Instrument":
        """Build an instrument from a dictionary."""
        return cls(
            instrument_id=data["instrument_id"],
            name=data["name"],
            instrument_type=data["instrument_type"],
            asset_class=data.get("asset_class"),
            currency=data["currency"],
            data_source=data["data_source"],
            data_symbol=data.get("data_symbol"),
            metadata=dict(data.get("metadata") or {}),
        )
