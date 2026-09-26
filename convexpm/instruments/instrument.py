"""Instrument model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


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

    @classmethod
    def from_yahoo(
        cls,
        symbol: str,
        *,
        instrument_id: str | None = None,
        ticker_factory: Callable[[str], Any] | None = None,
    ) -> "Instrument":
        """Build an instrument from Yahoo Finance metadata."""
        data_symbol = symbol.strip()
        if not data_symbol:
            raise ValueError("symbol is required")

        ticker = ticker_factory(data_symbol) if ticker_factory is not None else _yahoo_ticker(data_symbol)
        info = ticker.get_info() if hasattr(ticker, "get_info") else getattr(ticker, "info", {})
        if not isinstance(info, dict):
            info = {}

        quote_type = str(info.get("quoteType") or "equity").lower()
        instrument_type = _yahoo_instrument_type(quote_type)
        name = info.get("longName") or info.get("shortName") or data_symbol
        currency = info.get("currency") or "USD"
        resolved_id = instrument_id or info.get("symbol") or data_symbol
        metadata = {
            "source": "yahoo",
            "yahoo_symbol": data_symbol,
            "quote_type": info.get("quoteType"),
            "exchange": info.get("exchange"),
            "market": info.get("market"),
            "country": info.get("country"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
        }
        metadata = {key: value for key, value in metadata.items() if value is not None}

        return cls(
            instrument_id=str(resolved_id),
            name=str(name),
            instrument_type=instrument_type,
            asset_class=instrument_type,
            currency=str(currency),
            data_source="yahoo",
            data_symbol=data_symbol,
            metadata=metadata,
        )


def _yahoo_ticker(symbol: str) -> Any:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install yfinance to build instruments from Yahoo") from exc
    return yf.Ticker(symbol)


def _yahoo_instrument_type(quote_type: str) -> str:
    mapping = {
        "etf": "etf",
        "equity": "equity",
        "mutualfund": "fund",
        "index": "index",
        "currency": "currency",
        "cryptocurrency": "crypto",
    }
    return mapping.get(quote_type.replace("_", "").replace(" ", ""), quote_type or "equity")
