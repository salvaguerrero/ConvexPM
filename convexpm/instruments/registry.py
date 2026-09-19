"""Local Parquet-backed instrument registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from convexpm.instruments.instrument import Instrument

INSTRUMENT_COLUMNS = [
    "instrument_id",
    "name",
    "instrument_type",
    "asset_class",
    "currency",
    "data_source",
    "data_symbol",
    "metadata_json",
]


class InstrumentRegistry:
    """Simple in-memory registry persisted to Parquet."""

    def __init__(self, path: str | Path = "data/instruments.parquet", *, auto_load: bool = True) -> None:
        self.path = Path(path)
        self._instruments: dict[str, Instrument] = {}
        if auto_load and self.path.exists():
            self.load()

    def add(self, instrument: Instrument, *, replace: bool = True) -> None:
        """Add an instrument to the registry."""
        if not replace and instrument.instrument_id in self._instruments:
            raise KeyError(f"Instrument already exists: {instrument.instrument_id}")
        self._instruments[instrument.instrument_id] = instrument

    def add_many(self, instruments: Iterable[Instrument], *, replace: bool = True) -> None:
        """Add multiple instruments."""
        for instrument in instruments:
            self.add(instrument, replace=replace)

    def get(self, instrument_id: str) -> Instrument:
        """Return an instrument by ID."""
        try:
            return self._instruments[instrument_id]
        except KeyError as exc:
            raise KeyError(f"Unknown instrument_id: {instrument_id}") from exc

    def all(self) -> list[Instrument]:
        """Return all registered instruments sorted by ID."""
        return [self._instruments[key] for key in sorted(self._instruments)]

    def __contains__(self, instrument_id: object) -> bool:
        return instrument_id in self._instruments

    def __len__(self) -> int:
        return len(self._instruments)

    def save(self) -> None:
        """Persist instruments to Parquet."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for instrument in self.all():
            data = instrument.to_dict()
            metadata = data.pop("metadata")
            data["metadata_json"] = json.dumps(metadata, sort_keys=True)
            rows.append(data)
        pd.DataFrame(rows, columns=INSTRUMENT_COLUMNS).to_parquet(self.path, index=False)

    def load(self) -> None:
        """Load instruments from Parquet, replacing the current registry."""
        if not self.path.exists():
            self._instruments = {}
            return
        df = pd.read_parquet(self.path)
        instruments: dict[str, Instrument] = {}
        for _, row in df.iterrows():
            metadata_raw = row.get("metadata_json")
            metadata = json.loads(metadata_raw) if isinstance(metadata_raw, str) and metadata_raw else {}
            instrument = Instrument(
                instrument_id=row["instrument_id"],
                name=row["name"],
                instrument_type=row["instrument_type"],
                asset_class=row.get("asset_class") if pd.notna(row.get("asset_class")) else None,
                currency=row["currency"],
                data_source=row["data_source"],
                data_symbol=row.get("data_symbol") if pd.notna(row.get("data_symbol")) else None,
                metadata=metadata,
            )
            instruments[instrument.instrument_id] = instrument
        self._instruments = instruments
