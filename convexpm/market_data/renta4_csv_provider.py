"""Renta 4 CSV parser for fund NAV exports."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd

from convexpm.instruments import Instrument
from convexpm.market_data.store import MARKET_DATA_COLUMNS

ISIN_RE = re.compile(r"\b([A-Z]{2}[A-Z0-9]{10})\b")


class Renta4CSVProvider:
    """Parse Renta 4 CSV fund NAV exports into ConvexPM's normalized schema."""

    source = "renta4"

    def parse(self, path: str | Path) -> tuple[pd.DataFrame, list[Instrument]]:
        """Parse a Renta 4 CSV file."""
        text = Path(path).read_text(encoding="utf-8-sig")
        return self.parse_text(text)

    def parse_text(self, text: str) -> tuple[pd.DataFrame, list[Instrument]]:
        """Parse Renta 4 CSV text."""
        rows: list[dict[str, object]] = []
        instruments_by_id: dict[str, Instrument] = {}
        current_id: str | None = None

        for raw_line in text.splitlines():
            parts = [part.strip() for part in raw_line.strip().split(";") if part.strip()]
            if not parts:
                continue

            header = _parse_header(parts)
            if header is not None:
                current_id, name = header
                instruments_by_id[current_id] = Instrument(
                    instrument_id=current_id,
                    name=name,
                    instrument_type="fund",
                    asset_class="fund",
                    currency="EUR",
                    data_source=self.source,
                )
                continue

            if current_id is None or len(parts) < 2:
                continue

            nav_date = _parse_renta4_date(parts[0])
            if nav_date is None:
                continue

            rows.append(
                {
                    "date": pd.Timestamp(nav_date),
                    "instrument_id": current_id,
                    "value": _parse_european_number(parts[1]),
                    "currency": "EUR",
                    "value_type": "nav",
                    "source": self.source,
                }
            )

        market_df = pd.DataFrame(rows, columns=MARKET_DATA_COLUMNS)
        if not market_df.empty:
            market_df = market_df.sort_values(["instrument_id", "date"]).reset_index(drop=True)
        instruments = [instruments_by_id[key] for key in sorted(instruments_by_id)]
        return market_df, instruments

    def fetch_history(
        self,
        instrument: Instrument,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        """Renta 4 CSV exports are file-based, so use ``parse`` instead."""
        raise NotImplementedError("Use Renta4CSVProvider.parse(path) for file imports")


def _parse_header(parts: list[str]) -> tuple[str, str] | None:
    joined = " ".join(parts)
    match = ISIN_RE.search(joined)
    if match is None:
        return None
    isin = match.group(1)
    name = joined[match.end() :].strip()
    if name.startswith("-"):
        name = name[1:].strip()
    return isin, name or isin


def _parse_renta4_date(value: str) -> date | None:
    try:
        return pd.to_datetime(value, format="%d/%m/%Y", errors="raise").date()
    except (TypeError, ValueError):
        return None


def _parse_european_number(value: str) -> float:
    compact = value.strip().replace("\xa0", "").replace(" ", "")
    if "," in compact and "." in compact:
        if compact.rfind(",") > compact.rfind("."):
            compact = compact.replace(".", "").replace(",", ".")
        else:
            compact = compact.replace(",", "")
    elif "," in compact:
        compact = compact.replace(",", ".")
    return float(compact)
