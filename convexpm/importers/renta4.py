"""Renta 4 fund NAV, operation, and Morningstar report parsers."""

from __future__ import annotations

import re
import hashlib
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from convexpm.instruments import Instrument
from convexpm.market_data import MARKET_DATA_COLUMNS, Renta4CSVProvider
from convexpm.transactions import Trade

ISIN_RE = re.compile(r"\b[A-Z]{2}[A-Z0-9]{10}\b")


@dataclass(slots=True)
class MorningstarReport:
    """Structured summary extracted from a Morningstar PDF report."""

    name: str
    report_date: date | None
    source_path: str | None = None
    instrument_id: str | None = None
    rating: str | None = None
    category: str | None = None
    benchmark: str | None = None
    fund_benchmark: str | None = None
    risk_measures: dict[str, float | str] | None = None
    trailing_returns: dict[str, dict[str, float | None]] | None = None
    asset_allocation: dict[str, float | str] | None = None
    top_holdings: list[dict[str, object]] | None = None
    regions: dict[str, float | str] | None = None
    raw_text: str = ""

    def to_metadata(self) -> dict[str, Any]:
        """Return a compact metadata payload suitable for Instrument.metadata."""
        return {
            "name": self.name,
            "report_date": self.report_date.isoformat() if self.report_date else None,
            "source_path": self.source_path,
            "rating": self.rating,
            "category": self.category,
            "benchmark": self.benchmark,
            "fund_benchmark": self.fund_benchmark,
            "risk_measures": self.risk_measures or {},
            "trailing_returns": self.trailing_returns or {},
            "asset_allocation": self.asset_allocation or {},
            "top_holdings": self.top_holdings or [],
            "regions": self.regions or {},
        }


@dataclass(slots=True)
class Renta4ImportResult:
    """Normalized Renta 4 import output."""

    market_data: pd.DataFrame
    instruments: list[Instrument]
    trades: list[Trade]
    raw_transactions: pd.DataFrame
    morningstar_reports: list[MorningstarReport]


class Renta4ImportParser:
    """Parse a directory of Renta 4 NAV CSVs, operation XLS files, and reports."""

    source = "renta4"

    def __init__(self, nav_provider: Renta4CSVProvider | None = None) -> None:
        self.nav_provider = nav_provider or Renta4CSVProvider()

    def parse_directory(self, path: str | Path) -> Renta4ImportResult:
        """Parse all supported Renta 4 files in a directory."""
        root = Path(path)
        market_frames: list[pd.DataFrame] = []
        instruments_by_id: dict[str, Instrument] = {}
        for csv_path in sorted(root.glob("vliq_*_r4.csv")):
            market_df, instruments = self.nav_provider.parse(csv_path)
            market_frames.append(market_df)
            for instrument in instruments:
                instruments_by_id[instrument.instrument_id] = instrument

        reports = [self.parse_morningstar_pdf(pdf_path) for pdf_path in sorted(root.glob("morningstarreport*.pdf"))]
        for report in reports:
            match = _best_instrument_match(report.name, instruments_by_id.values())
            if match is not None:
                report.instrument_id = match.instrument_id
                match.metadata["morningstar"] = report.to_metadata()

        all_trades: list[Trade] = []
        raw_frames: list[pd.DataFrame] = []
        instruments_for_matching = list(instruments_by_id.values())
        name_map = _instrument_name_map(instruments_for_matching)
        for xls_path in sorted(root.glob("*.xls")):
            trades, raw = self.parse_transactions_xls(
                xls_path,
                instrument_name_map=name_map,
                instruments=instruments_for_matching,
            )
            all_trades.extend(trades)
            raw_frames.append(raw)
            for instrument in _fallback_instruments_from_transactions(raw, known_ids=set(instruments_by_id)):
                instruments_by_id[instrument.instrument_id] = instrument

        market_data = (
            pd.concat(market_frames, ignore_index=True)
            if market_frames
            else pd.DataFrame(columns=MARKET_DATA_COLUMNS)
        )
        raw_transactions = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()

        return Renta4ImportResult(
            market_data=market_data,
            instruments=[instruments_by_id[key] for key in sorted(instruments_by_id)],
            trades=sorted(all_trades, key=lambda trade: (trade.date, trade.instrument_id, trade.trade_id)),
            raw_transactions=raw_transactions,
            morningstar_reports=reports,
        )

    def parse_transactions_xls(
        self,
        path: str | Path,
        *,
        instrument_name_map: dict[str, str] | None = None,
        instruments: Iterable[Instrument] | None = None,
    ) -> tuple[list[Trade], pd.DataFrame]:
        """Parse Renta 4 fund-operation XLS exports."""
        df = pd.read_excel(path, header=None)
        return self.parse_transactions_dataframe(
            df,
            source_path=str(path),
            instrument_name_map=instrument_name_map,
            instruments=instruments,
        )

    def parse_transactions_dataframe(
        self,
        df: pd.DataFrame,
        *,
        source_path: str | None = None,
        instrument_name_map: dict[str, str] | None = None,
        instruments: Iterable[Instrument] | None = None,
    ) -> tuple[list[Trade], pd.DataFrame]:
        """Parse a Renta 4 operations DataFrame."""
        instrument_name_map = instrument_name_map or {}
        instruments = list(instruments or [])
        header_row = _find_header_row(df)
        rows: list[dict[str, Any]] = []
        current_fund_name: str | None = None
        for row_index in range(header_row + 1, len(df)):
            row = df.iloc[row_index]
            first_value = row.iloc[0]
            trade_date = _parse_date(first_value)
            if trade_date is None:
                name = _clean_text(first_value)
                if name:
                    current_fund_name = name
                continue
            if current_fund_name is None:
                continue
            instrument_id = _resolve_instrument_id(current_fund_name, instrument_name_map, instruments)
            operation_type = _clean_text(row.iloc[1])
            gross_amount = _parse_number(row.iloc[4])
            commission = _parse_number(row.iloc[5]) or 0.0
            withholding = _parse_number(row.iloc[6]) or 0.0
            net_amount = _parse_number(row.iloc[7])
            quantity = _parse_number(row.iloc[2])
            side = _renta4_side(operation_type)
            source_row = int(row_index) + 1
            rows.append(
                {
                    "date": pd.Timestamp(trade_date),
                    "fund_name": current_fund_name,
                    "instrument_id": instrument_id,
                    "operation_type": operation_type,
                    "side": side,
                    "quantity": quantity,
                    "gross_dividend": _parse_number(row.iloc[3]),
                    "gross_amount": gross_amount,
                    "commission": commission,
                    "withholding": withholding,
                    "net_amount": net_amount,
                    "status": _clean_text(row.iloc[8]),
                    "source_path": source_path,
                    "source_row": source_row,
                }
            )

        raw = pd.DataFrame(rows)
        trades = [_renta4_trade_from_row(row) for _, row in raw.dropna(subset=["side"]).iterrows()]
        return trades, raw

    def parse_morningstar_pdf(self, path: str | Path) -> MorningstarReport:
        """Parse a Morningstar PDF report using pypdf text extraction."""
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - exercised only without dependency installed.
            raise ImportError("Install pypdf to parse Morningstar PDF reports") from exc

        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        report = self.parse_morningstar_text(text)
        report.source_path = str(path)
        return report

    def parse_morningstar_text(self, text: str) -> MorningstarReport:
        """Parse extracted Morningstar report text."""
        lines = [_clean_text(line) for line in text.splitlines()]
        lines = [line for line in lines if line]
        name = lines[1] if len(lines) > 1 else "Unknown Morningstar Fund"
        return MorningstarReport(
            name=name,
            report_date=_parse_morningstar_report_date(lines[0] if lines else ""),
            rating=_line_after(lines, "Rating Morningstar"),
            category=_line_after(lines, "Categoría Morningstar"),
            benchmark=_collect_after(lines, "Benchmark Morningstar", stop_prefixes=["Usado a lo largo", "Benchmark del fondo"]),
            fund_benchmark=_collect_after(lines, "Benchmark del fondo", stop_prefixes=["Rating Morningstar", "Categoría Morningstar"]),
            risk_measures=_parse_labeled_numeric_block(lines, "Medidas de riesgo", "Rentab. acum."),
            trailing_returns=_parse_trailing_returns(lines),
            asset_allocation=_parse_labeled_numeric_block(lines, "Distribución de activos", "Style Box"),
            top_holdings=_parse_top_holdings(lines),
            regions=_parse_labeled_numeric_block(lines, "Desglose por regiones", "See disclosures"),
            raw_text=text,
        )


def _renta4_trade_from_row(row: pd.Series) -> Trade:
    quantity = abs(float(row["quantity"])) if pd.notna(row.get("quantity")) else None
    amount = abs(float(row["gross_amount"])) if pd.notna(row.get("gross_amount")) else None
    fees = abs(float(row.get("commission") or 0.0)) + abs(float(row.get("withholding") or 0.0))
    value_per_unit = amount / quantity if amount is not None and quantity else None
    metadata = {
        "source": "renta4",
        "fund_name": row["fund_name"],
        "operation_type": row["operation_type"],
        "gross_dividend": _none_if_na(row.get("gross_dividend")),
        "gross_amount": _none_if_na(row.get("gross_amount")),
        "commission": _none_if_na(row.get("commission")),
        "withholding": _none_if_na(row.get("withholding")),
        "net_amount": _none_if_na(row.get("net_amount")),
        "status": row.get("status"),
        "source_path": row.get("source_path"),
        "source_row": int(row["source_row"]),
    }
    return Trade(
        trade_id=_stable_trade_id(
            "renta4",
            [
                str(row["date"].date()),
                str(row["instrument_id"]),
                str(row["side"]),
                str(row.get("operation_type")),
                _stable_number(quantity),
                _stable_number(amount),
                _stable_number(fees),
            ],
        ),
        date=row["date"].date(),
        instrument_id=row["instrument_id"],
        side=row["side"],
        quantity=quantity,
        amount=amount,
        value_per_unit=value_per_unit,
        fees=fees,
        currency="EUR",
        account="renta4",
        tags=["renta4"],
        metadata=metadata,
    )


def _stable_trade_id(source: str, values: list[str]) -> str:
    digest = hashlib.sha1("|".join(values).encode("utf-8")).hexdigest()[:12]
    return f"{source}-{digest}"


def _stable_number(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.8f}"


def _find_header_row(df: pd.DataFrame) -> int:
    for index, value in enumerate(df.iloc[:, 0].tolist()):
        if _clean_text(value).lower() == "fecha":
            return index
    raise ValueError("Could not find Renta 4 transaction header row")


def _renta4_side(operation_type: str) -> str | None:
    normalized = _strip_accents(operation_type).upper()
    if "SUSCRIPCION" in normalized or "ENTRADA" in normalized:
        return "BUY"
    if "REEMBOLSO" in normalized or "SALIDA" in normalized:
        return "SELL"
    return None


def _instrument_name_map(instruments: Iterable[Instrument]) -> dict[str, str]:
    return {_normalize_name(instrument.name): instrument.instrument_id for instrument in instruments}


def _resolve_instrument_id(
    fund_name: str,
    instrument_name_map: dict[str, str],
    instruments: Iterable[Instrument],
) -> str:
    normalized = _normalize_name(fund_name)
    if normalized in instrument_name_map:
        return instrument_name_map[normalized]
    match = _best_instrument_match(fund_name, instruments)
    return match.instrument_id if match is not None else _slug_id(fund_name)


def _fallback_instruments_from_transactions(raw: pd.DataFrame, *, known_ids: set[str]) -> list[Instrument]:
    if raw.empty:
        return []
    instruments = []
    for _, row in raw.drop_duplicates("instrument_id").iterrows():
        instrument_id = row["instrument_id"]
        if instrument_id in known_ids:
            continue
        instruments.append(
            Instrument(
                instrument_id=instrument_id,
                name=row["fund_name"],
                instrument_type="fund",
                asset_class="fund",
                currency=_fund_currency(str(row["fund_name"])),
                data_source="renta4",
                metadata={
                    "source": "renta4_transactions",
                    "fund_name": row["fund_name"],
                    "created_from": "renta4_trade",
                },
            )
        )
    return instruments


def _fund_currency(fund_name: str) -> str:
    upper_name = fund_name.upper()
    for currency in ["EUR", "USD", "GBP", "CHF", "JPY"]:
        if re.search(rf"\b{currency}\b|\({currency}\)", upper_name):
            return currency
    return "EUR"


def _best_instrument_match(name: str, instruments: Iterable[Instrument]) -> Instrument | None:
    name_tokens = set(_normalize_name(name).split())
    best_score = 0.0
    best_instrument: Instrument | None = None
    for instrument in instruments:
        tokens = set(_normalize_name(instrument.name).split())
        if not tokens or not name_tokens:
            continue
        score = len(tokens & name_tokens) / len(tokens | name_tokens)
        if score > best_score:
            best_score = score
            best_instrument = instrument
    return best_instrument if best_score >= 0.30 else None


def _parse_trailing_returns(lines: list[str]) -> dict[str, dict[str, float | None]]:
    start = _index_of(lines, "Rentab. acum.")
    if start is None:
        return {}
    result: dict[str, dict[str, float | None]] = {}
    labels = ["3 meses", "6 meses", "1 año", "3 años anualiz.", "5 años anualiz.", "10 años anualiz."]
    for line in lines[start + 1 :]:
        if line.startswith("Datos acumulados"):
            break
        for label in labels:
            if line.startswith(label):
                values = [_parse_morningstar_number(item) for item in line[len(label) :].split()]
                result[label] = {
                    "fund": values[0] if len(values) > 0 else None,
                    "reference": values[1] if len(values) > 1 else None,
                    "category": values[2] if len(values) > 2 else None,
                }
                break
    return result


def _parse_top_holdings(lines: list[str]) -> list[dict[str, object]]:
    start = _index_of(lines, "Principales Posiciones")
    if start is None:
        return []
    holdings: list[dict[str, object]] = []
    for line in lines[start + 2 :]:
        if line.startswith("% de activos") or line.startswith("Número total"):
            break
        match = re.match(r"(.+?)\s+\S\s+(-?\d+(?:,\d+)?)$", line)
        if match is None:
            continue
        holdings.append({"name": match.group(1).strip(), "weight": _parse_morningstar_number(match.group(2))})
    return holdings


def _parse_labeled_numeric_block(lines: list[str], start_label: str, stop_label: str) -> dict[str, float | str]:
    start = _index_of(lines, start_label)
    if start is None:
        return {}
    result: dict[str, float | str] = {}
    for line in lines[start + 1 :]:
        if line.startswith(stop_label):
            break
        match = re.match(r"(.+?)\s+(-?\d+(?:,\d+)?|-|inf med|med|Alto|Bajo|MedianoAlto)$", line)
        if match is None:
            continue
        value = _parse_morningstar_number(match.group(2))
        result[match.group(1).strip()] = value if value is not None else match.group(2)
    return result


def _line_after(lines: list[str], label: str) -> str | None:
    index = _index_of(lines, label)
    if index is None or index + 1 >= len(lines):
        return None
    return lines[index + 1]


def _collect_after(lines: list[str], label: str, *, stop_prefixes: list[str]) -> str | None:
    index = _index_of(lines, label)
    if index is None:
        return None
    collected = []
    for line in lines[index + 1 :]:
        if any(line.startswith(stop) for stop in stop_prefixes):
            break
        collected.append(line)
    return " ".join(collected) or None


def _index_of(lines: list[str], label: str) -> int | None:
    for index, line in enumerate(lines):
        if line.startswith(label):
            return index
    return None


def _parse_morningstar_report_date(value: str) -> date | None:
    match = re.search(r"(\d{1,2})\s+([A-Za-záéíóúñ.]+)\s+(\d{4})", value)
    if match is None:
        return None
    months = {
        "ene": 1,
        "feb": 2,
        "mar": 3,
        "abr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "ago": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dic": 12,
    }
    month_key = _strip_accents(match.group(2).rstrip(".")).lower()[:3]
    month = months.get(month_key)
    if month is None:
        return None
    return date(int(match.group(3)), month, int(match.group(1)))


def _parse_morningstar_number(value: str) -> float | None:
    value = value.strip()
    if value == "-":
        return None
    try:
        return float(value.replace(".", "").replace(",", "."))
    except ValueError:
        return None


def _parse_date(value: object) -> date | None:
    if value is None or pd.isna(value):
        return None
    try:
        return pd.to_datetime(value, dayfirst=True, errors="raise").date()
    except (TypeError, ValueError):
        return None


def _parse_number(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    return float(text.replace(".", "").replace(",", "."))


def _none_if_na(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    return float(value) if isinstance(value, (int, float)) else value


def _clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _normalize_name(value: str) -> str:
    stripped = _strip_accents(value).lower()
    return re.sub(r"[^a-z0-9]+", " ", stripped).strip()


def _slug_id(value: str) -> str:
    normalized = _strip_accents(value).upper()
    slug = re.sub(r"[^A-Z0-9]+", "_", normalized).strip("_")
    return f"R4_{slug}"


def _strip_accents(value: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(char)
    )
