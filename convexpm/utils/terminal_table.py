"""ANSI-styled terminal tables for interactive REPL displays."""

from __future__ import annotations

from typing import Any

import pandas as pd


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
BLUE = "\033[38;5;39m"
CYAN = "\033[38;5;81m"
NAVY = "\033[38;5;33m"
WHITE = "\033[38;5;255m"
GREEN = "\033[38;5;42m"
RED = "\033[38;5;203m"
YELLOW = "\033[38;5;220m"


def terminal_title(title: str, subtitle: str | None = None) -> str:
    """Return a Bloomberg-like blue terminal title block."""
    line = f"{BOLD}{BLUE}{title}{RESET}"
    if subtitle:
        line += f" {DIM}{CYAN}{subtitle}{RESET}"
    return line


class StyledDataFrame(pd.DataFrame):
    """DataFrame with a blue terminal repr and ordinary pandas behavior."""

    _metadata = ["_terminal_title", "_terminal_subtitle", "_terminal_max_col_width", "_terminal_numeric_precision"]

    @property
    def _constructor(self):
        return StyledDataFrame

    @property
    def _constructor_sliced(self):
        return StyledSeries

    def __repr__(self) -> str:
        title = getattr(self, "_terminal_title", None)
        max_col_width = getattr(self, "_terminal_max_col_width", 32)
        numeric_precision = getattr(self, "_terminal_numeric_precision", 2)
        parts = [terminal_title(title)] if title else []
        subtitle = getattr(self, "_terminal_subtitle", None)
        if subtitle:
            parts.append(f"{DIM}{CYAN}{subtitle}{RESET}")
        parts.append(
            blue_table(
                pd.DataFrame(self),
                index=True,
                max_col_width=max_col_width,
                numeric_precision=numeric_precision,
            )
        )
        return "\n".join(parts)


class StyledSeries(pd.Series):
    """Series with a blue terminal repr and ordinary pandas behavior."""

    _metadata = ["_terminal_title", "_terminal_max_col_width", "_terminal_numeric_precision"]

    @property
    def _constructor(self):
        return StyledSeries

    @property
    def _constructor_expanddim(self):
        return StyledDataFrame

    def __repr__(self) -> str:
        title = getattr(self, "_terminal_title", None) or (self.name if self.name else None)
        frame = self.rename("value").to_frame()
        max_col_width = getattr(self, "_terminal_max_col_width", 32)
        numeric_precision = getattr(self, "_terminal_numeric_precision", 2)
        parts = [terminal_title(str(title))] if title else []
        parts.append(
            blue_table(
                frame,
                index=True,
                max_col_width=max_col_width,
                numeric_precision=numeric_precision,
            )
        )
        return "\n".join(parts)


def styled_dataframe(
    df: pd.DataFrame,
    title: str | None = None,
    *,
    subtitle: str | None = None,
    max_col_width: int = 32,
    numeric_precision: int = 2,
) -> StyledDataFrame:
    """Return ``df`` with a styled terminal repr."""
    result = StyledDataFrame(df.copy())
    result._terminal_title = title
    result._terminal_subtitle = subtitle
    result._terminal_max_col_width = max_col_width
    result._terminal_numeric_precision = numeric_precision
    return result


def styled_series(
    series: pd.Series,
    title: str | None = None,
    *,
    max_col_width: int = 32,
    numeric_precision: int = 2,
) -> StyledSeries:
    """Return ``series`` with a styled terminal repr."""
    result = StyledSeries(series.copy())
    result._terminal_title = title
    result._terminal_max_col_width = max_col_width
    result._terminal_numeric_precision = numeric_precision
    return result


def blue_table(
    df: pd.DataFrame,
    *,
    index: bool = False,
    max_col_width: int = 32,
    numeric_precision: int = 2,
    red_columns: set[str] | None = None,
    reverse_numeric_columns: set[str] | None = None,
) -> str:
    """Render a compact ANSI-blue table for terminal reprs."""
    if df.empty:
        return f"{DIM}(empty){RESET}"

    red_columns = red_columns or set()
    reverse_numeric_columns = reverse_numeric_columns or set()
    display = df.copy()
    if index:
        display = display.reset_index()

    rows = [[_format_value(value, numeric_precision=numeric_precision) for value in row] for row in display.to_numpy()]
    headers = [str(column) for column in display.columns]
    widths = []
    for column_index, header in enumerate(headers):
        values = [row[column_index] for row in rows]
        width = min(max([len(header), *(len(value) for value in values)]), max_col_width)
        widths.append(width)

    header_line = f"{BOLD}{CYAN}" + "  ".join(
        f"{_clip(header, width):<{width}}"
        for header, width in zip(headers, widths, strict=True)
    ) + RESET
    rule = f"{NAVY}" + "  ".join("-" * width for width in widths) + RESET
    body = []
    for row_index, row in enumerate(rows):
        if _is_separator_row(display.iloc[row_index]):
            body.append("")
            continue
        if _is_aggregation_row(display.iloc[row_index]):
            body.append(f"{BLUE}{BOLD}" + "  ".join("=" * width for width in widths) + RESET)
        body.append(
            "  ".join(
                _styled_cell(
                    value,
                    width,
                    _is_numeric(display.iloc[row_index, column_index]),
                    highlight=_is_aggregation_row(display.iloc[row_index]),
                    force_red=headers[column_index] in red_columns and value != "-",
                    reverse_numeric_color=headers[column_index] in reverse_numeric_columns,
                )
                for column_index, (value, width) in enumerate(zip(row, widths, strict=True))
            )
        )
    return "\n".join([header_line, rule, *body])


def _styled_cell(
    value: str,
    width: int,
    is_numeric: bool,
    *,
    highlight: bool = False,
    force_red: bool = False,
    reverse_numeric_color: bool = False,
) -> str:
    clipped = _clip(value, width)
    aligned = f"{clipped:>{width}}" if is_numeric else f"{clipped:<{width}}"
    color = RED if force_red else _value_color(value, is_numeric=is_numeric, reverse_numeric=reverse_numeric_color)
    style = BOLD if highlight else ""
    return f"{style}{color}{aligned}{RESET}"


def _format_value(value: Any, *, numeric_precision: int) -> str:
    if value is None or pd.isna(value):
        return "-"
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if hasattr(value, "isoformat") and value.__class__.__name__ == "date":
        return value.isoformat()
    if isinstance(value, float):
        return f"{value:,.{numeric_precision}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _clip(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 1:
        return value[:width]
    return value[: width - 1] + "~"


def _is_numeric(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _value_color(value: str, *, is_numeric: bool, reverse_numeric: bool = False) -> str:
    if value in {"BUY", "ok"}:
        return GREEN
    if value == "SELL" or value.startswith("missing"):
        return RED
    if value == "-":
        return DIM
    if is_numeric:
        number = _numeric_value(value)
        if reverse_numeric:
            if number > 0:
                return RED
            if number < 0:
                return GREEN
            return WHITE
        if value.startswith("-"):
            return RED
        if number > 0:
            return GREEN
        return WHITE
    return WHITE


def _numeric_value(value: str) -> float:
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return 0.0


def _is_aggregation_row(row: pd.Series) -> bool:
    values = [str(value) for value in row.tolist() if value is not None and not pd.isna(value)]
    labels = {value.strip().lower() for value in values}
    return any(
        label in labels
        for label in {
            "portfolio weight",
            "portfolio net inflow",
            "portfolio valuation",
            "portfolio return",
            "portfolio volatility",
            "portfolio sharpe",
            "total",
        }
    )


def _is_separator_row(row: pd.Series) -> bool:
    label = row.iloc[0] if len(row) else None
    values = row.iloc[1:] if len(row) > 1 else pd.Series(dtype=object)
    return str(label).strip() == "" and values.isna().all()
