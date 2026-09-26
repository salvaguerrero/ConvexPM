"""Small terminal-native plots for REPL workflows."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TerminalPlot:
    """A plot object that renders as text in the Python REPL."""

    text: str

    def __repr__(self) -> str:
        return self.text

    def __str__(self) -> str:
        return self.text


def terminal_line_plot(
    series: pd.Series,
    *,
    title: str,
    width: int = 80,
    height: int = 16,
) -> TerminalPlot:
    """Return a plotext terminal chart wrapped for REPL display."""
    clean = series.dropna()
    if clean.empty:
        return TerminalPlot(_panel(title, ["no data"], width=max(32, width)))

    clean = clean.sort_index()
    width = max(48, int(width))
    height = max(6, int(height))
    try:
        return TerminalPlot(_plotext_line_plot(clean, title=title, width=width, height=height))
    except Exception:
        return TerminalPlot(_fallback_line_plot(clean, title=title, width=width, height=height))


def _plotext_line_plot(series: pd.Series, *, title: str, width: int, height: int) -> str:
    import plotext as plt

    values = series.astype(float)
    x_values = list(range(len(values)))
    y_values = values.tolist()
    fig = plt.figure
    fig.clear()
    fig.plot_size(width, height)
    fig.theme("dark")
    fig.title(title)
    fig.ruler("x").ticks(
        [0, max(len(values) - 1, 0)],
        [_format_index(series.index[0]), _format_index(series.index[-1])],
    )
    fig.draw(fig.signal(x_values, y_values).lines())
    chart = str(fig.build()).rstrip()
    stats = _stats_line(values)
    return "\n".join([chart, stats])


def _fallback_line_plot(series: pd.Series, *, title: str, width: int, height: int) -> str:
    label_width = 11
    plot_width = max(16, width - label_width - 5)
    sampled = _sample_series(series, plot_width)
    values = sampled.astype(float)
    minimum = float(values.min())
    maximum = float(values.max())

    grid = [[" " for _ in range(plot_width)] for _ in range(height)]
    if maximum == minimum:
        rows = [height // 2 for _ in values]
    else:
        rows = [
            height - 1 - round((float(value) - minimum) / (maximum - minimum) * (height - 1))
            for value in values
        ]

    for column, row in enumerate(rows):
        grid[row][column] = "●"
        if column > 0:
            previous = rows[column - 1]
            if previous == row:
                grid[row][column - 1] = "─"
                grid[row][column] = "●"
            else:
                step = 1 if row > previous else -1
                for bridge_row in range(previous + step, row, step):
                    grid[bridge_row][column] = "│"
                grid[previous][column] = "╮" if row > previous else "╯"
                grid[row][column] = "╰" if row > previous else "╭"

    y_labels = _axis_labels(minimum, maximum, height)
    body_lines = [
        f"{label:>{label_width}} ┤{''.join(row)}"
        for label, row in zip(y_labels, grid, strict=True)
    ]
    first_label = _format_index(sampled.index[0])
    last_label = _format_index(sampled.index[-1])
    date_padding = max(1, plot_width - len(first_label) - len(last_label))
    footer = f"{' ' * (label_width + 2)}{first_label}{' ' * date_padding}{last_label}"
    stats = _stats_line(values)
    return _panel(title, [stats, "", *body_lines, footer], width=width)


def _stats_line(values: pd.Series) -> str:
    minimum = float(values.min())
    maximum = float(values.max())
    first = float(values.iloc[0])
    latest = float(values.iloc[-1])
    total_return = (latest / first - 1) if first else None
    return "   ".join(
        [
            f"Last {_format_number(latest)}",
            f"Min {_format_number(minimum)}",
            f"Max {_format_number(maximum)}",
            f"Return {_format_percent(total_return)}",
        ]
    )


def _panel(title: str, lines: list[str], *, width: int) -> str:
    inner_width = max(width - 2, len(title) + 2)
    clean_title = f" {title.strip()} "
    if len(clean_title) > inner_width:
        clean_title = clean_title[:inner_width]
    top = "┌" + clean_title + "─" * (inner_width - len(clean_title)) + "┐"
    bottom = "└" + "─" * inner_width + "┘"
    body = [f"│{line[:inner_width]:<{inner_width}}│" for line in lines]
    return "\n".join([top, *body, bottom])


def _axis_labels(minimum: float, maximum: float, height: int) -> list[str]:
    if height <= 1:
        return [_format_number(maximum)]
    if maximum == minimum:
        labels = [""] * height
        labels[height // 2] = _format_number(maximum)
        return labels
    return [
        _format_number(maximum - (maximum - minimum) * row / (height - 1))
        for row in range(height)
    ]


def _format_number(value: float) -> str:
    absolute = abs(value)
    if absolute >= 1000:
        return f"{value:,.0f}"
    if absolute >= 100:
        return f"{value:,.2f}"
    if absolute >= 1:
        return f"{value:,.4f}"
    return f"{value:,.6f}"


def _format_percent(value: float | None) -> str:
    if value is None:
        return "n.a."
    return f"{value:+.2%}"


def _sample_series(series: pd.Series, width: int) -> pd.Series:
    if len(series) <= width:
        return series
    positions = pd.Index([round(item) for item in _linspace(0, len(series) - 1, width)])
    return series.iloc[positions].drop_duplicates()


def _linspace(start: int, stop: int, count: int) -> list[float]:
    if count <= 1:
        return [float(start)]
    step = (stop - start) / (count - 1)
    return [start + step * item for item in range(count)]


def _format_index(value: object) -> str:
    if hasattr(value, "date"):
        return str(value.date())
    return str(value)
