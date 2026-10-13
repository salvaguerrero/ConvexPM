import pandas as pd

from convexpm.utils.terminal_table import BLUE, BOLD, GREEN, RED, WHITE, blue_table, styled_dataframe


def test_blue_table_colors_positive_negative_and_aggregation_separator():
    df = pd.DataFrame(
        {
            "asset": ["Asset A", "Portfolio Return"],
            "return": [12.34, -5.67],
        }
    )

    rendered = blue_table(df)

    assert f"{GREEN} 12.34" in rendered
    assert f"{RED} -5.67" in rendered
    assert f"{BLUE}{BOLD}================" in rendered
    assert f"{BOLD}{RED}" in rendered


def test_blue_table_separates_portfolio_cash_and_valuation_totals():
    df = pd.DataFrame(
        {
            "asset": ["Asset A", "Portfolio Net Inflow", "Portfolio Valuation"],
            "Total": [100.0, 100.0, 120.0],
        }
    )

    rendered = blue_table(df)

    assert rendered.count(f"{BLUE}{BOLD}") == 2
    assert f"{BOLD}{WHITE}Portfolio Net Inflow" in rendered
    assert f"{BOLD}{WHITE}Portfolio Valuation " in rendered


def test_blue_table_can_force_positive_column_red():
    df = pd.DataFrame({"fees": [1.25], "amount": [1.25]})

    rendered = blue_table(df, red_columns={"fees"})

    assert f"{RED}1.25" in rendered
    assert f"{GREEN}  1.25" in rendered


def test_blue_table_can_reverse_numeric_column_colors():
    df = pd.DataFrame({"fees": [1.25, 0.0, -0.50]})

    rendered = blue_table(df, reverse_numeric_columns={"fees"})

    assert f"{RED} 1.25" in rendered
    assert f"\033[38;5;255m 0.00" in rendered
    assert f"{GREEN}-0.50" in rendered


def test_styled_dataframe_can_use_compact_column_widths():
    df = pd.DataFrame({"Very Long Header": [1.23]}, index=["Very Long Row Label"])

    rendered = repr(styled_dataframe(df, "COMPACT", max_col_width=8))

    assert "Very Lo~" in rendered
    assert "Very Long Header" not in rendered
