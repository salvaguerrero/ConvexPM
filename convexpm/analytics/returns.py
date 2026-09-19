"""Return calculations for NAV histories."""

from __future__ import annotations

from datetime import date
from math import isfinite
from typing import Iterable

import numpy as np
import pandas as pd

from convexpm.transactions import Trade


def clean_nav_history(nav_history: pd.DataFrame) -> pd.DataFrame:
    """Return sorted NAV history with normalized columns."""
    if nav_history.empty:
        return pd.DataFrame(columns=["date", "nav"])
    df = nav_history.loc[:, ["date", "nav"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna(subset=["date", "nav"]).sort_values("date").reset_index(drop=True)
    return df


def nav_returns(nav_history: pd.DataFrame) -> pd.Series:
    """Daily percentage returns from a NAV history."""
    df = clean_nav_history(nav_history)
    if len(df) < 2:
        return pd.Series(dtype=float, name="return")
    series = df.set_index("date")["nav"]
    returns = series.pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    returns.name = "return"
    return returns


def total_return(nav_history: pd.DataFrame) -> float:
    """Simple total return based on first and last non-zero NAV."""
    df = clean_nav_history(nav_history)
    df = df.loc[df["nav"] != 0]
    if len(df) < 2:
        return 0.0
    return float(df["nav"].iloc[-1] / df["nav"].iloc[0] - 1.0)


def cagr(nav_history: pd.DataFrame) -> float:
    """Compound annual growth rate based on first and last non-zero NAV."""
    df = clean_nav_history(nav_history)
    df = df.loc[df["nav"] > 0]
    if len(df) < 2:
        return 0.0
    years = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25
    if years <= 0:
        return 0.0
    return float((df["nav"].iloc[-1] / df["nav"].iloc[0]) ** (1.0 / years) - 1.0)


def ytd_return(nav_history: pd.DataFrame, as_of: date | None = None) -> float:
    """Year-to-date return using available NAV observations in the year."""
    df = clean_nav_history(nav_history)
    if df.empty:
        return 0.0
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else df["date"].max()
    year_df = df.loc[(df["date"].dt.year == as_of_ts.year) & (df["date"] <= as_of_ts)]
    return total_return(year_df)


def mtd_return(nav_history: pd.DataFrame, as_of: date | None = None) -> float:
    """Month-to-date return using available NAV observations in the month."""
    df = clean_nav_history(nav_history)
    if df.empty:
        return 0.0
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else df["date"].max()
    month_df = df.loc[
        (df["date"].dt.year == as_of_ts.year)
        & (df["date"].dt.month == as_of_ts.month)
        & (df["date"] <= as_of_ts)
    ]
    return total_return(month_df)


def monthly_returns(nav_history: pd.DataFrame) -> pd.Series:
    """Monthly returns based on month-end NAV observations."""
    df = clean_nav_history(nav_history)
    if len(df) < 2:
        return pd.Series(dtype=float, name="monthly_return")
    monthly_nav = df.set_index("date")["nav"].resample("ME").last().dropna()
    returns = monthly_nav.pct_change().dropna()
    returns.name = "monthly_return"
    return returns


def rolling_returns(nav_history: pd.DataFrame, window: int = 252) -> pd.Series:
    """Rolling return over ``window`` observations."""
    df = clean_nav_history(nav_history)
    if len(df) <= window:
        return pd.Series(dtype=float, name=f"rolling_{window}_return")
    nav = df.set_index("date")["nav"]
    returns = nav / nav.shift(window) - 1.0
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    returns.name = f"rolling_{window}_return"
    return returns


def time_weighted_return(nav_history: pd.DataFrame, trades: Iterable[Trade] | None = None) -> float:
    """Approximate TWR by adjusting daily returns for same-day external flows.

    BUY trades are treated as contributions and SELL trades as withdrawals.
    When trade amounts are unavailable, the flow for that trade is ignored.
    """
    df = clean_nav_history(nav_history)
    if len(df) < 2:
        return 0.0
    flows = _daily_external_flows(trades or [])
    nav = df.set_index("date")["nav"]
    factors = []
    for idx in range(1, len(nav)):
        current_date = nav.index[idx]
        beginning = nav.iloc[idx - 1]
        ending = nav.iloc[idx]
        if beginning == 0:
            continue
        flow = flows.get(current_date.normalize(), 0.0)
        factors.append((ending - flow) / beginning)
    if not factors:
        return 0.0
    return float(np.prod(factors) - 1.0)


def xirr(cash_flows: Iterable[tuple[date | pd.Timestamp, float]]) -> float | None:
    """Calculate money-weighted return for irregular cash flows.

    Returns ``None`` when the supplied flows do not contain both signs or when
    no robust bracket can be found.
    """
    flows = [(pd.Timestamp(flow_date).normalize(), float(amount)) for flow_date, amount in cash_flows]
    flows = [(flow_date, amount) for flow_date, amount in flows if amount != 0]
    if len(flows) < 2:
        return None
    amounts = [amount for _, amount in flows]
    if not (any(amount > 0 for amount in amounts) and any(amount < 0 for amount in amounts)):
        return None

    start = min(flow_date for flow_date, _ in flows)

    def xnpv(rate: float) -> float:
        if rate <= -1.0:
            return np.inf
        total = 0.0
        for flow_date, amount in flows:
            years = (flow_date - start).days / 365.25
            total += amount / ((1.0 + rate) ** years)
        return total

    low = -0.999999
    high = 1.0
    low_value = xnpv(low)
    high_value = xnpv(high)
    expansions = 0
    while _same_sign(low_value, high_value) and expansions < 80:
        high = high * 2.0 + 1.0
        high_value = xnpv(high)
        expansions += 1
        if not isfinite(high_value):
            break

    if _same_sign(low_value, high_value):
        return None

    for _ in range(200):
        mid = (low + high) / 2.0
        mid_value = xnpv(mid)
        if abs(mid_value) < 1e-10:
            return float(mid)
        if _same_sign(low_value, mid_value):
            low = mid
            low_value = mid_value
        else:
            high = mid
    return float((low + high) / 2.0)


def money_weighted_return(nav_history: pd.DataFrame, trades: Iterable[Trade]) -> float | None:
    """Portfolio XIRR from trade cash flows plus terminal NAV."""
    df = clean_nav_history(nav_history)
    if df.empty:
        return None
    flows: list[tuple[date | pd.Timestamp, float]] = []
    for trade in trades:
        amount = _trade_amount(trade)
        if amount is None:
            continue
        if trade.side == "BUY":
            flows.append((trade.date, -(amount + trade.fees)))
        else:
            flows.append((trade.date, amount - trade.fees))
    terminal_nav = float(df["nav"].iloc[-1])
    if terminal_nav != 0:
        flows.append((df["date"].iloc[-1], terminal_nav))
    return xirr(flows)


def _daily_external_flows(trades: Iterable[Trade]) -> dict[pd.Timestamp, float]:
    flows: dict[pd.Timestamp, float] = {}
    for trade in trades:
        amount = _trade_amount(trade)
        if amount is None:
            continue
        signed_flow = amount + trade.fees if trade.side == "BUY" else -(amount - trade.fees)
        timestamp = pd.Timestamp(trade.date).normalize()
        flows[timestamp] = flows.get(timestamp, 0.0) + signed_flow
    return flows


def _trade_amount(trade: Trade) -> float | None:
    if trade.amount is not None:
        return float(trade.amount)
    if trade.quantity is not None and trade.value_per_unit is not None:
        return float(trade.quantity) * float(trade.value_per_unit)
    return None


def _same_sign(left: float, right: float) -> bool:
    return (left >= 0 and right >= 0) or (left <= 0 and right <= 0)
