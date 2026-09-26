"""ConvexPM REPL scratchpad.

Run this file line by line in a Python REPL / interactive editor. Expressions are
left bare intentionally so each result is displayed as you evaluate it.
"""

from convexpm import Portfolio


# Load the saved personal portfolio.
portfolio = Portfolio.load("personal")
portfolio


# -----------------------------------------------------------------------------
# PERFORMANCE VIEW
# -----------------------------------------------------------------------------

performance = portfolio.performance_view()

# Portfolio investment return. This is based on historical holdings and ignores
# the mechanical effect of adding/removing capital.
performance.summary

# One row per asset: current weight, asset return, and contribution to the
# portfolio return over the selected period.
performance.assets

# One row per asset and one column per calendar year. The final row is the total
# portfolio return for each year.
performance.yearly_return_contribution

# Series used by the main cumulative-return chart.
performance.cumulative_return

# Example: focus performance on a custom period.
performance_2025 = portfolio.performance_view(start="2025-01-01")
performance_2025.summary
performance_2025.yearly_return_contribution


# -----------------------------------------------------------------------------
# RISK VIEW
# -----------------------------------------------------------------------------

# Risk lookback is independent from the performance period. Change 2020 to the
# history you want to use for covariance / volatility estimates.
risk = portfolio.risk_view(start="2020-01-01")

# Current-allocation annualized volatility and historical max drawdown of that
# current allocation over the selected risk lookback.
risk.summary

# One row per current asset: weight, standalone volatility, and contribution to
# total portfolio volatility.
risk.assets

# One row per asset and one column per year. Each annual column uses the
# allocation held at that year-end. The final row is total portfolio volatility.
risk.yearly_risk_contribution

# Same total-risk evolution without the attribution rows.
risk.yearly_portfolio_volatility

# Correlation matrix over the selected risk lookback.
risk.correlation

# Calculation dates / conventions used by the view.
risk.metadata
