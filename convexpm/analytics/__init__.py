"""Portfolio analytics."""

from convexpm.analytics.asset_returns import instrument_returns
from convexpm.analytics.current_risk import CurrentRiskResult, current_risk, risk_contribution_from_covariance
from convexpm.analytics.exposure import exposure_by
from convexpm.analytics.performance import performance_summary
from convexpm.analytics.portfolio_analysis import PortfolioAnalysisResult, analyze_portfolio, return_attribution
from convexpm.analytics.views import PerformanceViewResult, RiskViewResult, performance_view, risk_view
from convexpm.analytics.returns import (
    cagr,
    monthly_returns,
    nav_returns,
    rolling_returns,
    total_return,
    xirr,
)
from convexpm.analytics.risk import (
    annualized_volatility,
    beta,
    correlation,
    current_drawdown,
    downside_deviation,
    historical_cvar,
    historical_var,
    max_drawdown,
    risk_summary,
    sharpe_ratio,
    sortino_ratio,
)

__all__ = [
    "PerformanceViewResult",
    "RiskViewResult",
    "performance_view",
    "risk_view",
    "CurrentRiskResult",
    "PortfolioAnalysisResult",
    "analyze_portfolio",
    "current_risk",
    "instrument_returns",
    "return_attribution",
    "risk_contribution_from_covariance",
    "annualized_volatility",
    "beta",
    "cagr",
    "correlation",
    "current_drawdown",
    "downside_deviation",
    "exposure_by",
    "historical_cvar",
    "historical_var",
    "max_drawdown",
    "monthly_returns",
    "nav_returns",
    "performance_summary",
    "risk_summary",
    "rolling_returns",
    "sharpe_ratio",
    "sortino_ratio",
    "total_return",
    "xirr",
]
