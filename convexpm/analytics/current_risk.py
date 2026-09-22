"""Risk of the current allocation using historical instrument returns."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from convexpm.analytics.asset_returns import instrument_returns
from convexpm.portfolio.allocation import weights as allocation_weights
from convexpm.utils.dates import DateLike


@dataclass(slots=True)
class CurrentRiskResult:
    """Structured current-allocation risk result."""

    metrics: pd.Series
    correlation: pd.DataFrame
    covariance: pd.DataFrame
    weights: pd.Series
    risk_contribution: pd.DataFrame
    synthetic_returns: pd.Series
    metadata: dict[str, object]

    def __getitem__(self, key: str) -> object:
        return self.metrics[key]

    def to_dict(self) -> dict[str, object]:
        return self.metrics.to_dict()


def current_risk(
    portfolio: object,
    *,
    start: DateLike | None = None,
    end: DateLike | None = None,
    benchmark: str | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> CurrentRiskResult:
    """Estimate risk for the allocation held at the analysis date."""
    weights_date = end or portfolio._default_date()
    current_weights = allocation_weights(portfolio, weights_date)
    if current_weights.empty:
        raise ValueError("Cannot calculate current risk for an empty portfolio")

    returns = instrument_returns(
        portfolio.market_data,
        current_weights.index,
        start=start,
        end=end,
    )
    if len(returns) < 2:
        raise ValueError("At least two aligned return observations are required")

    current_weights = current_weights.reindex(returns.columns).fillna(0.0)
    weight_sum = float(current_weights.sum())
    if abs(weight_sum) <= 1e-12:
        raise ValueError("Portfolio weights sum to zero")
    current_weights = current_weights / weight_sum

    correlation = returns.corr()
    covariance = returns.cov()
    synthetic = returns.mul(current_weights, axis=1).sum(axis=1)
    synthetic.name = "portfolio_return"

    annualized_volatility = float(synthetic.std(ddof=1) * np.sqrt(periods_per_year))
    target = risk_free_rate / periods_per_year
    downside = np.minimum(synthetic - target, 0.0)
    downside_volatility = float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(periods_per_year))
    var_95 = float(synthetic.quantile(0.05))
    tail = synthetic.loc[synthetic <= var_95]
    cvar_95 = float(tail.mean()) if not tail.empty else var_95

    metrics: dict[str, float] = {
        "annualized_volatility": annualized_volatility,
        "downside_volatility": downside_volatility,
        "historical_var_95": var_95,
        "historical_cvar_95": cvar_95,
    }

    if benchmark is not None:
        benchmark_returns = instrument_returns(
            portfolio.market_data,
            [benchmark],
            start=start,
            end=end,
        )[benchmark]
        aligned = pd.concat(
            [synthetic.rename("portfolio"), benchmark_returns.rename("benchmark")],
            axis=1,
        ).dropna()
        if len(aligned) >= 2:
            benchmark_variance = aligned["benchmark"].var(ddof=1)
            metrics["beta"] = (
                0.0
                if benchmark_variance == 0 or np.isnan(benchmark_variance)
                else float(aligned["portfolio"].cov(aligned["benchmark"]) / benchmark_variance)
            )
            corr = aligned["portfolio"].corr(aligned["benchmark"])
            metrics["benchmark_correlation"] = 0.0 if np.isnan(corr) else float(corr)
            metrics["tracking_error"] = float(
                (aligned["portfolio"] - aligned["benchmark"]).std(ddof=1)
                * np.sqrt(periods_per_year)
            )
        else:
            metrics["beta"] = 0.0
            metrics["benchmark_correlation"] = 0.0
            metrics["tracking_error"] = 0.0

    contribution = risk_contribution_from_covariance(
        current_weights,
        covariance,
        periods_per_year=periods_per_year,
    )
    metadata = {
        "analysis_start": returns.index.min().date(),
        "analysis_end": returns.index.max().date(),
        "frequency": "daily",
        "annualization_factor": periods_per_year,
        "observations": len(returns),
        "weights_date": pd.Timestamp(weights_date).date(),
        "benchmark": benchmark,
        "missing_data_policy": "intersection",
    }

    return CurrentRiskResult(
        metrics=pd.Series(metrics, dtype=float),
        correlation=correlation,
        covariance=covariance,
        weights=current_weights,
        risk_contribution=contribution,
        synthetic_returns=synthetic,
        metadata=metadata,
    )


def risk_contribution_from_covariance(
    weights: pd.Series,
    covariance: pd.DataFrame,
    *,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Return marginal, component, and percentage contribution to volatility."""
    labels = list(covariance.columns)
    w = weights.reindex(labels).fillna(0.0).to_numpy(dtype=float)
    annual_cov = covariance.loc[labels, labels].to_numpy(dtype=float) * periods_per_year
    variance = float(w @ annual_cov @ w)
    volatility = float(np.sqrt(max(variance, 0.0)))

    if volatility <= 1e-15:
        marginal = np.zeros_like(w)
        component = np.zeros_like(w)
        percentage = np.zeros_like(w)
    else:
        marginal = annual_cov @ w / volatility
        component = w * marginal
        percentage = component / volatility

    return pd.DataFrame(
        {
            "weight": w,
            "marginal_contribution": marginal,
            "component_contribution": component,
            "risk_contribution": percentage,
        },
        index=pd.Index(labels, name="instrument_id"),
    )
