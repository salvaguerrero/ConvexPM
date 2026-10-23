# ConvexPM

ConvexPM is a small Python toolkit for tracking and analyzing an investment portfolio.

The basic idea is simple:

```text
Instruments -> Trades -> Holdings -> Market Data -> NAV History -> Analytics -> Scenarios
```

You enter instruments and trades. ConvexPM derives holdings from those trades, values them with historical market data, and then calculates returns, risk, exposure, and what-if scenarios.

## Install

From this project folder:

```bash
pip install -e .
```

This installs the local analytics package plus its charting dependencies,
including `lightweight-charts` for interactive TradingView-style charts.

For development and tests:

```bash
pip install -e ".[dev]"
pytest
```

## Demo Notebook

Open [scripts/explore_my_portfolio.ipynb](scripts/explore_my_portfolio.ipynb) for a
demo portfolio with invented trades and Yahoo price history for real stocks,
with saved outputs covering cash settlement, EUR/USD
valuation, benchmark performance, attribution, correlations, and charts. It includes
a reference cell for the APIs used in `scripts/explore_my_portfolio_2.py`.

```bash
pip install jupyterlab ipykernel
jupyter lab scripts/explore_my_portfolio.ipynb
```

Select a kernel with ConvexPM installed and run all cells. The default demo uses
Yahoo downloads and isolated temporary stores, so Run All requires internet access.
Existing data files are never used. Tables use terminal formatting via `print()`;
the dedicated ConvexPM charts render inline in a trusted Jupyter notebook.
Broker imports are opt-in.

## Main Concepts

### `Instrument`

An instrument describes something you can hold, such as a stock, ETF, or fund.

```python
from convexpm import Instrument

repsol = Instrument(
    instrument_id="REP_MC",
    name="Repsol",
    instrument_type="equity",
    asset_class="equity",
    currency="EUR",
    data_source="yahoo",
    data_symbol="REP.MC",
)
```

For mutual funds, use the ISIN as the `instrument_id` when possible:

```python
fund = Instrument(
    instrument_id="ES0173311103",
    name="R4 Multigestion Numantia Patr.global",
    instrument_type="fund",
    asset_class="fund",
    currency="EUR",
    data_source="renta4",
)
```

### `InstrumentRegistry`

The registry stores instruments locally in `data/instruments.parquet`.

```python
from convexpm import InstrumentRegistry

registry = InstrumentRegistry("data/instruments.parquet")
registry.add(repsol)
registry.save()

same_registry = InstrumentRegistry("data/instruments.parquet")
print(same_registry.get("REP_MC"))
print(same_registry.all())
```

### `Trade`

A trade records a BUY or SELL. Holdings are calculated from trades; you do not manually maintain positions.

```python
from datetime import date
from convexpm import Trade

buy_repsol = Trade(
    instrument_id="REP_MC",
    date=date(2026, 1, 2),
    side="BUY",
    quantity=100,
    amount=1500.0,
    value_per_unit=15.0,
    currency="EUR",
)
```

You can also create amount-based fund trades:

```python
fund_buy = Trade(
    instrument_id="ES0173311103",
    date=date(2026, 1, 2),
    side="BUY",
    amount=5000.0,
    currency="EUR",
)
```

If `quantity` is missing, ConvexPM can derive it later when it has a `value_per_unit` or market data for the trade date.

When you add trades, you can optionally generate the matching cash settlement
leg at the same time:

```python
portfolio.add_trades([buy_repsol], cash_ledger=True)
portfolio.save()
```

Cash instruments such as `EUR_CASH` are priced at `1.0` and are treated as
cash positions. Generated settlement legs are internal portfolio activity, so
they are not double-counted as external contributions in performance views.

### Saving And Loading A Portfolio

Trades belong to a portfolio. ConvexPM stores each portfolio in its own local folder:

```text
data/
└── portfolios/
    └── personal/
        ├── portfolio.json
        └── trades.parquet
```

`portfolio.json` stores portfolio metadata such as the name, base currency,
and optional benchmark.
`trades.parquet` is the source-of-truth transaction ledger.

```python
from convexpm import Portfolio

portfolio = Portfolio(
    name="Personal Portfolio",
    base_currency="EUR",
    trades=[buy_repsol],
    registry=registry,
    market_data=store,
)

portfolio.save("personal")
```

Later, even after restarting Python:

```python
portfolio = Portfolio.load("personal")
print(portfolio.trades)
print(portfolio.holdings())
```

A loaded or previously saved portfolio remembers its folder. After adding a trade,
you can save it again without repeating the portfolio ID:

```python
portfolio.add_trade(new_trade)
portfolio.save()
```

You can also store a default benchmark once:

```python
portfolio.set_benchmark("IE0002639668")
portfolio.save()
```

`performance_view()` will use the stored benchmark unless you override it for a
single call.

The instrument registry (`data/instruments.parquet`) and market-data store
(`data/market_data.parquet`) remain shared across portfolios, so historical prices
are not duplicated.

The lower-level `save_trades()` and `load_trades()` helpers are still available
when direct trade-file access is useful.

## Market Data

All market data uses one normalized schema:

```text
date | instrument_id | value | currency | value_type | source
```

`value` means the value of one unit of the instrument on that date. For a stock this is usually a close price. For a mutual fund this is NAV.

### `MarketDataStore`

The market data store reads and writes `data/market_data.parquet`.

```python
from datetime import date

import pandas as pd
from convexpm import MarketDataStore

store = MarketDataStore("data/market_data.parquet")

store.upsert(
    pd.DataFrame(
        {
            "date": ["2026-01-02", "2026-01-03"],
            "instrument_id": ["REP_MC", "REP_MC"],
            "value": [15.0, 15.5],
            "currency": ["EUR", "EUR"],
            "value_type": ["close", "close"],
            "source": ["manual", "manual"],
        }
    )
)

print(store.get_history("REP_MC"))
print(store.get_value("REP_MC", date(2026, 1, 3)))
print(store.latest("REP_MC"))
```

`upsert()` deduplicates by `instrument_id + date`, so importing the same data twice is safe.

### Yahoo Finance

Use Yahoo for instruments where `data_source="yahoo"` and `data_symbol` is a Yahoo ticker.

```python
from convexpm.market_data import MarketDataUpdater, YahooProvider

updater = MarketDataUpdater(
    registry=registry,
    store=store,
    providers={"yahoo": YahooProvider()},
)

updater.update_instrument("REP_MC")
```

ConvexPM uses Yahoo daily close values by default. If you already have data stored through a date, the updater starts from the following day.

### Renta 4 CSV

Renta 4 CSV exports can contain one or many funds:

```text
;ES0173311103 - R4 Multigestion Numantia Patr.global

;12/06/2015;10;
;13/06/2015;10,25;
```

Import like this:

```python
from convexpm.market_data import Renta4CSVProvider

provider = Renta4CSVProvider()
market_df, discovered_instruments = provider.parse("renta4_export.csv")

registry.add_many(discovered_instruments)
registry.save()

store.upsert(market_df)
```

The parser assumes EUR for Renta 4 funds.

## Broker Imports

IBKR transaction-history CSV exports can be imported into normalized trades plus
the full cash-transaction ledger:

```python
from convexpm import IBKRTransactionParser

ibkr_import = IBKRTransactionParser().parse("data/ibkr/U24532596.TRANSACTIONS.1Y.csv")
trades = ibkr_import.trades
cash_transactions = ibkr_import.cash_transactions
```

For trades, `amount` is gross exposure in the statement base currency and
`fees` includes broker commissions plus explicit transaction-fee rows. The
original broker fields remain in `trade.metadata`.

Renta 4 folders can include fund NAV CSVs, a fund-operation `.xls`, and
Morningstar report PDFs:

```python
from convexpm import Renta4ImportParser

r4_import = Renta4ImportParser().parse_directory("data/r4")

registry.add_many(r4_import.instruments)
registry.save()
store.upsert(r4_import.market_data)
```

The Renta 4 importer maps fund operations to ISINs when the fund appears in a
NAV export. If a fund has transactions but no NAV file, it receives a generated
`R4_...` instrument ID.

Renta 4 `extractosPatrimonio*.xls` files are also read when they include the
`Saldo en Divisa` sheet. Only account transfers, custody fees, IVA, and
withholding rows become `EUR_CASH` trades from that sheet. Fund purchase and
sale cash rows in `Saldo en Divisa` are ignored because fund subscriptions and
redemptions come from the fund-operation `.xls` files; add those Renta 4 trades
with `cash_ledger=True` to generate the matching cash settlement legs.

## Build A Portfolio

A portfolio combines trades, instruments, and market data.

```python
from datetime import date

from convexpm import Portfolio

portfolio = Portfolio(
    name="Main",
    base_currency="EUR",
    trades=trades,
    registry=registry,
    market_data=store,
)
```

### Holdings

Holdings are derived from trades.

```python
print(portfolio.holdings())
print(portfolio.holdings(date(2026, 1, 3)))
```

Example:

```text
01 Jan BUY  10 REP_MC
03 Jan BUY   5 REP_MC
06 Jan SELL  3 REP_MC
```

Holding after 06 Jan:

```text
REP_MC = 12
```

### NAV

Current NAV:

```python
print(portfolio.nav())
```

NAV on a specific date:

```python
print(portfolio.nav(date(2026, 1, 3)))
```

Daily NAV history:

```python
nav = portfolio.nav_history()
print(nav)
```

Valuation is:

```text
NAV = sum(quantity * value converted to portfolio base currency)
```

FX conversion uses saved FX market-data instruments such as `USDEUR` or
`EURUSD` when an instrument currency differs from the portfolio base currency.

## Analytics

ConvexPM now has two analytics layers:

- Low-level analytics such as `performance()`, `risk()`, `current_risk()`, and
  `analyze()` remain available for programmatic use.
- The higher-level `performance_view()` is the preferred interactive workflow.
  It returns styled terminal tables with yearly columns and a `Total` column.

### Performance View

```python
performance = portfolio.performance_view(start="2020-01-01")
```

The summary table answers "how did my portfolio perform?" by year and in total:

```python
performance.summary
```

It includes:

- `portfolio_return (%)`
- `cagr (%)`
- `volatility (%)`
- `sharpe`
- `max_drawdown (%)`
- `benchmark_return (%)`
- `alpha (%)`
- `beta`
- `capital_inflows`
- `capital_outflows`
- `net_capital_flow`
- `start_nav`
- `end_nav`
- `delta_nav`
- `capital_gain`

Set a benchmark once on the portfolio:

```python
portfolio.set_benchmark("IE0002639668")
portfolio.save()
performance = portfolio.performance_view()
```

Or override it for one analysis:

```python
performance = portfolio.performance_view(benchmark="AAPL")
```

Benchmark lookup accepts `instrument_id`, ticker/data symbol, or exact
instrument name.

Standalone asset metrics ignore portfolio weights and answer how each asset
behaved while it was held:

```python
performance.standalone_asset_returns
performance.standalone_asset_volatility
performance.standalone_asset_alpha
performance.standalone_asset_beta
performance.standalone_asset_drawdown
```

Contribution tables include the portfolio-weighted effect of each asset:

```python
performance.asset_weights
performance.return_contribution
performance.volatility_contribution
performance.drawdown_contribution
```

`performance.return_contribution` uses assets as rows and years as columns.
Asset rows are chain-linked percentage-point return contributions, and the
final `Portfolio Return` row is the total investment return for each period.

Correlation helpers return compact, terminal-friendly matrices:

```python
performance.return_correlation()
performance.return_correlation(year=2026)
performance.return_correlation(start="2020-01-01", end="2026-09-25")

performance.volatility_correlation()
```

Long fund names are shortened in correlation matrices, preferring ticker/data
symbols when available, so the matrix remains readable in a terminal.

### Current Risk View

`risk_view()` is still available for current-allocation risk analysis:

```python
risk = portfolio.risk_view(start="2020-01-01")
risk.summary
risk.assets
risk.asset_volatility
risk.volatility_contribution
risk.asset_sharpe
risk.sharpe_contribution
risk.yearly_risk_contribution
risk.yearly_portfolio_volatility
risk.correlation
```

Risk view uses the current or year-end allocation and historical instrument
returns. This is different from realized performance, which uses the actual
historical holdings through time.

### Full Analysis

For a broader object that combines realized performance, current risk,
attribution, covariance/correlation matrices, and metadata:

```python
analysis = portfolio.analyze(
    start="2020-01-01",
    end="2026-09-19",
    benchmark="SP500",
)

analysis.portfolio
analysis.instruments
analysis.return_attribution
analysis.risk_contribution
analysis.correlation
analysis.covariance
analysis.weights
analysis.metadata
```

### Exposure

Group exposure by asset class:

```python
print(portfolio.exposure(by="asset_class"))
```

Group exposure by currency:

```python
print(portfolio.exposure(by="currency"))
```

You can also group by metadata keys you add to instruments, such as `country` or `sector`.

## Interactive Charts

ConvexPM includes lightweight TradingView-style local charts through
`lightweight-charts`.

### Instrument Chart

Open an empty chart and add instruments by ID, name, ticker, or unique name
fragment:

```python
from convexpm import instrument_plot

chart = instrument_plot(start="2026-01-01", curr="EUR")
chart.add_instrument("REP")
chart.add_instrument("Chevron")
chart.relative_chart()
chart.absolute_chart()
chart.remove_instrument("REP")
```

Use `curr="original"` when adding an instrument to keep its native currency:

```python
chart.add_instrument("Chevron", curr="original")
```

### Portfolio Chart

Open an interactive portfolio chart:

```python
portfolio_chart = portfolio.chart(horizon=90, drawdown_tolerance=0.15)
```

To chart a fund-style NAV per unit, where deposits and withdrawals issue or
redeem units instead of moving the displayed price, use:

```python
portfolio_chart = portfolio.chart(horizon=90, drawdown_tolerance=0.15, fund_nav=True)
```

The chart includes:

- NAV as the main line, or fund-style NAV per unit when `fund_nav=True`.
- Cumulative net contributions for regular NAV charts.
- `+/-1 sigma` and `+/-2 sigma` rolling volatility envelope.
- Drawdown tolerance floor from the NAV high-water mark.

The `horizon` argument sets the rolling volatility window in observations.
Common values are `30`, `90`, and `365`. The chart is configured at creation
time; create a new chart if you want a different horizon or drawdown tolerance.

For speed, portfolio charts value the portfolio only on actual market/trade
observation dates instead of every calendar day.

Use `instrument_plot()` for instrument charts and `portfolio.chart()` for
portfolio charts. Use `portfolio.performance_view()` for portfolio metrics.

## Terminal Display

Portfolio, trade-ledger, performance, risk, and chart-supporting tables use a
Bloomberg-like blue terminal style in the REPL. Positive numbers render green,
negative numbers render red, and aggregation rows such as `Portfolio Return`,
`Portfolio Volatility`, `Portfolio Drawdown`, and `Total` receive a highlighted
separator.

This styling is generic to ConvexPM's styled DataFrame/Series wrappers; it does
not globally override pandas display.

## What-if Scenarios

Scenarios add hypothetical trades to a cloned portfolio. The original portfolio is not changed.

### Current What-if

```python
from datetime import date

scenario = portfolio.what_if(
    Trade(
        instrument_id="REP_MC",
        date=date.today(),
        side="BUY",
        quantity=10,
    )
)

print(scenario.portfolio.holdings())
print(scenario.portfolio.nav())
print(scenario.portfolio.exposure(by="asset_class"))
```

If the hypothetical trade is dated after the latest stored market value, ConvexPM uses the latest available value. It does not forecast prices.

### Historical What-if

Ask questions like: "What if I had bought EUR 10k of this fund last year?"

```python
scenario = portfolio.what_if(
    Trade(
        instrument_id="ES0173311103",
        date=date(2025, 9, 18),
        side="BUY",
        amount=10_000,
    )
)

print(scenario.portfolio.nav_history())
print(scenario.portfolio.performance_view().summary)
```

### Compare A Scenario

```python
comparison = scenario.compare()
print(comparison)
```

The comparison table includes current and scenario values for NAV, return, volatility, drawdown, Sharpe, Sortino, VaR, and CVaR.

## Complete Tiny Example

This example uses manual market data so it can run without internet access.

```python
from datetime import date

import pandas as pd

from convexpm import Instrument, InstrumentRegistry, MarketDataStore, Portfolio, Trade

registry = InstrumentRegistry("data/instruments.parquet", auto_load=False)
registry.add(
    Instrument(
        instrument_id="REP_MC",
        name="Repsol",
        instrument_type="equity",
        asset_class="equity",
        currency="EUR",
        data_source="manual",
    )
)

store = MarketDataStore("data/market_data.parquet")
store.upsert(
    pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=3, freq="D"),
            "instrument_id": "REP_MC",
            "value": [10.0, 11.0, 12.0],
            "currency": "EUR",
            "value_type": "close",
            "source": "manual",
        }
    )
)

trades = [
    Trade(
        instrument_id="REP_MC",
        date=date(2026, 1, 1),
        side="BUY",
        quantity=10,
        amount=100,
        value_per_unit=10,
        currency="EUR",
    )
]

portfolio = Portfolio(
    name="Main",
    base_currency="EUR",
    trades=trades,
    registry=registry,
    market_data=store,
)

print(portfolio.holdings())
print(portfolio.nav_history())
performance = portfolio.performance_view()
print(performance.summary)
print(performance.return_contribution)
print(portfolio.exposure(by="asset_class"))
```

## Project Layout

```text
convexpm/
    instruments/     Instrument and InstrumentRegistry
    transactions/    Trade model and trade persistence
    market_data/     MarketDataStore plus Yahoo and Renta 4 providers
    portfolio/       Holdings and valuation
    analytics/       Returns, performance, risk, and exposure
    plotting.py       Interactive instrument and portfolio charts
    scenarios/       What-if scenario cloning and comparison
    utils/           Date and terminal-display helpers
tests/               Deterministic unit tests
examples/            Small usage scripts
data/                Local Parquet data files
```

## Current Limitations

- Only BUY and SELL trades are supported.
- Holdings are quantity-based.
- Valuation is market-value based and does not include tax lots or realized-gain accounting.
- FX conversion depends on saved FX market-data rows; missing FX data can still limit valuation.
- Generated cash settlement legs are supported, but there is no full broker reconciliation system yet.
- There is no tax, lot accounting, or full reconciliation system.
- There are no options Greeks, optimization, Monte Carlo, web UI, or live market data.
- TradingView-style charts are local Python windows, not a hosted web dashboard.

## Roadmap

Likely V2 improvements:

- Separate valuation models for options, futures, bonds, and FX forwards.
- Risk factors and scenario repricing.
- Broker reconciliation and richer cash accounting.
- Richer metadata for sector, country, duration, strategy, and account-level reporting.
