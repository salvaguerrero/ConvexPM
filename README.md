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

For development and tests:

```bash
pip install -e ".[dev]"
pytest
```

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

### Saving And Loading A Portfolio

Trades belong to a portfolio. ConvexPM stores each portfolio in its own local folder:

```text
data/
└── portfolios/
    └── personal/
        ├── portfolio.json
        └── trades.parquet
```

`portfolio.json` stores portfolio metadata such as the name and base currency.
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

The parser assumes EUR for Renta 4 funds in V1.

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

V1 valuation is:

```text
NAV = sum(quantity * value)
```

FX conversion is not implemented yet, so instruments should already be in the portfolio base currency.

## Analytics

### Performance

```python
performance = portfolio.performance()
print(performance)
```

Included metrics:

- `total_return`
- `cagr`
- `ytd_return`
- `mtd_return`
- `twr`
- `xirr`
- `latest_monthly_return`
- `latest_rolling_252_return`

### Risk

```python
risk = portfolio.risk(risk_free_rate=0.02)
print(risk)
```

Included metrics:

- `annualized_volatility`
- `max_drawdown`
- `current_drawdown`
- `downside_deviation`
- `sharpe_ratio`
- `sortino_ratio`
- `historical_var_95`
- `historical_cvar_95`

You can also pass benchmark NAV history to calculate beta and correlation:

```python
risk = portfolio.risk(benchmark_nav=benchmark_nav)
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
print(scenario.portfolio.performance())
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
print(portfolio.performance())
print(portfolio.risk())
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
    scenarios/       What-if scenario cloning and comparison
    utils/           Date helpers
tests/               Deterministic unit tests
examples/            Small usage scripts
data/                Local Parquet data files
```

## V1 Limitations

- Only BUY and SELL trades are supported.
- Holdings are quantity-based.
- Valuation is currently `quantity * value`.
- Instruments are assumed to already be in the portfolio base currency.
- There is no cash ledger yet.
- There is no tax, broker, lot accounting, or reconciliation system.
- There are no options, futures, Greeks, optimization, Monte Carlo, web UI, or live market data.

## Roadmap

Likely V2 improvements:

- Cash ledger and explicit contributions/withdrawals.
- FX rates and currency conversion.
- Separate valuation models for options, futures, bonds, and FX forwards.
- Risk factors and scenario repricing.
- Broker importers and reconciliation.
- Richer metadata for sector, country, duration, strategy, and account-level reporting.
