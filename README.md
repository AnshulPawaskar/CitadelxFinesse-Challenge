# Static Portfolio Selection — CitadelxFinesse Challenge

**Team:** Quantify
**Members:** Anshul Pawaskar, Jatin Dhanani, Ashish Kela

A static (buy-once, hold, zero-rebalancing) portfolio selection engine built on technical and
quantitative signals, covering the Nifty 100 + Midcap 100 + Smallcap 100 universe.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

TA-Lib requires the underlying C library on some platforms — if `pip install ta-lib` fails, install
`ta-lib` via your system package manager first (e.g. `apt install ta-lib` / `brew install ta-lib`).

Create a `.env` file in the repo root with your 5Paisa API credentials (only needed for `main.py`,
the live data extraction — not needed to run the offline research pipeline against the data already
in `Data/`):

```
APP_NAME=...
APP_SOURCE=...
USER_ID=...
PASSWORD=...
USER_KEY=...
ENCRYPTION_KEY=...
REQUEST_TOKEN=...
```

## What to run

### 1. Data extraction (optional — only if you need to refresh `Data/`)

```bash
python main.py
```

Pulls historical OHLCV data from 5Paisa into `Data/1d/{year}/{half}/{SYMBOL}_EQ.parquet`. Requires
a fresh `REQUEST_TOKEN` each run (single-use, short-lived).

### 2. Full research/backtest/reporting pipeline

```bash
python run_pipeline.py
```

Runs every stage in order: data audit → feature engineering → baseline backtest → model comparison
→ feature correlation → correlation-aware selection → weighting comparison → the full experiment
matrix → sensitivity analysis → the official competition backtest → charts/Excel/HTML reports →
the final submission report → the oracle (hindsight) comparison. Takes ~90 seconds.

```bash
python run_pipeline.py --list              # see all stage names
python run_pipeline.py --only experiments  # run just one stage
python run_pipeline.py --skip experiments  # run everything except one
```

### 3. Individual stages

Every stage can also be run standalone, e.g.:

```bash
python -m src.features.feature_pipeline        # rebuild results/features.parquet
python -m src.competition.final_submission     # the official competition backtest
python -m src.reporting.final_report           # regenerate the submission report
```

### 4. Tests

```bash
python -m pytest
```

## Repo structure

```
main.py                     Live 5Paisa data extraction (separate from the offline pipeline)
run_pipeline.py              Single entry point to run the offline pipeline end-to-end

Data/                        Local OHLCV parquet files (source of truth, not fetched at runtime)
Stocks/                      Nifty 100 / Midcap 100 / Smallcap 100 constituent lists
ScriptBook/                  Cached 5Paisa script master (symbol -> scrip code mapping)

src/
  data/                       Universe construction, data loading, data-quality validation
  features/                   Momentum, relative strength, trend, technical, volatility, volume features
  signals/                    Cross-sectional ranking and composite alpha scoring models
  portfolio/                  Stock selection (Top-N / correlation-filtered) and weighting methods
  backtest/                   Static portfolio construction, execution, costs, metrics, attribution
  research/                   Experiments, sensitivity analysis, model/weighting comparisons
  competition/                The official, isolated competition submission backtest
  reporting/                  Charts, Excel workbook, HTML report, final submission report

tests/                        Unit tests (pytest)
results/                      Generated CSVs/parquet outputs
reports/                      Generated HTML/Markdown reports
plots/                        Generated charts
```

## Known limitations

- No local Nifty 100/Midcap 100/Smallcap 100 index-level price series exists, so benchmark
  comparisons use a synthetic equal-weighted universe proxy instead of the real index.
- Corporate-action (split/bonus) adjustment status of the local OHLCV data is unverified — only raw
  `Close` is available, no `Adj Close` column.
