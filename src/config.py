from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "Data"
STOCKS_DIR = ROOT_DIR / "Stocks"
SCRIPTBOOK_DIR = ROOT_DIR / "ScriptBook"
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = ROOT_DIR / "plots"
REPORTS_DIR = ROOT_DIR / "reports"

TIMEFRAME = "1d"
INDICES = ["nse_100", "nse_midcap_100", "nse_smallcap_100"]

INITIAL_CAPITAL = 1_00_00_000  # Rs 1,00,00,000
MAX_HOLDINGS = 10
TRANSACTION_COST_RATE = 0.001
RISK_FREE_RATE = 0.0

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
