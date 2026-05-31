"""Central configuration — reads from .env, provides typed defaults."""

import os
from pathlib import Path
from dotenv import load_dotenv

from config.api_keys_loader import external_apis_dir, load_external_api_keys

# External APIs folder first, then project .env (does not override existing env vars)
load_external_api_keys(sync_env=True)
load_dotenv()

# ── Project paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / os.getenv("CACHE_DIR", "data/cache")
DB_PATH = BASE_DIR / os.getenv("DB_PATH", "data/cache/investment_cache.db")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── API Keys ───────────────────────────────────────────────────────────────────
EXTERNAL_APIS_DIR: str = str(external_apis_dir())
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")

# ── LLM settings ──────────────────────────────────────────────────────────────
DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "4096"))

# ── Cache TTLs ─────────────────────────────────────────────────────────────────
MARKET_DATA_TTL_HOURS: int = int(os.getenv("MARKET_DATA_TTL_HOURS", "24"))
FUNDAMENTAL_DATA_TTL_DAYS: int = int(os.getenv("FUNDAMENTAL_DATA_TTL_DAYS", "90"))

# ── Finance model constants ────────────────────────────────────────────────────
MARKET_RISK_PREMIUM: float = float(os.getenv("MARKET_RISK_PREMIUM", "0.07"))
DEFAULT_TAX_RATE: float = float(os.getenv("DEFAULT_TAX_RATE", "0.21"))
DCF_FORECAST_YEARS: int = int(os.getenv("DCF_FORECAST_YEARS", "5"))
MONTE_CARLO_SIMULATIONS: int = int(os.getenv("MONTE_CARLO_SIMULATIONS", "10000"))

# Beta regression lookback (months)
BETA_LOOKBACK_MONTHS: int = 60

# Alpha Vantage base URL
ALPHA_VANTAGE_BASE_URL: str = "https://www.alphavantage.co/query"

# FRED series IDs
FRED_SERIES = {
    "risk_free_rate": "DGS10",       # 10-Year Treasury Constant Maturity
    "gdp_growth": "A191RL1Q225SBEA", # Real GDP growth rate
    "cpi": "CPIAUCSL",               # CPI for inflation estimate
    "fed_funds": "FEDFUNDS",
}


def validate_keys() -> dict[str, bool]:
    """Return which API keys are configured."""
    return {
        "openai": bool(OPENAI_API_KEY),
        "anthropic": bool(ANTHROPIC_API_KEY),
        "alpha_vantage": bool(ALPHA_VANTAGE_API_KEY),
        "fred": bool(FRED_API_KEY),
    }
