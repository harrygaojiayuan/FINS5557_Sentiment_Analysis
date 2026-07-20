"""Project Configuration"""

import os
from dotenv import load_dotenv

load_dotenv()

# According to SEC EDGAR Fair Access Policy (Accessible via: https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data)
# Maximun request rate is capped at 10 requests
# User Agent is also required to be decleared to identify the requester

sec_user_agent = os.getenv(
    'SEC_USER_AGENT', # read SEC_USER_AGENT in .env
    'FINS5557 Project contact@example.com' # Default value if SEC_USER_AGENT is blank in .env
)

SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL") or "ProsusAI/finbert" # Default model if SENTIMENT_MODEL is blank in .env

# LLM providers. Set the API key for whichever you have
# Detection priority is Gemini > Anthropic > OpenAI > mock
# Adapted from week6-seminar.ipynb §2) (FINS5557 Applied AI in Finance Hric & Lin 2026).
# With no key set, the summariser runs in mock mode.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

GEMINI_MODEL = os.getenv("GEMINI_MODEL") or "gemini-3.5-flash"
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL") or "claude-haiku-4-5-20251001"
OPENAI_MODEL = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")


def active_provider() -> str:
    """Detect API key specified in the .env,
    and return the LLM provider to use, by key-presence priority.
    """
    if GOOGLE_API_KEY:
        return "gemini"
    if ANTHROPIC_API_KEY:
        return "anthropic"
    if OPENAI_API_KEY:
        return "openai"
    return "mock"

def model_for(provider: str) -> str:
    """Return the model id configured for a provider ('' for mock/unknown)."""
    return {
        "gemini": GEMINI_MODEL,
        "anthropic": ANTHROPIC_MODEL,
        "openai": OPENAI_MODEL,
    }.get(provider, "")

# Cap FinBERT workload per section if necessary
MAX_SENTENCES_PER_SECTION = int(os.getenv("MAX_SENTENCES_PER_SECTION", "300"))