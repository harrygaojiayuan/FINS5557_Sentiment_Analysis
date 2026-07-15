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

SENTIMENT_MODEL = os.getenv(
    "SENTIMENT_MODEL", # read SENTIMENT_MODEL in .env
    "ProsusAI/finbert", # Default model if SENTIMENT_MODEL is blank in .env
)
