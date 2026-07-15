"""Use EDGAR APIs to fetch 10-Q fillings of a company:

The code fetches tickers, CIK (Central Index Key) from company_tickers.json on sec.gov
Then use the CIK to fetch filling history though data.sec.gov/submissions/CIK##########.json

"""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests
import config

tickers_url = 'https://www.sec.gov/files/company_tickers.json'
submissions_url = 'https://data.sec.gov/submissions/CIK{cik:010d}.json'
archive_url = 'https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}'

_min_request_interval = 0.15  # seconds — stays well under EDGAR's 10 req/s cap
_last_request_time = 0.0


class EdgarError(Exception):
    """Raised when EDGAR data cannot be retrieved or understood."""


@dataclass(frozen=True)
class Filing:
    cik: int
    company: str
    ticker: str
    form: str
    accession: str
    filing_date: str
    report_date: str
    primary_doc: str

    @property
    def document_url(self) -> str:
        """Return the full URL for this filing"""
        return archive_url.format(
            cik=self.cik,
            accession=self.accession.replace("-", ""), # Replace '-' in the original fetch with space as SEC Archive directory does not allow
            doc=self.primary_doc,
        )

    @property
    def label(self) -> str:
        """Return the label for this filing"""
        return f"{self.form} — period {self.report_date} (filed {self.filing_date})" # Return in the format of 'Form Type' - 'Report Date' ('filling date')

def _get(url: str) -> requests.Response:
    """GET with EDGAR-compliant headers, throttling, and readable errors."""
    global _last_request_time
    wait = _min_request_interval - (time.time() - _last_request_time) # Set wait time to be minimum of 0.15 sec to comply with EDGAR'S request cap
    if wait > 0:
        time.sleep(wait)
    try:
        resp = requests.get(
            url, headers={"User-Agent": config.sec_user_agent}, timeout=30 # Get from specified URL using the SEC_USER_AGENT in the config.py which is specified in the .env. Exit if not done within 30 seconds
        )
        _last_request_time = time.time() # Update request time for the next get
        resp.raise_for_status()
        return resp
    except requests.Timeout as exc: # Raise EdgarError when request is over 30sec
        raise EdgarError("SEC EDGAR timed out — please try again shortly.") from exc
    except requests.HTTPError as exc: # Raise HTTPError if 400<= response status <600
        raise EdgarError(
            f"SEC EDGAR returned HTTP {exc.response.status_code} for {url}. "
            "Check the SEC_USER_AGENT setting in .env identifies you correctly."
        ) from exc
    except requests.RequestException as exc: # Deal with other Errors
        raise EdgarError(f"Could not reach SEC EDGAR: {exc}") from exc

def ticker_to_cik(ticker: str) -> tuple[int, str]:
    """Return (CIK, registered company name) for a US-listed ticker."""
    symbol = ticker.strip().upper().replace(".", "-") # Clean ticker input, replace '.' with '-'. E.g. Berkshire Hathaway B ticker should be BRK-B
    if not symbol or len(symbol) > 10: # Validate ticker input
        raise EdgarError("Please enter a valid ticker symbol (e.g. AAPL, MSFT).")
    data = _get(tickers_url).json() # Request data from https://www.sec.gov/files/company_tickers.json
    for row in data.values(): # iterate within values of data dictionary, return CIK and company name if ticker exists
        if row["ticker"] == symbol:
            return int(row["cik_str"]), row["title"]
    raise EdgarError( # Raise EdgarError if ticker entered does not exist
        f"Ticker '{symbol}' was not found on SEC EDGAR. Only US-listed "
        "companies file 10-Qs; check the symbol (class shares use a dash, "
        "e.g. BRK-B)."
    )

def list_10q_filings(ticker: str, limit: int = 12) -> list[Filing]:
    """Return the most recent 10-Q filings for a ticker, newest first."""
    cik, company = ticker_to_cik(ticker)
    data = _get(submissions_url.format(cik=cik)).json() # Access SEC submission data
    recent = data.get("filings", {}).get("recent", {}) # get recent filing
    filings: list[Filing] = []
    rows = zip(
        recent.get("form", []),
        recent.get("accessionNumber", []),
        recent.get("filingDate", []),
        recent.get("reportDate", []),
        recent.get("primaryDocument", []),
    ) # mapping attributes to each forms
    for form, accession, filing_date, report_date, primary_doc in rows:
        if form != "10-Q" or not primary_doc:
            continue # only proceed if the form is 10-Q and primary_doc is not null, otherwise go to the next iteration
        filings.append( # append new Filing to filing
            Filing(
                cik=cik,
                company=company,
                ticker=ticker.strip().upper().replace(".", "-"),
                form=form,
                accession=accession,
                filing_date=filing_date,
                report_date=report_date,
                primary_doc=primary_doc,
            )
        )
        if len(filings) >= limit: # check the appended limit
            break
    if not filings:
        raise EdgarError( # Raise EdgarError if no 10-Q files has been found. Limitation of this project: Only American companies are considered as foreign companies submit 6-K or 20-K
            f"No 10-Q filings found for {company}. Foreign issuers file "
            "6-K/20-F instead — try a US-domiciled company."
        )
    return filings

def fetch_filing_html(filing: Filing) -> str:
    """Download the primary 10-Q document (HTML) for a filing."""
    return _get(filing.document_url).text

if __name__ == "__main__":
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    TICKER = "AAPL"
    RESULTS_DIR = Path(__file__).resolve().parent / "test_result" / "sec_edgar"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"1/3 ticker_to_cik({TICKER!r})...")
    cik, company = ticker_to_cik(TICKER)
    print(f"    CIK {cik} — {company}")

    print("2/3 list_10q_filings...")
    filings = list_10q_filings(TICKER, limit=8)
    for f in filings:
        print(f"    {f.label}")

    print("3/3 fetch_filing_html (latest)...")
    latest = filings[0]
    html = fetch_filing_html(latest)
    print(f"    {len(html):,} chars from {latest.document_url}")

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker": latest.ticker,
        "cik": cik,
        "company": company,
        "latest_html_chars": len(html),
        "filings": [
            {
                "form": f.form,
                "accession": f.accession,
                "filing_date": f.filing_date,
                "report_date": f.report_date,
                "document_url": f.document_url,
            }
            for f in filings
        ],
    }
    out = RESULTS_DIR / f"{latest.ticker}_filings.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    html_out = RESULTS_DIR / f"{latest.ticker}_{latest.report_date}.htm"
    html_out.write_text(html, encoding="utf-8")
    print(f"\nSaved: {out}\nSaved: {html_out}")
