"""Earnings-surprise context from Finnhub (free tier).

The /stock/earnings endpoint returns, for each recent quarter, the analyst
consensus EPS estimate as it stood at announcement time, together with the
reported actual and the surprise — no point-in-time reconstruction needed.
Shown as market context next to the filing analysis; NOT part of the
sentiment model (see report, Limitations: narrative tone vs market surprise).
"""

from __future__ import annotations

from datetime import date

import requests

import config

_URL = "https://finnhub.io/api/v1/stock/earnings"


class MarketDataError(Exception):
    """Raised when Finnhub data cannot be retrieved or understood."""


def get_earnings_surprises(ticker: str, limit: int = 8) -> list[dict]:
    """Return recent quarterly {period, estimate, actual, surprise_pct} rows."""
    if not config.FINNHUB_API_KEY:
        return []  # feature off — callers must tolerate empty
    try:
        resp = requests.get(
            _URL,
            params={"symbol": ticker, "limit": limit,
                    "token": config.FINNHUB_API_KEY},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        raise MarketDataError(f"Finnhub request failed: {exc}") from exc
    if not isinstance(data, list):
        raise MarketDataError(f"Unexpected Finnhub response: {str(data)[:200]}")
    return [
        {
            "period": row.get("period"),
            "estimate": row.get("estimate"),
            "actual": row.get("actual"),
            "surprise_pct": row.get("surprisePercent"),
        }
        for row in data
        if row.get("period")
    ]


def surprise_for_period(ticker: str, report_date: str) -> dict | None:
    """Match a filing's quarter end to the nearest Finnhub period (≤45 days).

    Fiscal quarter ends (e.g. Apple's 2026-03-28) rarely equal calendar
    quarter ends (Finnhub's 2026-03-31), so match by nearest date.
    """
    try:
        rows = get_earnings_surprises(ticker)
    except MarketDataError:
        return None  # degrade silently — context feature must never break the app
    target = date.fromisoformat(report_date)
    best, best_gap = None, 999
    for row in rows:
        try:
            gap = abs((date.fromisoformat(row["period"]) - target).days)
        except (TypeError, ValueError):
            continue
        if gap < best_gap:
            best, best_gap = row, gap
    return best if best is not None and best_gap <= 45 else None


if __name__ == "__main__":
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    TICKER = "NFLX"
    RESULTS_DIR = Path(__file__).resolve().parent / "test_result" / "market_data"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    print(f"1/2 get_earnings_surprises({TICKER!r})...")
    rows = get_earnings_surprises(TICKER)
    for row in rows:
        print(f"    {row['period']}  est={row['estimate']}  "
              f"actual={row['actual']}  surprise={row['surprise_pct']}%")
    if not rows:
        print("    (empty — is FINNHUB_API_KEY set in .env?)")

    print("2/2 surprise_for_period, latest 10-Q...")
    from sec_edgar import list_10q_filings
    filing = list_10q_filings(TICKER, limit=1)[0]
    match = surprise_for_period(TICKER, filing.report_date)
    print(f"    filing period {filing.report_date} -> {match}")

    out = RESULTS_DIR / f"{TICKER}_surprises_{run_ts}.json"
    out.write_text(
        json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "ticker": TICKER, "rows": rows, "matched": match},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved: {out}")