"""Tone-threshold sensitivity analysis for the +/-0.05 tone boundary.

Usage:
  python threshold_analysis.py --collect AAPL MSFT NVDA TSLA NFLX JPM COST KO
  python threshold_analysis.py --analyse

--collect scores each ticker's latest 10-Q MD&A with FinBERT (no LLM cost)
and caches the sentence-weighted score. --analyse sweeps thresholds
0.02-0.10 and reports how often the tone label flips versus t=0.05.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent / "evaluation"
SCORES = EVAL_DIR / "tone_scores.json"
THRESHOLDS = [round(0.02 + 0.005 * i, 3) for i in range(17)]  # 0.02 .. 0.10
BASE = 0.05


def tone(score: float, t: float) -> str:
    return "positive" if score > t else "negative" if score < -t else "neutral"


def collect(tickers: list[str]) -> None:
    from sec_edgar import fetch_filing_html, list_10q_filings
    from section_parser import extract_sentences, html_to_text, split_sections
    from sentiment import aggregate, load_pipeline, score_sentences

    EVAL_DIR.mkdir(exist_ok=True)
    entries = json.loads(SCORES.read_text(encoding="utf-8")) if SCORES.exists() else []
    have = {(e["ticker"], e["period"]) for e in entries}
    nlp = load_pipeline()
    for ticker in tickers:
        filing = list_10q_filings(ticker, limit=1)[0]
        if (filing.ticker, filing.report_date) in have:
            print(f"{ticker}: already collected")
            continue
        text = split_sections(html_to_text(fetch_filing_html(filing))).get("mdna", "")
        sentences = extract_sentences(text)
        if not sentences:
            print(f"{ticker}: no MD&A sentences, skipped")
            continue
        agg = aggregate(score_sentences(nlp, sentences))
        entries.append(
            {"ticker": filing.ticker, "period": filing.report_date,
             "weighted_score": agg["weighted_score"], "n_sentences": agg["n_sentences"]}
        )
        print(f"{ticker} {filing.report_date}: weighted={agg['weighted_score']:+.4f}")
    SCORES.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"Saved {len(entries)} filings -> {SCORES}")


def analyse() -> None:
    entries = json.loads(SCORES.read_text(encoding="utf-8"))
    if len(entries) < 5:
        raise SystemExit("Collect at least ~5 filings first (--collect T1 T2 ...).")
    flips = {t: 0 for t in THRESHOLDS}
    print(f"{'ticker':<8} {'period':<12} {'score':>7}  tones at t=0.02..0.10 (pos/neu/neg)")
    for e in entries:
        s = e["weighted_score"]
        base_tone = tone(s, BASE)
        marks = []
        for t in THRESHOLDS:
            x = tone(s, t)
            if x != base_tone:
                flips[t] += 1
            marks.append(x[:3])
        print(f"{e['ticker']:<8} {e['period']:<12} {s:>+7.3f}  {' '.join(marks)}")
    print(f"\nTone flips relative to t={BASE}:")
    for t in THRESHOLDS:
        print(f"  t={t:.3f}: {flips[t]}/{len(entries)} filings change label")
    csv_path = EVAL_DIR / "threshold_sensitivity.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("ticker,period,weighted_score," + ",".join(map(str, THRESHOLDS)) + "\n")
        for e in entries:
            f.write(
                f"{e['ticker']},{e['period']},{e['weighted_score']},"
                + ",".join(tone(e["weighted_score"], t) for t in THRESHOLDS) + "\n"
            )
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", nargs="+", metavar="TICKER")
    parser.add_argument("--analyse", action="store_true")
    args = parser.parse_args()
    if args.collect:
        collect(args.collect)
    elif args.analyse:
        analyse()
    else:
        parser.print_help()