"""Performance benchmark for the 10-Q analysis pipeline.

Measures FinBERT cold-load time once, then per filing: EDGAR retrieval,
parsing + sentence extraction, FinBERT inference, LLM summary generation,
and end-to-end wall time. Results are written to evaluation/ as report
evidence (rubric 2d: application performance metrics).

Usage:
  python performance_test.py                        # 8 default tickers x 1 filing
  python performance_test.py --per-ticker 2         # more filings per ticker
  python performance_test.py --tickers AAPL MSFT    # custom ticker list
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import config
from sec_edgar import fetch_filing_html, list_10q_filings
from section_parser import (
    SECTION_TITLES,
    extract_sentences,
    html_to_text,
    split_sections,
)
from sentiment import aggregate, score_sentences
from summariser import generate_summary

EVAL_DIR = Path(__file__).resolve().parent / "evaluation"
DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "NFLX", "JPM", "COST", "KO"]
METRICS = ("fetch_s", "parse_s", "finbert_inference_s", "llm_generation_s", "end_to_end_s")


def benchmark_filing(filing, nlp, cap: int) -> dict:
    """Run the full pipeline for one filing, timing each stage."""
    row = {"ticker": filing.ticker, "period": filing.report_date}
    t_start = time.perf_counter()

    t = time.perf_counter()
    html = fetch_filing_html(filing)
    row["fetch_s"] = time.perf_counter() - t
    row["html_chars"] = len(html)

    t = time.perf_counter()
    sections_text = split_sections(html_to_text(html))
    section_sentences = {
        k: extract_sentences(v)[:cap] for k, v in sections_text.items()
    }
    row["parse_s"] = time.perf_counter() - t

    t = time.perf_counter()
    section_analyses = {}
    n_scored = 0
    for key, sentences in section_sentences.items():
        if not sentences:
            continue
        scored = score_sentences(nlp, sentences)
        n_scored += len(scored)
        section_analyses[key] = {
            "title": SECTION_TITLES.get(key, key),
            "aggregate": aggregate(scored),
            "sentences": scored,
        }
    row["finbert_inference_s"] = time.perf_counter() - t
    row["n_sentences_scored"] = n_scored

    meta = {
        "ticker": filing.ticker, "company": filing.company, "form": filing.form,
        "report_date": filing.report_date, "filing_date": filing.filing_date,
    }
    t = time.perf_counter()
    summary = generate_summary(meta, sections_text, section_analyses)
    row["llm_generation_s"] = time.perf_counter() - t
    row["generated_by"] = summary.get("generated_by")

    row["end_to_end_s"] = time.perf_counter() - t_start
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--per-ticker", type=int, default=1)
    args = parser.parse_args()

    EVAL_DIR.mkdir(exist_ok=True)
    cap = getattr(config, "MAX_SENTENCES_PER_SECTION", 300)

    print("Measuring FinBERT cold-load time...")
    t = time.perf_counter()
    from sentiment import load_pipeline

    nlp = load_pipeline()
    cold_load_s = time.perf_counter() - t
    print(f"  cold load: {cold_load_s:.1f}s "
          "(first-ever run would additionally download ~440 MB)\n")

    runs = []
    for ticker in args.tickers:
        try:
            filings = list_10q_filings(ticker, limit=args.per_ticker)
        except Exception as exc:
            print(f"{ticker}: listing failed ({exc}), skipped")
            continue
        for filing in filings[: args.per_ticker]:
            print(f"Benchmarking {ticker} {filing.report_date} ...")
            try:
                row = benchmark_filing(filing, nlp, cap)
            except Exception as exc:
                print(f"  FAILED: {exc}")
                continue
            runs.append(row)
            print(
                f"  fetch {row['fetch_s']:.1f}s | parse {row['parse_s']:.1f}s | "
                f"FinBERT {row['finbert_inference_s']:.1f}s ({row['n_sentences_scored']} sents) | "
                f"LLM {row['llm_generation_s']:.1f}s | total {row['end_to_end_s']:.1f}s"
            )

    if not runs:
        raise SystemExit("No successful runs.")

    print(f"\n=== Summary over {len(runs)} filings "
          f"(cap={cap} sentences/section, provider={config.active_provider()}) ===")
    stats = {}
    for metric in METRICS:
        values = [r[metric] for r in runs]
        stats[metric] = {
            "mean": round(statistics.mean(values), 2),
            "median": round(statistics.median(values), 2),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
        }
        print(f"  {metric:<22} mean {stats[metric]['mean']:>7.2f}s   "
              f"median {stats[metric]['median']:>7.2f}s   "
              f"min {stats[metric]['min']:>6.2f}s   max {stats[metric]['max']:>7.2f}s")

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "machine": platform.platform(),
            "python": platform.python_version(),
            "provider": config.active_provider(),
            "sentiment_model": config.SENTIMENT_MODEL,
            "max_sentences_per_section": cap,
        },
        "finbert_cold_load_s": round(cold_load_s, 2),
        "runs": [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()} for r in runs],
        "summary_stats": stats,
    }
    out = EVAL_DIR / f"performance_results_{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()