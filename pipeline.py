"""End-to-end analysis pipeline producing one standardised JSON document.

Chains sec_edgar → section_parser → sentiment → summariser, and is kept free
of any UI code so the Streamlit app and the module self-tests share the same
logic. Provenance fields (generated_at, model ids, app version) follow the
audit-trail pattern in week9-case-study.ipynb Station 4
(FINS5557 Applied AI in Finance, Hric & Lin 2026).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

import config
from sec_edgar import Filing
from section_parser import (
    SECTION_TITLES,
    extract_sentences,
    html_to_text,
    split_sections,
)
from sentiment import aggregate, score_sentences
from summariser import generate_summary


def analyse_filing(
    filing: Filing,
    html: str,
    nlp,
    max_sentences: int = config.MAX_SENTENCES_PER_SECTION,
    progress: Callable[[str], None] = lambda msg: None,
) -> dict:
    """Run parse → FinBERT → summarise; return the standard analysis JSON.

    Args:
        filing: Filing metadata from sec_edgar.list_10q_filings.
        html: Filing HTML from sec_edgar.fetch_filing_html.
        nlp: Loaded FinBERT pipeline (sentiment.load_pipeline) — passed in
            so the caller controls caching (e.g. st.cache_resource).
        max_sentences: Per-section cap keeping CPU inference responsive.
        progress: Optional callback receiving stage descriptions — the
            Streamlit app can feed these into st.status.
    """
    meta = {
        "ticker": filing.ticker,
        "company": filing.company,
        "cik": filing.cik,
        "form": filing.form,
        "accession": filing.accession,
        "filing_date": filing.filing_date,
        "report_date": filing.report_date,
        "source_url": filing.document_url,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "app_version": getattr(config, "APP_VERSION", "1.0.0"),
    }

    progress("Extracting text and splitting sections…")
    sections_text = split_sections(html_to_text(html))
    if not sections_text:
        raise ValueError(
            "No standard 10-Q sections could be located in this document. "
            "It may use a non-standard layout — try another filing."
        )

    section_analyses: dict[str, dict] = {}
    for key, text in sections_text.items():
        title = SECTION_TITLES.get(key, key)
        progress(f"Scoring sentiment: {title}…")
        sentences = extract_sentences(text)[:max_sentences]
        if not sentences:
            continue
        scored = score_sentences(nlp, sentences)
        section_analyses[key] = {
            "title": title,
            "aggregate": aggregate(scored),
            "sentences": scored,
        }

    if not section_analyses:
        raise ValueError(
            "Sections were located but no scoreable sentences survived "
            "filtering — the filing may be table-only or non-standard."
        )

    progress("Generating executive summary…")
    summary = generate_summary(meta, sections_text, section_analyses)

    return {
        "meta": meta,
        "models": {
            "sentiment_model": config.SENTIMENT_MODEL,
            "summary_generated_by": summary.get("generated_by"),
        },
        "sections": section_analyses,
        "executive_summary": summary,
    }


if __name__ == "__main__":
    import json
    from pathlib import Path

    from sec_edgar import fetch_filing_html, list_10q_filings
    from sentiment import load_pipeline

    TICKER = "AAPL"
    RESULTS_DIR = Path(__file__).resolve().parent / "test_result" / "pipeline"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    print(f"1/3 Fetching latest 10-Q for {TICKER}...")
    filing = list_10q_filings(TICKER, limit=1)[0]
    html = fetch_filing_html(filing)
    print(f"    {filing.company} — {filing.label}")

    print("2/3 Loading FinBERT...")
    nlp = load_pipeline()

    print("3/3 Running full pipeline...")
    analysis = analyse_filing(filing, html, nlp, progress=lambda m: print(f"    {m}"))

    summary = analysis["executive_summary"]
    print("\nResult:")
    print(f"    sections     : {len(analysis['sections'])}")
    print(f"    generated_by : {summary.get('generated_by')}")
    print(f"    llm_error    : {summary.get('llm_error', '(none)')}")
    print(f"    tone         : {summary['overall_assessment']['tone']}")

    out = RESULTS_DIR / f"{filing.ticker}_{filing.report_date}_analysis_{run_ts}.json"
    out.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out}")
