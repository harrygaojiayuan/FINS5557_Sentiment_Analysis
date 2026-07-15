""" FinBERT Sentence-Level Sentiment Analysis

Use FinBERT to perform sentence level sentiment analysis
Adapted from week2-seminar.ipynb (FINS5557 Applied AI in Finance Hric & Lin 2026), and
chapter-08-sentiment case study (Applied Data Science in FinTech Hric & Lin 2026)
"""

from __future__ import annotations

import config

LABELS = ('positive', 'negative', 'neutral',)

def load_pipeline():
    """ Load the FinBERT pipeline for sentiment analysis

    Approx. 440MB download on first run, recommend cache the returned object

    Adapted from week2-seminar.ipynb, Section 2
    (FINS5557 Applied AI in Finance Hric & Lin 2026)
    """
    from transformers import pipeline # Import when called

    return pipeline(
        "text-classification",
        model=config.SENTIMENT_MODEL, # Set model if specified in the .env, otherwise default as specified in the config
        truncation=True, # truncation if max_length is reached, avoid errors
        max_length=512, # Maximum tokeniser token
        top_k=None # Return all three label's predictions
    )

def score_sentences(nlp, sentences: list[str], batch_size: int=16) -> list[dict]:
    """ Calculate sentence level sentiment score

    Pass sentences into pipeline in batches

    Return:
        A list of dictionaries with each sentence and its sentiment score

    Adapted from week2-seminar.ipynb, Section 2
    (FINS5557 Applied AI in Finance Hric & Lin 2026)
    """

    results: list[dict] = []
    for start in range(0, len(sentences), batch_size): # Setup batches for long filing
        batch = sentences[start : start + batch_size]
        preds = nlp(batch, batch_size=batch_size)

        for sentence, dist in zip(batch, preds): # Convert the full class distribution into a label-to-probability map.
            probs = {d["label"].lower(): float(d["score"]) for d in dist}
            label = max(probs, key=probs.get)
            results.append(
                {
                    "sentence": sentence,
                    "label": label,
                    "confidence": round(probs[label], 4),
                    "prob_score": round(
                        probs.get("positive", 0.0) - probs.get("negative", 0.0), 4
                    ),
                }
            )
    return results

def aggregate(scored: list[dict], top_n: int = 3) -> dict:
    """Aggregate sentence-level scores into section-level results.

    FinBERT returns the full positive, neutral, and negative class distribution for each sentence.
    Aggregate sentiment statistics by sections

    Two section-level sentiment scores are calculated:

        net_score =
            (positive_count - negative_count) / total_sentences

        weighted_score =
            mean(P(positive) - P(negative)) across all sentences

    Returns:
        A dictionary containing sentence count, label counts and shares,
        label-based and probability-based net scores, and top evidence.
    """

    total = len(scored)
    counts = {label: 0 for label in LABELS}
    for row in scored: # Count sentences by their highest-probability label.
        if row["label"] in counts:
            counts[row["label"]] += 1
    shares = { # Convert label counts into section-level sentence shares.
        label: round(counts[label] / total, 4) if total else 0.0
        for label in LABELS
    }
    net_score = ( # Calculate net positive counts percentage
        round((counts["positive"] - counts["negative"]) / total, 4)
        if total
        else 0.0
    )
    prob_scores = [r.get("prob_score", 0.0) for r in scored] # Get the probability of the score
    weighted_score = round(sum(prob_scores) / total, 4) if total else 0.0 # Average scores across sections

    def top(label: str) -> list[dict]: # Rank evidence by winning-label confidence within each polarity.
        matching = [r for r in scored if r["label"] == label]
        matching.sort(key=lambda r: r["confidence"], reverse=True)
        return matching[:top_n]

    return {
        "n_sentences": total,
        "counts": counts,
        "shares": shares,
        "net_score": net_score,
        "weighted_score": weighted_score,
        "top_positive": top("positive"),
        "top_negative": top("negative"),
    }

if __name__ == "__main__":
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    from sec_edgar import fetch_filing_html, list_10q_filings
    from section_parser import (
        SECTION_TITLES,
        extract_sentences,
        html_to_text,
        split_sections,
    )

    TICKER = "NVDA"
    MAX_SENTENCES = 50

    # Anchor output to THIS file's folder, not the working directory,
    # so the path is identical in PyCharm, terminal, and across devices.
    RESULTS_DIR = Path(__file__).resolve().parent / "test_result" / "sentiment"
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"1/4 Fetching latest 10-Q for {TICKER} from EDGAR...")
    filing = list_10q_filings(TICKER, limit=1)[0]
    print(f"    {filing.company} — {filing.label}")

    print("2/4 Parsing sections...")
    html = fetch_filing_html(filing)
    sections = split_sections(html_to_text(html))
    print(f"    {len(sections)} sections: {', '.join(sections)}")

    print("3/4 Loading FinBERT (first run downloads ~440 MB)...")
    nlp = load_pipeline()

    print(f"4/4 Scoring sections (capped at {MAX_SENTENCES} sentences each)...\n")
    print(
        f"    {'Section':<56} {'Net':>7} {'Wtd':>7} "
        f"{'Pos':>5} {'Neu':>5} {'Neg':>5}   Tone"
    )
    print("    " + "-" * 92)

    result = {
        "ticker": filing.ticker,
        "company": filing.company,
        "filing": filing.label,
        "report_date": filing.report_date,
        "max_sentences_per_section": MAX_SENTENCES,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sections": {},
    }

    for key, text in sections.items():
        sentences = extract_sentences(text)[:MAX_SENTENCES]
        if not sentences:
            continue
        scored = score_sentences(nlp, sentences)
        agg = aggregate(scored)
        result["sections"][key] = {
            "title": SECTION_TITLES[key],
            "aggregate": agg,
            "sentences": scored,
        }
        c = agg["counts"]
        tone = (
            "positive" if agg["net_score"] > 0.05
            else "negative" if agg["net_score"] < -0.05
            else "neutral"
        )
        print(
            f"    {SECTION_TITLES[key]:<56} "
            f"{agg['net_score']:>+7.2f} {agg.get('weighted_score', 0.0):>+7.2f} "
            f"{c['positive']:>5} {c['neutral']:>5} {c['negative']:>5}   {tone}"
        )

    out_path = RESULTS_DIR / f"{filing.ticker}_{filing.report_date}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_path}")