"""Baseline evaluation: FinBERT vs VADER on hand-labelled 10-Q sentences.

Usage:
  python evaluation.py --make-sample   # build evaluation/sample_to_label.csv
  python evaluation.py --score         # after labelling, compute metrics

Three team members independently fill label_1/2/3 with p / u / n
(positive / neutral / negative); majority vote is ground truth.
Serves rubric 2d (baseline comparison) and 1a/2c (model selection evidence).
VADER usage adapted from chapter-08 case study (Hric & Lin 2026).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent / "evaluation"
SAMPLE_CSV = EVAL_DIR / "sample_to_label.csv"
LABELS = ("positive", "neutral", "negative")
CODE = {"p": "positive", "u": "neutral", "n": "negative"}
PER_FILING = 20
SEED = 5557


def make_sample() -> None:
    EVAL_DIR.mkdir(exist_ok=True)
    pattern = str(Path(__file__).resolve().parent / "test_result" / "section_parser" / "*_sections*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit("No section JSONs — run `python section_parser.py` for a few tickers first.")
    random.seed(SEED)
    latest = {}
    for path in files:  # sorted + timestamped names → later file wins
        data = json.load(open(path, encoding="utf-8"))
        latest[(data["ticker"], data.get("report_date"))] = data

    rows = []
    for data in latest.values():
        pool = [
            (key, s)
            for key, sec in data["sections"].items()
            for s in sec["sentences"]
        ]
        for key, sentence in random.sample(pool, min(PER_FILING, len(pool))):
            rows.append({"ticker": data["ticker"], "section": key, "sentence": sentence})
    random.shuffle(rows)
    with open(SAMPLE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "ticker", "section", "sentence", "label_1", "label_2", "label_3"]
        )
        writer.writeheader()
        for i, row in enumerate(rows, 1):
            writer.writerow({"id": i, **row, "label_1": "", "label_2": "", "label_3": ""})
    print(f"Wrote {len(rows)} sentences -> {SAMPLE_CSV}")
    print("Each member fills ONE label column with p / u / n, then run --score.")


def _majority(votes: list[str]) -> str | None:
    votes = [CODE.get(v.strip().lower()) for v in votes if v.strip()]
    votes = [v for v in votes if v]
    if len(votes) < 2:
        return None
    top, n = Counter(votes).most_common(1)[0]
    return top if n >= 2 else None


def _vader_label(analyzer, text: str) -> str:
    c = analyzer.polarity_scores(text)["compound"]
    return "positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral"

def _lm_label(lm, text: str) -> str:
    """Loughran-McDonald dictionary label: compare finance-domain
    positive vs negative word counts (Loughran & McDonald, 2011)."""
    score = lm.get_score(lm.tokenize(text))
    pos, neg = score["Positive"], score["Negative"]
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"

def _metrics(y_true, y_pred):
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    per = {}
    for lab in LABELS:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred))
        fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred))
        fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per[lab] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return acc, per, sum(per.values()) / len(LABELS)


def score() -> None:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    from sentiment import load_pipeline, score_sentences

    rows = list(csv.DictReader(open(SAMPLE_CSV, encoding="utf-8")))
    labelled = [(r, _majority([r["label_1"], r["label_2"], r["label_3"]])) for r in rows]
    labelled = [(r, t) for r, t in labelled if t]
    if len(labelled) < 30:
        raise SystemExit(f"Only {len(labelled)} rows have >=2 agreeing votes — label more first.")

    full_agree = sum(
        1 for r, _ in labelled
        if len({CODE.get(r[c].strip().lower()) for c in ("label_1", "label_2", "label_3")}) == 1
    )
    sentences = [r["sentence"] for r, _ in labelled]
    y_true = [t for _, t in labelled]

    print(f"Scoring {len(labelled)} sentences (all-3-agree: {full_agree}/{len(labelled)})...")
    nlp = load_pipeline()
    fb = [x["label"] for x in score_sentences(nlp, sentences)]
    vd = [_vader_label(SentimentIntensityAnalyzer(), s) for s in sentences]

    import pysentiment2 as ps
    lm = ps.LM()
    lmd = [_lm_label(lm, s) for s in sentences]

    report = {"n": len(labelled), "inter_annotator_full_agreement": round(full_agree / len(labelled), 3)}
    for name, pred in (("FinBERT", fb), ("VADER", vd), ("Loughran-McDonald", lmd)):
        acc, per, macro = _metrics(y_true, pred)
        report[name] = {
            "accuracy": round(acc, 3),
            "macro_f1": round(macro, 3),
            "per_class_f1": {k: round(v, 3) for k, v in per.items()},
        }
        print(f"\n{name}: accuracy={acc:.3f}  macro-F1={macro:.3f}")
        for lab in LABELS:
            print(f"   {lab:<9} F1={per[lab]:.3f}")

    disagreements = [
        {"sentence": r["sentence"], "truth": t, "vader": v}
        for (r, t), f, v in zip(labelled, fb, vd)
        if f == t != v
    ]
    print("\nFinBERT right, VADER wrong (report examples):")
    for d in disagreements[:5]:
        print(f"  [{d['truth']}] vs VADER={d['vader']}: {d['sentence'][:90]}")

    out = EVAL_DIR / f"baseline_results_{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(
        json.dumps({**report, "finbert_right_vader_wrong": disagreements}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--make-sample", action="store_true")
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()
    if args.make_sample:
        make_sample()
    elif args.score:
        score()
    else:
        parser.print_help()