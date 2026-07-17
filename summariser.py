"""Executive Summary generation using LLM with a standardised JSON output schema

Adapted from course Jupyter Notebook (FINS5557 Applied AI in Finance Hric & Lin 2026):
- week5-seminar.ipynb §7 "Structured Alternative Data Extraction" —
  "Return ONLY valid JSON" prompts, API-level JSON mode
  (response_mime_type), code-fence stripping, and mock-JSON fallback;
- week6-seminar.ipynb §2 — unified call_llm()/parse_json_response(), and
  Challenge 2 "LLM Earnings Report Parser" — JSON schema embedded in the
  prompt with a mock output structurally identical to the live one;
- week7-seminar.ipynb §6 "LLM Board Report Generator" — structured data
  block serialised into the prompt, cite-figures-exactly instruction, and
  try/except degradation to mock commentary.
"""

from __future__ import annotations

import json
import re

import config
from section_parser import SECTION_TITLES

# The single source of truth for the agent's output format.
SUMMARY_SCHEMA: dict = {
    "financial_highlights": [{"point": "str", "evidence": "verbatim quote"}],
    "business_drivers": [{"point": "str", "evidence": "verbatim quote"}],
    "management_outlook": [{"point": "str", "evidence": "verbatim quote"}],
    "material_risks": [{"point": "str", "evidence": "verbatim quote"}],
    "overall_assessment": {"tone": "positive|neutral|negative|mixed","rationale": "str",},
}

_CATEGORY_KEYS = [
    "financial_highlights",
    "business_drivers",
    "management_outlook",
    "material_risks",
]

_VALID_TONES = {"positive", "neutral", "negative", "mixed"}

# Sections most informative for a summary, in priority order for the prompt's character budget.
_PROMPT_PRIORITY = [
    "mdna",
    "risk_factors",
    "market_risk",
    "legal",
    "controls",
    "financial_statements",
]

# Sections whose language reflects management's own tone.
# Part II sections (risk factors, legal) enumerate risks by regulatory design
# and would bias every filing negative if included in the overall-tone average.
_TONE_SECTIONS = ("mdna",)

def _tone_from_scores(section_analyses: dict[str, dict]) -> tuple[str, float]:
    """Deterministic overall tone: sentence-weighted FinBERT score of MD&A.

    Falls back to all sections when MD&A is absent from the filing.
    """
    pool = [
        a["aggregate"] for k, a in section_analyses.items() if k in _TONE_SECTIONS
    ] or [a["aggregate"] for a in section_analyses.values()]
    total = sum(agg["n_sentences"] for agg in pool)
    if not total:
        return "neutral", 0.0
    mean_score = sum(
        agg.get("weighted_score", agg["net_score"]) * agg["n_sentences"]
        for agg in pool
    ) / total
    tone = (
        "positive" if mean_score > 0.05
        else "negative" if mean_score < -0.05
        else "neutral"
    )
    return tone, round(mean_score, 4)


SYSTEM_PROMPT = (
    "You are an equity research assistant summarising an SEC 10-Q filing for "
    "a junior analyst. Respond with ONLY a valid JSON object — no markdown, "
    "no commentary — matching exactly this schema:\n"
    + json.dumps(SUMMARY_SCHEMA, indent=2)
    + "\nRules: 3-5 entries per list; each 'evidence' value must be a short "
    "verbatim quote from the filing text provided; "
    "inside string values replace any double quotes from the source text with single quotes; "
    "base every point strictly on that text; "
    "if a category has no supporting content, return an empty list for it. "
    "Do not give investment advice."
)


class SummaryError(Exception):
    """Raised when the LLM response cannot be parsed into the schema."""


def build_user_prompt(
    meta: dict, sections_text: dict[str, str],
    section_analyses: dict[str, dict],char_budget: int = 24000,
) -> str:
    """Assemble filing context for the LLM within a character budget."""
    lines = [
        f"Company: {meta.get('company')} ({meta.get('ticker')})",
        f"Filing: {meta.get('form')} for period ending {meta.get('report_date')}"
        f" (filed {meta.get('filing_date')})",
        "",
        "FinBERT section sentiment (positive/neutral/negative shares, net and weighted scores):",
    ]
    for key, analysis in section_analyses.items():
        agg = analysis["aggregate"]
        s = agg["shares"]
        lines.append(
            f"- {SECTION_TITLES.get(key, key)}: "
            f"{s['positive']:.2f}/{s['neutral']:.2f}/{s['negative']:.2f}, "
            f"net {agg['net_score']:+.2f}, weighted {agg.get('weighted_score', 0.0):+.2f} "
            f"({agg['n_sentences']} sentences)"
        )
    lines.append("")

    remaining = char_budget - sum(len(line) for line in lines)
    for key in _PROMPT_PRIORITY:
        text = sections_text.get(key)
        if not text or remaining < 500:
            continue
        excerpt = text[:remaining]
        lines.append(f"=== {SECTION_TITLES.get(key, key)} ===")
        lines.append(excerpt)
        lines.append("")
        remaining -= len(excerpt)
    return "\n".join(lines)


def _extract_json(raw: str) -> dict:
    """Parse a JSON object out of an LLM response, tolerating code fences."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise SummaryError("LLM response contained no JSON object.")
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise SummaryError(f"LLM returned malformed JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SummaryError("LLM JSON was not an object.")
    return data


def _normalise(data: dict) -> dict:
    """Coerce arbitrary LLM output into the exact SUMMARY_SCHEMA shape."""
    result: dict = {}
    for key in _CATEGORY_KEYS:
        items = data.get(key) or []
        clean = []
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and item.get("point"):
                    clean.append(
                        {
                            "point": str(item["point"]).strip(),
                            "evidence": str(item.get("evidence", "")).strip(),
                        }
                    )
                elif isinstance(item, str) and item.strip():
                    clean.append({"point": item.strip(), "evidence": ""})
        result[key] = clean
    assessment = data.get("overall_assessment") or {}
    tone = str(assessment.get("tone", "mixed")).lower()
    result["overall_assessment"] = {
        "tone": tone if tone in _VALID_TONES else "mixed",
        "rationale": str(assessment.get("rationale", "")).strip(),
    }
    return result


def _mock_summary(section_analyses: dict[str, dict]) -> dict:
    """Deterministic fallback built from FinBERT evidence — used when no LLM
    API key is configured or the LLM call fails, so the app stays functional."""

    def sentences_from(key: str, label: str, limit: int) -> list[dict]:
        analysis = section_analyses.get(key)
        if not analysis:
            return []
        ranked = [r for r in analysis["sentences"] if r["label"] == label]
        ranked.sort(key=lambda r: r["confidence"], reverse=True)
        return [
            {"point": r["sentence"][:160], "evidence": r["sentence"]}
            for r in ranked[:limit]
        ]

    def keyword_matches(key: str, regex: re.Pattern, limit: int) -> list[dict]:
        matched = []
        for row in section_analyses.get(key, {}).get("sentences", []):
            if regex.search(row["sentence"]):
                matched.append(
                    {"point": row["sentence"][:160], "evidence": row["sentence"]}
                )
            if len(matched) >= limit:
                break
        return matched

    outlook = keyword_matches(
        "mdna",
        re.compile(r"\b(expect|anticipat|outlook|guidance|will continue|intend)", re.I),
        4,
    )
    drivers = keyword_matches(
        "mdna",
        re.compile(r"\b(due to|driven by|primarily|demand|growth in)\b", re.I),
        3,
    )
    
    tone, mean_net = _tone_from_scores(section_analyses)

    return _normalise(
        {
            "financial_highlights": sentences_from("mdna", "positive", 3),
            "business_drivers": drivers or sentences_from("mdna", "positive", 3),
            "management_outlook": outlook,
            "material_risks": sentences_from("risk_factors", "negative", 3)
            or sentences_from("mdna", "negative", 3),
            "overall_assessment": {
                "tone": tone,
                "rationale": (
                    "Heuristic assessment from FinBERT section scores "
                    f"(mdna sentence-weighted score {mean_net:+.2f}). Set an LLM API key "
                    "in .env for a full LLM-generated summary."
                ),
            },
        }
    )


def _call_gemini(system: str, prompt: str) -> str:
    import google.generativeai as genai

    genai.configure(api_key=config.GOOGLE_API_KEY)
    model = genai.GenerativeModel(
        config.GEMINI_MODEL,
        generation_config={
            "temperature": 0, # Default temperature=0
            "response_mime_type": "application/json",
        },
    )
    return model.generate_content(f"{system}\n\n{prompt}").text


def _call_anthropic(system: str, prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        # Full summaries (4 categories × several quoted items) run long; a
        # tight cap truncates the JSON mid-response and breaks parsing.
        max_tokens=4096,
        temperature=0, # Default temperature=0
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def _call_openai(system: str, prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    try:
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=messages,
            temperature=0, # Default temperature=0
            response_format={"type": "json_object"},
        )
    except Exception:
        # Some OpenAI-compatible providers reject response_format.
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL, messages=messages, temperature=0 # Default temperature=0
        )
    return response.choices[0].message.content


_PROVIDER_CALLS = {
    "gemini": _call_gemini,
    "anthropic": _call_anthropic,
    "openai": _call_openai,
}


def call_llm(system: str, prompt: str) -> str:
    """Send a prompt to the active provider and return raw text.

    Raises KeyError if called in mock mode (no provider configured).
    """
    return _PROVIDER_CALLS[config.active_provider()](system, prompt)


def generate_summary(
    meta: dict, sections_text: dict[str, str], section_analyses: dict[str, dict]
) -> dict:
    """Produce a schema-conformant executive summary.

    Returns the summary dict plus provenance fields: 'generated_by'
    ('provider:model' or 'finbert-heuristic-mock') and, if the LLM failed,
    'llm_error'. Any provider error degrades to the mock summary so the app
    never crashes.
    """
    provider = config.active_provider()
    if provider == "mock":
        summary = _mock_summary(section_analyses)
        summary["generated_by"] = "finbert-heuristic-mock"
        return summary

    user_prompt = build_user_prompt(meta, sections_text, section_analyses)
    try:
        raw = call_llm(SYSTEM_PROMPT, user_prompt)
        try:
            parsed = _extract_json(raw)
        except SummaryError as parse_exc:
            # One self-repair retry: show the model its own broken output.
            repair_prompt = (
                f"Your previous response was not valid JSON ({parse_exc}). "
                "Return the SAME content as ONE valid JSON object matching "
                "the schema. Escape or replace any double quotes inside "
                f"string values.\n\nPrevious response:\n{raw}"
            )
            raw = call_llm(SYSTEM_PROMPT, repair_prompt)
            parsed = _extract_json(raw)
        summary = _normalise(parsed)
        summary["overall_assessment"]["tone"], _ = _tone_from_scores(section_analyses)
        summary["generated_by"] = f"{provider}:{config.model_for(provider)}"
        return summary
    except Exception as exc:  # noqa: BLE001 — degrade gracefully, never crash the app
        summary = _mock_summary(section_analyses)
        summary["generated_by"] = "finbert-heuristic-mock"
        summary["llm_error"] = f"{provider} call failed: {exc}"
        return summary


if __name__ == "__main__":
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    from sec_edgar import fetch_filing_html, list_10q_filings
    from section_parser import SECTION_TITLES, extract_sentences, html_to_text, split_sections
    from sentiment import aggregate, load_pipeline, score_sentences

    TICKER = "AAPL"
    MAX_SENTENCES = 50
    RESULTS_DIR = Path(__file__).resolve().parent / "test_result" / "summariser"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    print(f"1/3 Fetching and scoring {TICKER} (provider: {config.active_provider()})...")
    filing = list_10q_filings(TICKER, limit=1)[0]
    sections_text = split_sections(html_to_text(fetch_filing_html(filing)))
    nlp = load_pipeline()
    section_analyses = {}
    for key, text in sections_text.items():
        sentences = extract_sentences(text)[:MAX_SENTENCES]
        if sentences:
            scored = score_sentences(nlp, sentences)
            section_analyses[key] = {
                "title": SECTION_TITLES[key],
                "aggregate": aggregate(scored),
                "sentences": scored,
            }

    print("2/3 Generating executive summary...")
    meta = {
        "company": filing.company, "ticker": filing.ticker, "form": filing.form,
        "report_date": filing.report_date, "filing_date": filing.filing_date,
    }
    summary = generate_summary(meta, sections_text, section_analyses)

    print("3/3 Result:")
    print(f"    generated_by : {summary.get('generated_by')}")
    print(f"    llm_error    : {summary.get('llm_error', '(none)')}")
    print(f"    tone         : {summary['overall_assessment']['tone']}")
    for key in ("financial_highlights", "business_drivers", "management_outlook", "material_risks"):
        print(f"    {key:<22}: {len(summary.get(key, []))} items")

    out = RESULTS_DIR / f"{filing.ticker}_{filing.report_date}_summary_{run_ts}.json"
    out.write_text(
        json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(timespec='seconds'),
                    "meta": meta, "summary": summary}, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"\nSaved: {out}")