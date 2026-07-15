"""Split a 10-Q filing's HTML into its standard Item sections and sentences.

A 10-Q filing section could be extremely long and thus over FinBERT's token input limit
Sentiment analysis sentence by sentence is helpful in aggregating sectional sentimental score.
"""

from __future__ import annotations

import re
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Modern filings are inline-XBRL XHTML. Let lxml's HTML parser handles them and ignore warning messages
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

_ITEM_PREFIX = r"item\s*{num}\s*[.:\-–—]?\s*"

# Canonical 10-Q sections. Primary patterns match "Item N. Title" on one line;
# fallback patterns match the bare title, for filers (e.g. banks) that only use item numbers in the table of contents.
# "first" anchors pick the earliest surviving candidate (Part I sections),
# "last" the latest (Part II sections, which follow the notes where similar wording can reappear).
SECTION_SPECS: list[dict] = [
    {
        "key": "financial_statements",
        "title": "Part  I, Item  1 — Financial Statements",
        "item": _ITEM_PREFIX.format(num="1")
        + r"(?:condensed\s+)?(?:consolidated\s+)?financial\s+statements",
        "fallback": r"(?:condensed\s+)?(?:consolidated\s+)?financial\s+statements(?:\s*\(unaudited\))?",
        "anchor": "first",
    },
    {
        "key": "mdna",
        "title": "Part  I, Item  2 — Management's Discussion & Analysis",
        "item": _ITEM_PREFIX.format(num="2") + r"management[’']?s?\s+discussion",
        "fallback": r"management[’']?s?\s+discussion\s+and\s+analysis.{0,80}",
        "anchor": "first",
    },
    {
        "key": "market_risk",
        "title": "Part  I, Item  3 — Market Risk Disclosures",
        "item": _ITEM_PREFIX.format(num="3") + r"quantitative\s+and\s+qualitative",
        "fallback": r"quantitative\s+and\s+qualitative\s+disclosures.{0,40}",
        "anchor": "first",
    },
    {
        "key": "controls",
        "title": "Part  I, Item  4 — Controls and Procedures",
        "item": _ITEM_PREFIX.format(num="4") + r"controls\s+and\s+procedures",
        "fallback": None,
        "anchor": "first",
    },
    {
        "key": "legal",
        "title": "Part II, Item  1 — Legal Proceedings",
        "item": _ITEM_PREFIX.format(num="1") + r"legal\s+proceedings",
        "fallback": None,
        "anchor": "last",
    },
    {
        "key": "risk_factors",
        "title": "Part II, Item 1A — Risk Factors",
        "item": _ITEM_PREFIX.format(num="1a") + r"risk\s+factors",
        "fallback": r"risk\s+factors",
        "anchor": "last",
    },
]

SECTION_TITLES = {spec["key"]: spec["title"] for spec in SECTION_SPECS}

# Headings that end the useful content (remaining Part II items, signatures,
# exhibit index). Used only as slice boundaries, never as sections.
_TERMINAL_PATTERNS = [
    _ITEM_PREFIX.format(num="2") + r"unregistered\s+sales",
    _ITEM_PREFIX.format(num="3") + r"defaults?\s+upon",
    _ITEM_PREFIX.format(num="4") + r"mine\s+safety",
    _ITEM_PREFIX.format(num="5") + r"other\s+information",
    _ITEM_PREFIX.format(num="6") + r"exhibits",
    r"signatures?",
]

_MAX_HEADING_CHARS = 160
_PAGE_NUMBER = re.compile(r"^\d{1,3}$")
_BLOCK_CLOSE = re.compile(r"</(?:p|div|td|th|tr|h[1-6]|li)\s*>|<br\s*/?>", re.I)


def html_to_text(html: str) -> str:
    """Extract filing text with one line per block element.

    Block-close tags become newlines BEFORE parsing, then text nodes are
    joined with no separator — so inline spans (which may split words) do not
    fragment a heading across lines.
    """
    html = _BLOCK_CLOSE.sub("\n", html)
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    raw = soup.get_text("")
    lines = (re.sub(r"[ \t\xa0]+", " ", line).strip() for line in raw.split("\n"))
    return "\n".join(line for line in lines if line)


def _candidates(text: str, lowered: str, pattern: str, max_line: int) -> list[int]:
    """Line-start matches of `pattern` that look like real headings.

    Rejects lines longer than `max_line` (prose, not a heading) and matches
    immediately followed by a bare page number — same line or next line —
    which marks a table-of-contents row rather than a real heading.
    """
    positions: list[int] = []
    for match in re.finditer(rf"^(?:part\s+i+[.:]?\s*)?{pattern}", lowered, re.MULTILINE):
        start, end = match.start(), match.end()
        first_line_end = text.find("\n", start)
        first_line_end = first_line_end if first_line_end != -1 else len(text)
        if first_line_end - start > max_line:
            continue
        tail_end = text.find("\n", end)
        tail_end = tail_end if tail_end != -1 else len(text)
        tail = text[end:tail_end].strip()
        next_line = text[tail_end + 1 :].split("\n", 1)[0].strip()
        if _PAGE_NUMBER.match(tail) or _PAGE_NUMBER.match(next_line):
            continue
        positions.append(start)
    return positions


def split_sections(text: str) -> dict[str, str]:
    """Locate Item headings and slice the filing into named sections.

    Sections whose headings cannot be located are absent from the result —
    callers must tolerate partial output (a documented limitation for
    non-standard filers).
    """
    lowered = text.lower()
    found: list[tuple[int, str]] = []
    for spec in SECTION_SPECS:
        positions = _candidates(text, lowered, spec["item"], _MAX_HEADING_CHARS)
        if not positions and spec["fallback"]:
            positions = _candidates(text, lowered, spec["fallback"], 90)
        if positions:
            pos = positions[0] if spec["anchor"] == "first" else positions[-1]
            found.append((pos, spec["key"]))

    boundaries = [pos for pos, _ in found]
    for pattern in _TERMINAL_PATTERNS:
        boundaries.extend(_candidates(text, lowered, pattern, 90))
    boundaries.append(len(text))

    found.sort()
    sections: dict[str, str] = {}
    for start, key in found:
        end = min(b for b in boundaries if b > start)
        body = text[start:end].strip()
        if body:
            sections[key] = body
    return sections


# Common abbreviations that would otherwise break sentence splitting.
_ABBREVIATIONS = [
    "U.S.", "U.K.", "No.", "Inc.", "Corp.", "Co.", "Ltd.", "L.P.",
    "Mr.", "Ms.", "Dr.", "vs.", "approx.", "e.g.", "i.e.", "et al.",
]

# Realised headings are cut into sentences.
# Heading normally do not contain terminal punctuations
_SENTENCE_END = re.compile(r"[.!?][\"”’')\]]*$")


def extract_sentences(text: str, min_words: int = 5, max_chars: int = 600) -> list[str]:
    """Split section text into sentences suitable for FinBERT scoring.

    Since html_to_text yields one block (paragraph, heading, or table cell) per line, sentences never span lines
    Splitting per line keeps bare subsection headings from bleeding into the following sentence.
    Filters out table fragments: too few alphabetic words, or mostly digits.
    """
    pieces: list[str] = []
    for line in text.split("\n"):
        for i, abbr in enumerate(_ABBREVIATIONS):
            line = line.replace(abbr, f"\x00{i}\x00")
        pieces.extend(re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])", line))
    sentences: list[str] = []
    for piece in pieces:
        for i, abbr in enumerate(_ABBREVIATIONS):
            piece = piece.replace(f"\x00{i}\x00", abbr)
        sentence = piece.strip()
        if not _SENTENCE_END.search(sentence): # Excluding sentences with no terminal punctuations
            continue
        if len(sentence) > max_chars:
            sentence = sentence[:max_chars].rsplit(" ", 1)[0] + "…"
        words = re.findall(r"[A-Za-z]{2,}", sentence)
        if len(words) < min_words:
            continue
        digit_ratio = sum(c.isdigit() for c in sentence) / len(sentence)
        if digit_ratio > 0.3:
            continue
        sentences.append(sentence)
    return sentences

if __name__ == "__main__":
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    from sec_edgar import fetch_filing_html, list_10q_filings

    TICKER = "AAPL"
    RESULTS_DIR = Path(__file__).resolve().parent / "test_result" / "section_parser"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"1/3 Fetching latest 10-Q for {TICKER}...")
    filing = list_10q_filings(TICKER, limit=1)[0]
    html = fetch_filing_html(filing)
    print(f"    {filing.company} — {filing.label} ({len(html):,} chars)")

    print("2/3 html_to_text + split_sections...")
    text = html_to_text(html)
    sections = split_sections(text)
    print(f"    plain text {len(text):,} chars, {len(sections)} sections found")

    print("3/3 extract_sentences per section...\n")
    print(f"    {'Section':<52} {'Chars':>9} {'Sents':>6}")
    print("    " + "-" * 70)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker": filing.ticker,
        "company": filing.company,
        "report_date": filing.report_date,
        "sections": {},
    }
    for key, body in sections.items():
        sentences = extract_sentences(body)
        print(f"    {SECTION_TITLES[key]:<52} {len(body):>9,} {len(sentences):>6}")
        result["sections"][key] = {
            "title": SECTION_TITLES[key],
            "char_len": len(body),
            "n_sentences": len(sentences),
            "sentences": sentences,
        }

    out = RESULTS_DIR / f"{filing.ticker}_{filing.report_date}_sections.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out}")