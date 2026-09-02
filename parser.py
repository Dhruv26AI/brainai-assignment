r"""
Structure-aware ingestion pipeline for the Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS)
bare-act PDF.

Layout facts this parser relies on (verified against the actual PDF via PyMuPDF block
coordinates before writing this):
  - Main body text column:      x0 in ~[57, 480]
  - Marginal headnote column:   x0 in ~[480, 545]  -> informal "section title" for each section
  - Running header/footer:      block matching r"Sec\.\s*\d+\]|GAZETTE OF INDIA|EXTRAORDINARY"
  - Chapter heading:             a block whose text is exactly "CHAPTER <roman numeral>"
                                  immediately followed by a centered all-caps title block
  - Section start:                a body-column block beginning with r"^\d{1,3}\.\s"

See DECISIONS.md for why we chose block-coordinate splitting over pdftotext's reading-order
text and why marginal headnotes are treated as section_title.
"""
import re
import json
import hashlib
import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

import fitz  # PyMuPDF

ACT_NAME = "Bharatiya Nagarik Suraksha Sanhita, 2023"
ACT_SHORT = "BNSS"

# Bare-act headnotes sit in the OUTER margin, which alternates sides on
# verso/recto pages: right margin (x0 > BODY_MAX_X0) on odd-numbered pages,
# left margin (x0 < BODY_MIN_X0) on even-numbered pages. Confirmed by inspecting
# raw block coordinates on both odd and even sample pages before writing this.
BODY_MIN_X0 = 110.0
BODY_MAX_X0 = 480.0
HEADER_FOOTER_RE = re.compile(r"Sec\.\s*\d+\]|GAZETTE OF INDIA|EXTRAORDINARY|^\d{1,4}$")
SECTION_START_RE = re.compile(r"^(\d{1,3}[A-Z]?)\.\s+(.*)", re.DOTALL)
CHAPTER_RE = re.compile(r"^CHAPTER\s+([IVXLCDM]+)\s*$")
# The numbered-section body ends where the Schedules begin (First Schedule = offence
# classification table, Second Schedule = the statutory forms, pages 190-249 -- handled
# separately by the Part B forms pipeline, not by this section parser).
SCHEDULE_START_RE = re.compile(r"^THE\s+[A-Z ]+SCHEDULE\b")
SUBSECTION_RE = re.compile(r"\((\d+)\)")
CLAUSE_RE = re.compile(r"\(([a-z])\)")
PROVISO_RE = re.compile(r"\bProvided\b", re.I)
EXCEPTION_RE = re.compile(r"\bException\b", re.I)
EXPLANATION_RE = re.compile(r"\bExplanation\b", re.I)
ILLUSTRATION_RE = re.compile(r"\bIllustration\b", re.I)
CROSSREF_RE = re.compile(r"section\s+(\d{1,3}[A-Z]?)(\(\d+\))?", re.I)
HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")


@dataclass
class RawSection:
    section_number: str
    title: Optional[str]
    body_text: str
    chapter: Optional[str]
    chapter_title: Optional[str]
    page_start: int
    page_end: int


def _dehyphenate(text: str) -> str:
    return HYPHEN_BREAK_RE.sub(r"\1\2", text)


def extract_page_blocks(page):
    """Return (body_blocks, margin_blocks) for a page, each as list of (y0, text)."""
    raw = page.get_text("blocks")
    body, margin = [], []
    for x0, y0, x1, y1, text, *_ in raw:
        text = text.strip()
        if not text or HEADER_FOOTER_RE.search(text.replace("\n", " ")):
            continue
        if x0 >= BODY_MAX_X0 or x0 < BODY_MIN_X0:
            margin.append((y0, text.replace("\n", " ").strip()))
        else:
            body.append((y0, text))
    body.sort(key=lambda b: b[0])
    margin.sort(key=lambda b: b[0])
    return body, margin


def nearest_margin_title(margin_blocks, y0, used_indices, max_dy=40.0):
    """Find the margin headnote whose y0 is closest to (and not far below) the section start."""
    best_idx, best_dist = None, None
    for i, (my0, text) in enumerate(margin_blocks):
        if i in used_indices:
            continue
        dist = abs(my0 - y0)
        if dist <= max_dy and (best_dist is None or dist < best_dist):
            best_idx, best_dist = i, dist
    if best_idx is not None:
        used_indices.add(best_idx)
        return margin_blocks[best_idx][1]
    return None


def parse_pdf(pdf_path: str) -> list[RawSection]:
    doc = fitz.open(pdf_path)
    sections: list[RawSection] = []

    current_chapter = None
    current_chapter_title = None
    pending_chapter_roman = None  # seen "CHAPTER V", waiting for title block next

    # in-progress section accumulator
    cur_num = None
    cur_title = None
    cur_text_parts: list[str] = []
    cur_page_start = None
    cur_chapter = None
    cur_chapter_title = None

    def flush():
        nonlocal cur_num, cur_title, cur_text_parts, cur_page_start
        if cur_num is not None:
            sections.append(
                RawSection(
                    section_number=cur_num,
                    title=cur_title,
                    body_text=_dehyphenate(" ".join(cur_text_parts)).strip(),
                    chapter=cur_chapter,
                    chapter_title=cur_chapter_title,
                    page_start=cur_page_start,
                    page_end=page_idx + 1,
                )
            )
        cur_num, cur_title, cur_text_parts, cur_page_start = None, None, [], None

    in_schedule = False

    for page_idx in range(len(doc)):
        if in_schedule:
            break  # Schedules (offence table + forms) are out of scope for this parser

        page = doc[page_idx]
        body_blocks, margin_blocks = extract_page_blocks(page)
        used_margin = set()

        i = 0
        while i < len(body_blocks):
            y0, text = body_blocks[i]

            if SCHEDULE_START_RE.match(text.strip()):
                flush()
                in_schedule = True
                break

            # Chapter number line, e.g. "CHAPTER V"
            m_chap = CHAPTER_RE.match(text.strip())
            if m_chap:
                pending_chapter_roman = m_chap.group(1)
                i += 1
                continue

            # Chapter title line immediately follows a pending chapter roman numeral.
            # Heuristic: short-ish, ALL CAPS line, no leading digit.
            if pending_chapter_roman and text.strip().isupper() and not text.strip()[0].isdigit():
                current_chapter = pending_chapter_roman
                current_chapter_title = text.strip().title()
                pending_chapter_roman = None
                i += 1
                continue

            m_sec = SECTION_START_RE.match(text)
            if m_sec:
                # New section begins -> flush whatever was being accumulated
                flush()
                cur_num = m_sec.group(1)
                cur_chapter = current_chapter
                cur_chapter_title = current_chapter_title
                cur_page_start = page_idx + 1
                cur_text_parts.append(m_sec.group(2))
                cur_title = nearest_margin_title(margin_blocks, y0, used_margin)
            else:
                if cur_num is not None:
                    cur_text_parts.append(text)
                # else: preamble / schedule text before first section on this page -> ignore
            i += 1

    flush()
    return sections


def slug_chunk_id(section_number: str, seq: int) -> str:
    return f"bnss-s{section_number}-{seq:03d}"


def split_section_into_chunks(sec: RawSection, max_chars: int = 1600) -> list[dict]:
    """
    Section is the atomic unit: never split if it fits under max_chars.
    If it doesn't fit, split at subsection boundaries only (never mid-sentence),
    and keep provisos/exceptions/explanations/illustrations attached to the
    subsection/clause they follow rather than spun into their own chunk.
    """
    text = sec.body_text
    references = sorted(set(m.group(0) for m in CROSSREF_RE.finditer(text)))

    def flags(t: str) -> dict:
        return {
            "has_illustration": bool(ILLUSTRATION_RE.search(t)),
            "has_proviso": bool(PROVISO_RE.search(t)),
            "has_exception": bool(EXCEPTION_RE.search(t)),
        }

    def split_by_clause(sub_text: str, sub_id: Optional[str]) -> list[dict]:
        """Second-level split for a subsection that's still oversized and has no
        further numbered subsections of its own (e.g. Section 2's definitions:
        one subsection (1), dozens of lettered clauses (a)...(z))."""
        parts = re.split(r"(?=\([a-z]\)\s)", sub_text)
        out_c, buf, buf_clause = [], "", None
        for part in parts:
            m = CLAUSE_RE.match(part.strip())
            clause_id = f"({m.group(1)})" if m else buf_clause
            if buf and len(buf) + len(part) > max_chars:
                out_c.append({"subsection": sub_id, "clause": buf_clause, "text": buf.strip(), **flags(buf)})
                buf, buf_clause = part, clause_id
            else:
                buf += part
                buf_clause = buf_clause or clause_id
        if buf.strip():
            out_c.append({"subsection": sub_id, "clause": buf_clause, "text": buf.strip(), **flags(buf)})
        return out_c

    chunks = []
    if len(text) <= max_chars:
        chunks.append(
            {
                "subsection": None,
                "clause": None,
                "text": text,
                **flags(text),
            }
        )
    else:
        # Level 1: split at subsection boundaries: "(1) ... (2) ... (3) ..."
        parts = re.split(r"(?=\(\d+\)\s)", text)
        buf = ""
        buf_sub = None
        for part in parts:
            m = SUBSECTION_RE.match(part.strip())
            sub_id = f"({m.group(1)})" if m else buf_sub
            if buf and len(buf) + len(part) > max_chars:
                chunks.append({"subsection": buf_sub, "clause": None, "text": buf.strip(), **flags(buf)})
                buf = part
                buf_sub = sub_id
            else:
                buf += part
                buf_sub = buf_sub or sub_id
        if buf.strip():
            chunks.append({"subsection": buf_sub, "clause": None, "text": buf.strip(), **flags(buf)})

        # Level 2: any subsection chunk still over max_chars gets split at clause
        # boundaries instead (never mid-sentence -- clause markers are always
        # sentence/clause-initial, so this is a safe split point).
        refined = []
        for c in chunks:
            if len(c["text"]) > max_chars and CLAUSE_RE.search(c["text"]):
                refined.extend(split_by_clause(c["text"], c["subsection"]))
            else:
                refined.append(c)
        chunks = refined

    out = []
    for seq, c in enumerate(chunks, start=1):
        out.append(
            {
                "act": ACT_NAME,
                "act_short": ACT_SHORT,
                "chapter": sec.chapter,
                "chapter_title": sec.chapter_title,
                "section_number": sec.section_number,
                "section_title": sec.title,
                "subsection": c["subsection"],
                "clause": c["clause"],
                "text": c["text"],
                "has_illustration": c["has_illustration"],
                "has_proviso": c["has_proviso"],
                "has_exception": c["has_exception"],
                "page_start": sec.page_start,
                "page_end": sec.page_end,
                "chunk_id": slug_chunk_id(sec.section_number, seq),
                "source_uri": "data/raw/bnss_bare_act_2023.pdf",
                "references": references,
                "ingested_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        )
    return out


def ingest(pdf_path: str) -> list[dict]:
    sections = parse_pdf(pdf_path)
    all_chunks = []
    for sec in sections:
        all_chunks.extend(split_section_into_chunks(sec))
    return all_chunks


if __name__ == "__main__":
    import sys

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/bnss_bare_act_2023.pdf"
    chunks = ingest(pdf_path)
    print(f"Parsed {len(chunks)} chunks from {len({c['section_number'] for c in chunks})} sections")
    out_path = "data/processed/bnss_chunks.jsonl"
    import os

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Wrote {out_path}")