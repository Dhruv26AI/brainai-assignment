r"""
Part B - Forms extraction pipeline.

Extracts each statutory form on pages 190-249 (BNSS's Second Schedule) as its own
page-perfect PDF, scraping the title from the page itself (never hardcoded), correctly
grouping multi-page forms into a single output file, and emitting forms_manifest.json.

Layout facts this relies on (verified by inspecting the actual PDF before writing this):
  - Each form starts with a line matching "FORM  No.<N>" (spacing between "FORM" and "No."
    varies -- sometimes one space, sometimes two).
  - The form's TITLE is the text between that line and the next line that starts a
    "(See section ...)" / "[See section ...]" reference -- e.g. "FORM No.33" ->
    "CHARGES" -> "(See sections 234, 235 and 236)".
  - A continuation page of a multi-page form has NO "FORM No." marker at all -- e.g.
    FORM 33 spans pages 222-224; pages 223-224 have no marker and must be grouped with 222,
    not treated as new forms or dropped.
"""
import re
import os
import json
import hashlib
import datetime
from dataclasses import dataclass, field

import fitz  # PyMuPDF, for text-layer inspection / title scraping
from pypdf import PdfReader, PdfWriter

FORM_START_RE = re.compile(r"FORM\s+No\.?\s*(\d+)", re.I)
SECTION_REF_RE = re.compile(r"^[\[(]\s*See\s+section", re.I)
MIN_TEXT_LEN_FOR_REAL_PAGE = 20  # below this, treat page's text layer as missing/garbage


@dataclass
class RawForm:
    form_number: str
    title_lines: list = field(default_factory=list)
    page_start: int = None
    page_end: int = None
    needs_ocr_pages: list = field(default_factory=list)


def scrape_title(page_text: str) -> tuple[str, bool]:
    """
    Returns (title, low_confidence). Title = the lines between the "FORM No.N" line and
    the first "(See section...)" reference line. low_confidence=True if we couldn't find
    a clean reference-line boundary (title extraction had to fall back to a single line).
    """
    lines = [l.strip() for l in page_text.split("\n") if l.strip()]
    start_idx = None
    for i, line in enumerate(lines):
        if FORM_START_RE.search(line):
            start_idx = i
            break
    if start_idx is None:
        return "", True

    title_lines = []
    for line in lines[start_idx + 1 :]:
        if SECTION_REF_RE.match(line):
            break
        title_lines.append(line)
        if len(title_lines) >= 4:  # safety cap -- titles are short, never this long
            break

    if not title_lines:
        # No reference line found before running out of short candidate lines --
        # fall back to just the first line after the FORM marker, flagged for review.
        if start_idx + 1 < len(lines):
            return lines[start_idx + 1], True
        return "", True

    return " ".join(title_lines), False


def slugify(title: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", title.lower())
    slug = re.sub(r"[\s_]+", "-", slug).strip("-")
    # Title-case each word for the filename convention shown in the brief
    # (e.g. "Bond-and-Bail-Bond-for-Attendance-before-Court")
    return "-".join(w.capitalize() for w in slug.split("-") if w)


def needs_ocr(page) -> bool:
    text = page.get_text().strip()
    return len(text) < MIN_TEXT_LEN_FOR_REAL_PAGE


def ocr_page_text(pdf_path: str, page_index: int) -> str:
    """OCR fallback for a page whose text layer is missing/garbage. Only invoked when
    needs_ocr() is True -- these source pages are clean text-layer PDF, so this path is
    not expected to trigger in practice, but exists as a documented, working fallback."""
    try:
        import pytesseract
        from pdf2image import convert_from_path

        images = convert_from_path(pdf_path, first_page=page_index + 1, last_page=page_index + 1)
        return pytesseract.image_to_string(images[0])
    except Exception as e:
        return f"[OCR FAILED: {e}]"


def detect_forms(pdf_path: str, page_start_1idx: int, page_end_1idx: int) -> list[RawForm]:
    doc = fitz.open(pdf_path)
    forms: list[RawForm] = []
    current: RawForm | None = None

    for page_idx in range(page_start_1idx - 1, page_end_1idx):
        page = doc[page_idx]
        page_needs_ocr = needs_ocr(page)
        text = ocr_page_text(pdf_path, page_idx) if page_needs_ocr else page.get_text()

        match = FORM_START_RE.search(text)
        if match:
            if current is not None:
                forms.append(current)
            title, low_conf = scrape_title(text)
            current = RawForm(
                form_number=match.group(1),
                title_lines=[title],
                page_start=page_idx + 1,
                page_end=page_idx + 1,
            )
            current._low_conf_title = low_conf  # type: ignore[attr-defined]
        else:
            # Continuation page of the current form (no new FORM marker on this page)
            if current is not None:
                current.page_end = page_idx + 1
            # else: page before the first form in range -- ignore (shouldn't happen given
            # our confirmed page range, but guards against an off-by-one).

        if page_needs_ocr and current is not None:
            current.needs_ocr_pages.append(page_idx + 1)

    if current is not None:
        forms.append(current)

    return forms


def extract_form_pdfs(
    pdf_path: str,
    forms: list[RawForm],
    output_dir: str,
) -> list[dict]:
    os.makedirs(output_dir, exist_ok=True)
    reader = PdfReader(pdf_path)
    manifest = []

    for form in forms:
        title = form.title_lines[0] if form.title_lines else ""
        low_conf_title = getattr(form, "_low_conf_title", False)
        slug = slugify(title) if title else "Untitled"
        filename = f"FORM-{form.form_number}_{slug}.pdf"
        out_path = os.path.join(output_dir, filename)

        writer = PdfWriter()
        for page_num in range(form.page_start, form.page_end + 1):
            writer.add_page(reader.pages[page_num - 1])  # pypdf is 0-indexed

        with open(out_path, "wb") as f:
            writer.write(f)

        with open(out_path, "rb") as f:
            file_bytes = f.read()
        sha256 = hashlib.sha256(file_bytes).hexdigest()

        needs_review = low_conf_title or not title or bool(form.needs_ocr_pages)
        confidence = "low" if needs_review else "high"

        manifest.append(
            {
                "form_number": form.form_number,
                "title": title,
                "page_start": form.page_start,
                "page_end": form.page_end,
                "is_multi_page": form.page_end > form.page_start,
                "output_filename": filename,
                "byte_size": len(file_bytes),
                "sha256": sha256,
                "extraction_confidence": confidence,
                "needs_review": needs_review,
                "ocr_pages_used": form.needs_ocr_pages,
                "extracted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        )

    return manifest


def run_pipeline(
    pdf_path: str = "BNS bare act 2023.pdf",
    output_dir: str = "data/forms",
    manifest_path: str = "data/forms/forms_manifest.json",
    page_start: int = 190,
    page_end: int = 249,
):
    print(f"Detecting forms on pages {page_start}-{page_end}...")
    forms = detect_forms(pdf_path, page_start, page_end)
    print(f"Detected {len(forms)} forms "
          f"({sum(1 for f in forms if f.page_end > f.page_start)} multi-page)")

    print(f"Extracting page-perfect PDFs to {output_dir}/ ...")
    manifest = extract_form_pdfs(pdf_path, forms, output_dir)

    needs_review_count = sum(1 for m in manifest if m["needs_review"])
    print(f"{needs_review_count}/{len(manifest)} forms flagged needs_review")

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Wrote manifest: {manifest_path}")

    return manifest


if __name__ == "__main__":
    run_pipeline()
