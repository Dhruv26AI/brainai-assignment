import json
import hashlib
import subprocess
from pathlib import Path
import fitz

FORMS_DIR = Path("data/forms")
MANIFEST = FORMS_DIR / "forms_manifest.json"
REQUIRED = {
    "form_number", "title", "page_start", "page_end",
    "output_filename", "byte_size", "sha256",
    "extraction_confidence", "needs_review"
}


def hashes():
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in FORMS_DIR.glob("FORM-*.pdf")
    }


def run_extraction():
    subprocess.run(
        ["python", "backend/app/forms/extract_forms.py"],
        check=True,
        capture_output=True,
        text=True,
    )


def test_idempotency():
    run_extraction()
    first = hashes()
    run_extraction()
    assert first == hashes()


def test_filename_collisions():
    assert len(hashes()) == 58
    assert len(set(hashes())) == 58


def test_manifest():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert len(data) == 58
    for form in data:
        assert REQUIRED <= form.keys()
        assert form["output_filename"] in hashes()


def test_form_33():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    form = next(x for x in data if x["form_number"] == "33")
    pdf = fitz.open(FORMS_DIR / form["output_filename"])
    assert len(pdf) == 3
    assert form["page_start"] == 222
    assert form["page_end"] == 224


def test_ocr_fallback():
    source = Path(
        "backend/app/forms/extract_forms.py"
    ).read_text(encoding="utf-8")
    assert "pytesseract" in source
    assert "needs_review" in source