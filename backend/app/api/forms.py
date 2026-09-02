"""
Part B - Forms API router.

Exposes the 4 required endpoints over the manifest + extracted PDFs produced by
extract_forms.py. Reads the manifest fresh on each request (cheap -- it's a small JSON
file) rather than caching, so it always reflects the latest extraction run.
"""
import io
import json
import os
import zipfile

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

router = APIRouter(prefix="/api/v1/forms", tags=["forms"])

FORMS_DIR = "data/forms"
MANIFEST_PATH = os.path.join(FORMS_DIR, "forms_manifest.json")


def _load_manifest() -> list[dict]:
    if not os.path.exists(MANIFEST_PATH):
        raise HTTPException(
            status_code=503,
            detail="Forms have not been extracted yet. Run backend/app/forms/extract_forms.py first.",
        )
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_form(manifest: list[dict], form_id: str) -> dict:
    match = next((m for m in manifest if m["form_number"] == form_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"Form {form_id} not found")
    return match


@router.get("")
def list_forms():
    """GET /api/v1/forms -- list with title, form number, page range, size."""
    manifest = _load_manifest()
    return [
        {
            "form_number": m["form_number"],
            "title": m["title"],
            "page_start": m["page_start"],
            "page_end": m["page_end"],
            "byte_size": m["byte_size"],
            "needs_review": m["needs_review"],
        }
        for m in sorted(manifest, key=lambda x: int(x["form_number"]))
    ]


@router.get("/search")
def search_forms(q: str = ""):
    """GET /api/v1/forms/search?q= -- case-insensitive substring title search."""
    manifest = _load_manifest()
    q_lower = q.strip().lower()
    if not q_lower:
        return []
    results = [m for m in manifest if q_lower in m["title"].lower()]
    return [
        {
            "form_number": m["form_number"],
            "title": m["title"],
            "page_start": m["page_start"],
            "page_end": m["page_end"],
        }
        for m in sorted(results, key=lambda x: int(x["form_number"]))
    ]


@router.get("/download-all")
def download_all_forms():
    """GET /api/v1/forms/download-all -- zip of every extracted form PDF."""
    manifest = _load_manifest()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for m in manifest:
            file_path = os.path.join(FORMS_DIR, m["output_filename"])
            if os.path.exists(file_path):
                zf.write(file_path, arcname=m["output_filename"])
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=bnss_forms.zip"},
    )


@router.get("/{form_id}/download")
def download_form(form_id: str):
    """GET /api/v1/forms/{id}/download -- single form PDF."""
    manifest = _load_manifest()
    form = _find_form(manifest, form_id)
    file_path = os.path.join(FORMS_DIR, form["output_filename"])
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Form file missing on disk")
    return FileResponse(file_path, media_type="application/pdf", filename=form["output_filename"])
