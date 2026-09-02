"""
A4 - Post-generation citation validation guard.

Non-negotiable per the brief: "Post-generation validation: check that every section number
cited in the answer actually appears in the retrieved context. If the model invents s.999,
strip it or regenerate. This guard must exist in code, not just in the prompt."

This is deliberately independent of any prompt wording -- it re-parses the LLM's own output
after the fact and cross-checks it against ground truth (the actual retrieved chunks), so it
catches hallucinated citations even if the model ignores its instructions.
"""
import re

CITATION_RE = re.compile(r"\[BNSS?\s*s\.\s*(\d{1,3}[A-Z]?)(?:\((\d+)\))?\]", re.IGNORECASE)


def extract_cited_sections(answer_text: str) -> list[tuple[str, str | None]]:
    """Returns [(section_number, subsection_or_None), ...] for every citation found."""
    return [(m.group(1), m.group(2)) for m in CITATION_RE.finditer(answer_text)]


def get_valid_section_numbers(retrieved_chunks: list[dict]) -> set[str]:
    return {c["section_number"] for c in retrieved_chunks}


def get_valid_subsections_by_section(retrieved_chunks: list[dict]) -> dict[str, set[str | None]]:
    """Maps section_number -> set of subsection values actually present in retrieved
    context (values look like "(1)"; None means the chunk has no subsection split)."""
    mapping: dict[str, set] = {}
    for c in retrieved_chunks:
        mapping.setdefault(c["section_number"], set()).add(c.get("subsection"))
    return mapping


def validate_and_strip_citations(answer_text: str, retrieved_chunks: list[dict]) -> dict:
    """
    Checks every citation in the answer against what was actually retrieved.
    Returns:
        {
            "cleaned_answer": str,       # answer with hallucinated citations removed
            "valid_citations": [...],    # citations that checked out
            "invalid_citations": [...],  # citations that were stripped (hallucinated)
            "had_hallucination": bool,
        }
    """
    valid_sections = get_valid_section_numbers(retrieved_chunks)
    valid_subsections_by_section = get_valid_subsections_by_section(retrieved_chunks)
    cited = extract_cited_sections(answer_text)

    def _is_valid(section_num: str, subsection: str | None) -> bool:
        if section_num not in valid_sections:
            return False
        if subsection is None:
            return True  # citing just the section (no subsection claim) is always fine
        # cited subsection like "1" must match a retrieved chunk's subsection "(1)" for
        # this section -- catches a fabricated subsection tacked onto an otherwise real section.
        cited_form = f"({subsection})"
        return cited_form in valid_subsections_by_section.get(section_num, set())

    valid_citations = [c for c in cited if _is_valid(*c)]
    invalid_citations = [c for c in cited if not _is_valid(*c)]

    cleaned_answer = answer_text
    for section_num, subsection in invalid_citations:
        # Remove the specific hallucinated citation tag itself. We strip the citation
        # rather than the whole sentence -- removing surrounding prose risks deleting
        # legitimate, differently-cited content that happens to share the sentence.
        sub_pattern = rf"(?:\({re.escape(subsection)}\))?" if subsection else r"(?:\(\d+\))?"
        pattern = re.compile(
            rf"\[BNSS?\s*s\.\s*{re.escape(section_num)}{sub_pattern}\]",
            re.IGNORECASE,
        )
        cleaned_answer = pattern.sub("[citation removed - not found in source material]", cleaned_answer)

    return {
        "cleaned_answer": cleaned_answer,
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations,
        "had_hallucination": len(invalid_citations) > 0,
    }