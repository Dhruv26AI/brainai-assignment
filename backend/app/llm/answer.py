"""
A4 - Answer orchestration: confidence threshold + refusal path + generation + citation guard.

Confidence signal: we use the cross-encoder RERANK score (not the raw RRF fusion score),
because RRF scores are rank-based and not meaningfully comparable to an absolute relevance
notion -- two totally irrelevant results can still have a "top" RRF score. The cross-encoder
(ms-marco-MiniLM-L-6-v2) outputs a relevance logit where scores well below 0 reliably mean
"not actually relevant" in our testing. See DECISIONS.md for the exact threshold value and
how we chose it.
"""

from .prompts import SYSTEM_PROMPT, build_user_prompt
from .citation_guard import validate_and_strip_citations
from .providers import LLMProvider


CONFIDENCE_THRESHOLD = 0.0  # cross-encoder rerank score; see DECISIONS.md

REFUSAL_MESSAGE = (
    "I don't have enough information in the statute to answer that. "
    "This system only answers from the retrieved BNSS text and does not guess."
)


def answer_question(
    question: str,
    retrieval_result: dict,
    llm: LLMProvider,
) -> dict:
    """
    retrieval_result: the dict returned by HybridRetriever.search(), i.e.
        {"results": [...], "retrieval_path": ..., "detected_section": ...}

    Returns a dict with the final answer, whether it was refused, citation validation info,
    and the source chunks for the UI's expandable source panel.
    """

    results = retrieval_result["results"]

    if not results:
        return _refusal(reason="no_results")

    # Direct-lookup results are deterministic exact matches -- always confident by
    # definition, never subject to the similarity threshold.
    if retrieval_result["retrieval_path"] != "direct_lookup":
        top_score = results[0].get("rerank_score", results[0]["score"])

        if top_score < CONFIDENCE_THRESHOLD:
            return _refusal(
                reason="below_confidence_threshold",
                top_score=top_score,
            )

    # Keep the top score so it is available in the final result as well.
    top_score = (
        results[0].get("rerank_score", results[0].get("score"))
        if results
        else None
    )

    chunks = [r["payload"] for r in results]

    user_prompt = build_user_prompt(question, chunks)

    raw_answer = llm.generate(
        SYSTEM_PROMPT,
        user_prompt,
    )

    validation = validate_and_strip_citations(
        raw_answer,
        chunks,
    )

    # Detect a model-generated refusal.
    is_refusal_style = raw_answer.strip().lower().startswith(
        "i don't have enough information"
    )

    # Non-negotiable per the brief: "A chatbot that answers legal questions without
    # citations" is an automatic rejection. If the model produced a substantive answer
    # with ZERO valid citations (not just zero hallucinated ones -- zero total), we do not
    # show it as a normal answer.
    #
    # A refusal-style answer is exempt, since that's a legitimate no-citation response.
    if not validation["valid_citations"] and not is_refusal_style:
        return {
            "answer": (
                "I'm not able to provide a properly cited answer to that question right now. "
                "This system requires every legal statement to carry a citation to the "
                "statute, and the generated response did not include one."
            ),
            "refused": True,
            "refusal_reason": "no_valid_citations_generated",
            "top_score": top_score,
            "retrieval_path": retrieval_result["retrieval_path"],
            "had_hallucination": validation["had_hallucination"],
            "invalid_citations_stripped": validation["invalid_citations"],
            "sources": [],
            "_debug_raw_answer": raw_answer,
        }

    # Model-generated refusal-style response.
    #
    # IMPORTANT:
    # Keep the result schema consistent with every other refusal path.
    # This prevents callers such as ask_bnss.py from getting a KeyError
    # when accessing result["refusal_reason"].
    if is_refusal_style:
        return {
            "answer": validation["cleaned_answer"],
            "refused": True,
            "refusal_reason": "llm_refusal_style",
            "top_score": top_score,
            "retrieval_path": retrieval_result["retrieval_path"],
            "had_hallucination": validation["had_hallucination"],
            "invalid_citations_stripped": validation["invalid_citations"],
            "sources": [],
            "_debug_raw_answer": raw_answer,
        }

    # Normal successful answer.
    return {
        "answer": validation["cleaned_answer"],
        "refused": False,
        "refusal_reason": None,
        "top_score": top_score,
        "retrieval_path": retrieval_result["retrieval_path"],
        "had_hallucination": validation["had_hallucination"],
        "invalid_citations_stripped": validation["invalid_citations"],
        "sources": [
            {
                "act_short": c.get("act_short"),
                "chapter": c.get("chapter"),
                "chapter_title": c.get("chapter_title"),
                "section_number": c["section_number"],
                "section_title": c.get("section_title"),
                "subsection": c.get("subsection"),
                "text": c["text"],
                "page_start": c["page_start"],
                "page_end": c["page_end"],
            }
            for c in chunks
        ],
        "_debug_raw_answer": raw_answer,
    }


def _refusal(
    reason: str,
    top_score: float | None = None,
) -> dict:
    return {
        "answer": REFUSAL_MESSAGE,
        "refused": True,
        "refusal_reason": reason,
        "top_score": top_score,
        "retrieval_path": None,
        "had_hallucination": False,
        "invalid_citations_stripped": [],
        "sources": [],
        "_debug_raw_answer": None,
    }

