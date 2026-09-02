
"""
A4 - Prompt construction.

Simple prompt design for small local models such as llama3.2:1b.
The model must answer only from the retrieved BNSS text.
"""

SYSTEM_PROMPT = """You are a BNSS question answering assistant.

Answer the QUESTION using ONLY the BNSS text provided in the CONTEXT.

Follow these rules:

1. If the answer is present in the CONTEXT, answer the question.
2. Do NOT refuse when the answer is present in the CONTEXT.
3. Do NOT use outside legal knowledge.
4. Do NOT copy the full BNSS provision.
5. Do NOT output the section title or the retrieved text.
6. Give a short, direct answer in one or two sentences.
7. Include the relevant citation in this exact format:
[BNSS s.X]
8. Only cite a section that appears in the CONTEXT.
9. If the CONTEXT genuinely does not contain enough information to answer the question, say exactly:
I don't have enough information in the statute to answer that.

Example:

Question:
Can a police officer arrest someone without a warrant?

Answer:
Yes. A police officer may arrest a person without a warrant in the circumstances specified in Section 35. [BNSS s.35]
"""


def build_context_block(retrieved_chunks: list[dict]) -> str:
    parts = []

    for c in retrieved_chunks:
        citation = f"BNSS s.{c['section_number']}"

        if c.get("subsection"):
            citation += f"{c['subsection']}"

        parts.append(
            f"[{citation}]\n"
            f"Title: {c.get('section_title') or 'Untitled'}\n"
            f"Text:\n{c['text']}"
        )

    return "\n\n---\n\n".join(parts)


def build_user_prompt(
    question: str,
    retrieved_chunks: list[dict]
) -> str:

    context = build_context_block(retrieved_chunks)

    return f"""CONTEXT:

{context}

---

QUESTION:

{question}

ANSWER:

Use the BNSS context above.

If the answer is present, answer directly.
Do not copy the context.
Do not refuse.

Your answer must be only one or two sentences and must include a citation such as [BNSS s.35].
"""




