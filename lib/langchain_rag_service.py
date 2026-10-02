"""LangChain-supported RAG workflow service."""

from lib import prompt_templates    
from lib.config import CHAT_MODEL, DEFAULT_TOP_K
from lib.response_formatter import (
    format_fallback_response,
    format_langchain_debug,
    format_sources,
    format_success_response,
)
from lib.vector_store import retrieve_context

from langchain_ollama import ChatOllama
from lib.prompt_templates import build_rag_prompt
import langchain_core.output_parsers

from lib.config import CHAT_MODEL, DEFAULT_TOP_K
from lib.response_formatter import (
    format_fallback_response,
    format_langchain_debug,
    format_sources,
    format_success_response,
)
from lib.vector_store import retrieve_context


class LangChainServiceError(Exception):
    """Raised when the LangChain RAG service cannot complete a request."""


def build_chat_model():
    """Build the local chat model wrapper."""
    return ChatOllama(model=CHAT_MODEL, temperature=0)


def build_chain():
    """Build the LangChain prompt to model to parser sequence."""
    prompt_template = prompt_templates.build_rag_prompt()
    llm = build_chat_model()
    return prompt_template | llm | langchain_core.output_parsers.StrOutputParser()


def has_usable_context(scored_documents):
    """Return True when retrieval produced at least one document with text."""

    if not isinstance(scored_documents, list) or not scored_documents:
        return False

    for item in scored_documents:
        doc = item[0] if isinstance(item, tuple) and len(item) == 2 else item
        content = getattr(doc, "page_content", None) if not isinstance(doc, dict) else (doc.get("page_content") or doc.get("text"))
        if isinstance(content, str) and content.strip():
            return True

    return False


def format_context(scored_documents):
    """Format retrieved LangChain documents into prompt-ready context text."""

    if not isinstance(scored_documents, list):
        return ""

    formatted_chunks = []
    for idx, item in enumerate(scored_documents, start=1):
        doc = item[0] if isinstance(item, tuple) and len(item) == 2 else item
        distance = item[1] if isinstance(item, tuple) and len(item) == 2 else None

        if doc is None:
            continue

        metadata = getattr(doc, "metadata", {}) if not isinstance(doc, dict) else doc.get("metadata", doc)
        text = getattr(doc, "page_content", None) if not isinstance(doc, dict) else (doc.get("page_content") or doc.get("text", ""))

        if not isinstance(text, str) or not text.strip():
            continue

        source_id = metadata.get("source_id", "unknown")
        title = metadata.get("title", "Untitled")
        category = metadata.get("category", "Uncategorized")
        section = metadata.get("section", "Unspecified")
        chunk_id = metadata.get("chunk_id") or metadata.get("id", "unknown")
        dist_str = f"{distance:.4f}" if distance is not None else "N/A"

        chunk_str = (
            f"[Context {idx}]\n"
            f"Source ID: {source_id}\n"
            f"Title: {title}\n"
            f"Category: {category}\n"
            f"Section: {section}\n"
            f"Chunk ID: {chunk_id}\n"
            f"Distance: {dist_str}\n"
            f"Content:\n{text.strip()}"
        )
        formatted_chunks.append(chunk_str)

    return "\n\n".join(formatted_chunks)


def answer_question(
    question,
    *,
    vector_store=None,
    chain=None,
    top_k=DEFAULT_TOP_K,
):
    """Run the LangChain-supported RAG workflow for one validated question."""

    if not isinstance(question, str) or not question.strip():
        raise LangChainServiceError("Question must be a non-empty string.")

    cleaned_question = question.strip()

    try:
        scored_documents = retrieve_context(cleaned_question, vector_store=vector_store, top_k=top_k)
    except Exception as exc:
        raise LangChainServiceError(f"Vector store unavailable: {exc}") from exc

    has_context = has_usable_context(scored_documents)
    context_text = format_context(scored_documents) if has_context else ""

    debug_info = format_langchain_debug(scored_documents, context_text, top_k, not has_context)

    if not has_context:
        return format_fallback_response(debug_info)

    rag_chain = chain if chain is not None else build_chain()

    try:
        raw_answer = rag_chain.invoke({"context": context_text, "question": cleaned_question})
    except Exception as exc:
        raise LangChainServiceError(f"LangChain execution failed: {exc}") from exc

    if not isinstance(raw_answer, str) or not raw_answer.strip():
        raise LangChainServiceError("Model returned empty answer.")

    answer = raw_answer.strip()
    sources = format_sources(scored_documents)
    return format_success_response(answer, sources, debug_info)
