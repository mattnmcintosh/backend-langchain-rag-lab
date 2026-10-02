"""Response formatting helpers for the LangChain RAG API."""

CHAIN_EXPRESSION = "ChatPromptTemplate | ChatOllama | StrOutputParser"

COMPONENTS = {
    "vector_store": "Chroma",
    "retrieval_method": "similarity_search_with_score",
    "prompt_template": "ChatPromptTemplate",
    "chat_model": "ChatOllama",
    "output_parser": "StrOutputParser",
}

SCORE_TYPE = "Chroma distance; lower usually means closer in this lesson setup"

FALLBACK_ANSWER = (
    "I do not have enough approved runbook context to answer that reliably."
)


def format_sources(scored_documents):
    """Format retrieved documents as source metadata for the API response."""
    if not isinstance(scored_documents, list):
        return []

    sources = []
    seen_chunk_ids = set()

    for item in scored_documents:
        doc = None
        distance = None
        if isinstance(item, tuple) and len(item) == 2:
            doc, distance = item
        elif isinstance(item, dict):
            doc = item.get("document", item)
            distance = item.get("distance")
        else:
            doc = item

        metadata = getattr(doc, "metadata", {}) if not isinstance(doc, dict) else doc.get("metadata", doc)
        
        source_id = metadata.get("source_id") or "unknown"
        title = metadata.get("title") or "Untitled"
        category = metadata.get("category") or "Uncategorized"
        section = metadata.get("section") or "Unspecified"
        chunk_id = metadata.get("chunk_id") or metadata.get("id") or "unknown"

        if chunk_id in seen_chunk_ids:
            continue
        if chunk_id is not None:
            seen_chunk_ids.add(chunk_id)

        source_entry = {
            "source_id": source_id,
            "title": title,
            "category": category,
            "section": section,
            "chunk_id": chunk_id,
        }
        if distance is not None:
            source_entry["distance"] = round(float(distance), 4)

        sources.append(source_entry)

    return sources


def format_langchain_debug(scored_documents, context, top_k, fallback):
    """Return LangChain debug metadata for inspectability."""
    chunk_ids = []
    if isinstance(scored_documents, list):
        for item in scored_documents:
            doc = item[0] if isinstance(item, tuple) and len(item) == 2 else item
            metadata = getattr(doc, "metadata", {}) if not isinstance(doc, dict) else doc.get("metadata", doc)
            cid = metadata.get("chunk_id") or metadata.get("id")
            if cid:
                chunk_ids.append(cid)

    return {
        "chain_expression": CHAIN_EXPRESSION,
        "components": COMPONENTS,
        "retrieved_count": len(chunk_ids),
        "retrieved_chunk_ids": chunk_ids,
        "top_k": top_k,
        "context_characters": len(context) if isinstance(context, str) else 0,
        "fallback": fallback,
        "score_type": SCORE_TYPE,
    }


def format_success_response(answer, sources, debug):
    """Format a successful RAG response."""

    cleaned_answer = answer.strip() if isinstance(answer, str) else ""
    return {
        "answer": cleaned_answer,
        "sources": sources if isinstance(sources, list) else [],
        "langchain": debug if isinstance(debug, dict) else {},
    }


def format_fallback_response(debug):
    """Format a safe response when no usable context is available."""

    return {
        "answer": FALLBACK_ANSWER,
        "sources": [],
        "langchain": debug if isinstance(debug, dict) else {},
    }


def format_error_response(error, message):
    """Format an API error response."""
    return {
        "error": str(error),
        "message": str(message),
    }
