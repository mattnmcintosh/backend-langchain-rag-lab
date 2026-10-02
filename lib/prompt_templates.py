"""Prompt template helpers for the LangChain RAG workflow."""

from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """
You are an internal reliability assistant.

Answer using only the approved retrieved context provided by the backend.
If the context does not contain enough information to answer reliably, say that
you do not have enough approved context to answer. Do not invent policies,
procedures, metrics, or incident steps.

Keep the response concise, specific, and useful to an engineer during an incident.
"""


def build_rag_prompt():
    """Build the reusable LangChain prompt template for RAG answers."""

    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT.strip()),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
        ]
    )
