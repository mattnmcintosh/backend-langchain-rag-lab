"""Flask API for the LangChain RAG lab."""

from flask import Flask, jsonify, request

from lib.langchain_rag_service import LangChainServiceError, answer_question
from lib.response_formatter import format_error_response
from lib.validation import validate_question_payload


def create_app():
    """Create and configure the Flask application."""

    app = Flask(__name__)

    @app.post("/api/ask")
    def ask():
        """Accept a question and return a source-backed LangChain RAG response."""

        payload = request.get_json(silent=True)

        question, error = validate_question_payload(payload)
        
        if error:
            error_dict = format_error_response(
                error=error.get("error", "invalid_request"),
                message=error.get("message", "Invalid question payload.")
            )
            return jsonify(error_dict), 400

        try:
            result = answer_question(question)
        except LangChainServiceError as exc:
            error_dict = format_error_response(
                error="langchain_service_error",
                message=str(exc)
            )
            return jsonify(error_dict), 502
        except Exception as exc:
            error_dict = format_error_response(
                error="internal_service_error",
                message=f"An unexpected error occurred: {exc}"
            )
            return jsonify(error_dict), 502

        return jsonify(result), 200

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
