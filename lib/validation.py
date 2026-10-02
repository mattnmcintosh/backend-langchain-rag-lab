"""Request validation helpers for POST /api/ask."""

from lib.config import MAX_QUESTION_LENGTH, MIN_QUESTION_LENGTH


def validate_question_payload(payload):
    """Validate the JSON body for POST /api/ask.

    Return:
        (question, None) when valid
        (None, error_dict) when invalid
    """

    if not isinstance(payload, dict):
        return None, {
            "error": "invalid_request",
            "message": "Request payload must be a JSON object."
        }
    
    if "question" not in payload:
        return None, {
            "error": "missing_question",
            "message": "The 'question' field is required."
        }

    question = payload.get("question")

    if not isinstance(question, str):
        return None, {
            "error": "invalid_question",
            "message": "The 'question' field must be a string."
        }

    question = question.strip()

    if not question:
        return None, {
            "error": "empty_question",
            "message": "The 'question' field cannot be blank or whitespace only."
        }

    if len(question) < MIN_QUESTION_LENGTH:
        return None, {
            "error": "short_question",
            "message": f"Question must be at least {MIN_QUESTION_LENGTH} characters long."
        }

    if len(question) > MAX_QUESTION_LENGTH:
        return None, {
            "error": "long_question",
            "message": f"Question must not exceed {MAX_QUESTION_LENGTH} characters."
        }
    
    return question, None