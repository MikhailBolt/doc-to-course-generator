from course_generator.health import check_ollama, format_ollama_message


def require_ollama(model: str, timeout: float = 10.0) -> None:
    """Raise RuntimeError with actionable help if Ollama is down or the model is missing."""
    info = check_ollama(model, timeout=timeout)
    if not info.get("ok") or not info.get("model_available"):
        raise RuntimeError(format_ollama_message(info))
