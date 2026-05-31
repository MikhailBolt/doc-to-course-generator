from course_generator.health import format_ollama_message


def test_format_ollama_unreachable():
    msg = format_ollama_message({"ok": False, "host": "http://127.0.0.1:11434", "error": "refused"})
    assert "ollama serve" in msg.lower()
    assert "refused" in msg


def test_format_ollama_missing_model():
    msg = format_ollama_message(
        {
            "ok": True,
            "host": "http://127.0.0.1:11434",
            "model": "llama3",
            "model_available": False,
            "models_sample": ["gemma3:4b"],
            "models_count": 1,
        }
    )
    assert "ollama pull llama3" in msg
    assert "gemma3" in msg


def test_format_ollama_ok():
    msg = format_ollama_message(
        {"ok": True, "host": "http://127.0.0.1:11434", "model": "llama3", "model_available": True}
    )
    assert "OK" in msg
