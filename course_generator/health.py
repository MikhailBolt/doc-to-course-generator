import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List


def ollama_base_url() -> str:
    return os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")


def list_ollama_models(timeout: float = 5.0) -> List[str]:
    url = f"{ollama_base_url()}/api/tags"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [str(item.get("name", "")).strip() for item in data.get("models", []) if item.get("name")]


def model_is_available(model: str, models: List[str] | None = None) -> bool:
    model = str(model).strip()
    if not model:
        return False
    names = models if models is not None else list_ollama_models()
    for name in names:
        if name == model or name.startswith(f"{model}:") or name.split(":")[0] == model:
            return True
    return False


def check_ollama(model: str, timeout: float = 5.0) -> Dict[str, Any]:
    """Return connectivity + whether the requested model appears in Ollama tags."""
    try:
        models = list_ollama_models(timeout=timeout)
        available = model_is_available(model, models)
        return {
            "ok": True,
            "host": ollama_base_url(),
            "model": model,
            "model_available": available,
            "models_sample": models[:15],
            "models_count": len(models),
        }
    except urllib.error.URLError as exc:
        return {
            "ok": False,
            "host": ollama_base_url(),
            "model": model,
            "model_available": False,
            "error": str(exc.reason if hasattr(exc, "reason") else exc),
        }
    except Exception as exc:
        return {
            "ok": False,
            "host": ollama_base_url(),
            "model": model,
            "model_available": False,
            "error": str(exc),
        }


def check_embeddings(model: str) -> Dict[str, Any]:
    """Verify the embedding model loads and returns a vector (may download on first run)."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name=model)
        vector = embeddings.embed_query("health check")
        return {
            "ok": True,
            "model": model,
            "dimensions": len(vector) if isinstance(vector, list) else 0,
        }
    except Exception as exc:
        return {"ok": False, "model": model, "error": str(exc)}


def format_ollama_message(info: Dict[str, Any]) -> str:
    """Human-readable Ollama status for CLI, Streamlit, and pipeline errors."""
    host = info.get("host", ollama_base_url())
    model = info.get("model", "")

    if not info.get("ok"):
        err = info.get("error", "connection failed")
        return (
            f"Ollama is not reachable at {host}.\n"
            f"  • Start the Ollama app or run: ollama serve\n"
            f"  • Check OLLAMA_HOST in .env if Ollama runs on another machine\n"
            f"  • Detail: {err}"
        )

    if not info.get("model_available"):
        sample = info.get("models_sample") or []
        sample_txt = ", ".join(sample[:5]) if sample else "(none listed)"
        return (
            f"Model '{model}' is not installed in Ollama.\n"
            f"  • Run: ollama pull {model}\n"
            f"  • Or pick another model (--model / UI dropdown)\n"
            f"  • Installed models ({info.get('models_count', len(sample))}): {sample_txt}"
        )

    return f"Ollama OK at {host} — model '{model}' is available."
