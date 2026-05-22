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
