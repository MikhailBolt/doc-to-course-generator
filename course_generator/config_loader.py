import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Set

# Keys that may appear in a generation config JSON file (underscore form).
CONFIG_KEYS: Set[str] = {
    "docs_path",
    "db",
    "manifest_file",
    "output_dir",
    "log_dir",
    "embedding_model",
    "model",
    "chunk_size",
    "chunk_overlap",
    "top_k",
    "quiz_questions",
    "pretest_questions",
    "difficulty",
    "retrieval_type",
    "language",
    "max_preview_chars_per_file",
    "output_prefix",
    "min_lessons",
    "max_lessons",
    "disable_review_pass",
    "skip_pretest",
    "skip_final_quiz",
    "include_source_excerpts",
    "rebuild",
    "skip_outline_rag",
    "outline_rag_max_chunks",
    "outline_rag_max_chars",
    "no_delivery_zip",
    "export_docx",
    "export_pdf",
    "export_quiz_csv",
    "export_flashcards",
    "quality_llm_review",
    "recursive_docs",
    "outline_only",
    "from_outline",
    "checkpoint",
    "resume_checkpoint",
    "ollama_timeout",
    "max_files",
    "preset",
}


def load_config_file(path: str) -> Dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config file must be a JSON object.")
    unknown = set(data.keys()) - CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(sorted(unknown))}")
    return data


def apply_config_file(args: Namespace, path: str, defaults: Namespace) -> None:
    """Apply config values only where the CLI left argparse defaults unchanged."""
    for key, value in load_config_file(path).items():
        if not hasattr(args, key):
            continue
        if getattr(args, key) == getattr(defaults, key):
            setattr(args, key, value)
