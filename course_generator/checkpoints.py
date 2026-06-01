import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional

from course_generator import __version__
from course_generator.io import sanitize_filename_part


def checkpoint_path_for_args(args: Namespace) -> Path:
    prefix = sanitize_filename_part(args.output_prefix) if args.output_prefix else "default"
    return Path(args.output_dir) / ".checkpoints" / prefix / "checkpoint.json"


def save_checkpoint(
    path: Path,
    *,
    outline: Dict[str, Any],
    lesson_payloads: List[Dict[str, Any]],
    outline_rag_used: bool,
    stage: str,
    args: Namespace,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "generator_version": __version__,
        "stage": stage,
        "outline": outline,
        "lesson_payloads": lesson_payloads,
        "lessons_completed": len(lesson_payloads),
        "outline_rag_used": outline_rag_used,
        "docs_path": args.docs_path,
        "model": args.model,
        "language": args.language,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(path)


def load_checkpoint(path: str) -> Dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "outline" not in data:
        raise ValueError("Invalid checkpoint file: missing outline")
    return data
