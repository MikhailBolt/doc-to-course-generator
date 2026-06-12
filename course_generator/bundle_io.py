import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from course_generator.generation import validate_outline


def parse_lesson_indices(spec: Optional[str]) -> List[int]:
    """Parse comma-separated 1-based lesson numbers, e.g. '1,3,5'."""
    if not spec or not str(spec).strip():
        return []
    indices: List[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if part.isdigit():
            indices.append(int(part))
    return sorted(set(n for n in indices if n > 0))


def load_bundle_json(path: str) -> Dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Bundle file not found: {path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Bundle is not a JSON object.")
    if "outline" not in data:
        raise ValueError("Bundle missing 'outline'.")
    data["outline"] = validate_outline(data["outline"])
    lessons = data.get("lessons", data.get("lesson_payloads", []))
    if not isinstance(lessons, list):
        raise ValueError("Bundle 'lessons' must be a list.")
    data["lessons"] = lessons
    for key in ("pretest", "final_quiz"):
        if key in data and not isinstance(data[key], list):
            raise ValueError(f"Bundle '{key}' must be a list.")
    if "documents" in data and not isinstance(data["documents"], list):
        raise ValueError("Bundle 'documents' must be a list.")
    return data


def validate_bundle_file(path: str) -> Dict[str, Any]:
    """Load bundle and return a short validation summary."""
    bundle = load_bundle_json(path)
    outline = bundle["outline"]
    lessons = bundle.get("lessons", [])
    return {
        "ok": True,
        "path": path,
        "course_title": outline.get("course_title", ""),
        "lessons_count": len(outline.get("lessons", [])),
        "lesson_payloads_count": len(lessons),
        "pretest_count": len(bundle.get("pretest", [])),
        "quiz_count": len(bundle.get("final_quiz", [])),
        "documents_count": len(bundle.get("documents", [])),
    }
