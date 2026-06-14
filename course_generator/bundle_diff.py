import json
from pathlib import Path
from typing import Any, Dict, List

from course_generator.bundle_io import load_bundle_json


def _lesson_titles(bundle: Dict[str, Any]) -> List[str]:
    outline = bundle.get("outline") or {}
    lessons = outline.get("lessons") or []
    return [str(item.get("title", "")).strip() for item in lessons if isinstance(item, dict)]


def diff_course_bundles(path_a: str, path_b: str) -> Dict[str, Any]:
    """Compare two course_bundle.json files."""
    a = load_bundle_json(path_a)
    b = load_bundle_json(path_b)

    def summary(bundle: Dict[str, Any], path: str) -> Dict[str, Any]:
        outline = bundle.get("outline") or {}
        return {
            "path": path,
            "generated_at": bundle.get("generated_at"),
            "generator_version": bundle.get("generator_version"),
            "course_title": outline.get("course_title", ""),
            "lessons_count": len(outline.get("lessons", [])),
            "lesson_payloads_count": len(bundle.get("lessons", [])),
            "pretest_count": len(bundle.get("pretest", [])),
            "quiz_count": len(bundle.get("final_quiz", [])),
            "lesson_titles": _lesson_titles(bundle),
        }

    left = summary(a, path_a)
    right = summary(b, path_b)

    deltas: Dict[str, Any] = {}
    for key in ("course_title", "lessons_count", "lesson_payloads_count", "pretest_count", "quiz_count"):
        if left.get(key) != right.get(key):
            deltas[key] = {"a": left.get(key), "b": right.get(key)}

    title_changes: List[Dict[str, str]] = []
    for idx, (ta, tb) in enumerate(zip(left["lesson_titles"], right["lesson_titles"]), start=1):
        if ta != tb:
            title_changes.append({"lesson": idx, "a": ta, "b": tb})

    if len(left["lesson_titles"]) != len(right["lesson_titles"]):
        deltas["lesson_title_count"] = {
            "a": len(left["lesson_titles"]),
            "b": len(right["lesson_titles"]),
        }

    return {
        "bundle_a": path_a,
        "bundle_b": path_b,
        "a": {k: v for k, v in left.items() if k != "lesson_titles"},
        "b": {k: v for k, v in right.items() if k != "lesson_titles"},
        "deltas": deltas,
        "lesson_title_changes": title_changes,
    }
