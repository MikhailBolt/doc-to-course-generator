import json
from pathlib import Path
from typing import Any, Dict


def _load_report(path: str) -> Dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Report not found: {path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid report JSON: {path}")
    return data


def diff_generation_reports(path_a: str, path_b: str) -> Dict[str, Any]:
    """Compare two generation_report.json files."""
    a = _load_report(path_a)
    b = _load_report(path_b)

    def q(report: Dict[str, Any]) -> Dict[str, Any]:
        quality = report.get("quality") or {}
        return {
            "generated_at": report.get("generated_at"),
            "generator_version": report.get("generator_version"),
            "model": report.get("model"),
            "lessons_count": report.get("lessons_count"),
            "quality_score": quality.get("overall_score"),
            "grade": quality.get("grade"),
            "elapsed_seconds": report.get("elapsed_seconds"),
            "lessons_fallback_count": report.get("lessons_fallback_count"),
        }

    left = q(a)
    right = q(b)
    deltas: Dict[str, Any] = {}
    for key in left:
        if left.get(key) != right.get(key):
            deltas[key] = {"a": left.get(key), "b": right.get(key)}

    return {"report_a": path_a, "report_b": path_b, "a": left, "b": right, "deltas": deltas}
