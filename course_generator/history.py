import json
from pathlib import Path
from typing import Any, Dict, List


def list_recent_reports(output_dir: str, limit: int = 8) -> List[Dict[str, Any]]:
    """Load metadata from the newest generation_report.json files in output_dir."""
    root = Path(output_dir)
    if not root.exists():
        return []

    candidates = list(root.glob("*generation_report.json")) + list(root.glob("generation_report.json"))
    candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)

    reports: List[Dict[str, Any]] = []
    for path in candidates[:limit]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            reports.append({
                "path": str(path),
                "generated_at": data.get("generated_at", ""),
                "generator_version": data.get("generator_version", ""),
                "model": data.get("model", ""),
                "lessons_count": data.get("lessons_count", 0),
                "quality_score": (data.get("quality") or {}).get("overall_score"),
                "grade": (data.get("quality") or {}).get("grade"),
                "elapsed_seconds": data.get("elapsed_seconds"),
            })
        except Exception:
            continue
    return reports


def list_recent_bundles(output_dir: str, limit: int = 8) -> List[Dict[str, Any]]:
    """Load metadata from the newest course_bundle.json files in output_dir."""
    root = Path(output_dir)
    if not root.exists():
        return []

    candidates = list(root.glob("*course_bundle.json")) + list(root.glob("course_bundle.json"))
    candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)

    bundles: List[Dict[str, Any]] = []
    for path in candidates[:limit]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            outline = data.get("outline") if isinstance(data.get("outline"), dict) else {}
            lessons = data.get("lessons", data.get("lesson_payloads", []))
            bundles.append({
                "path": str(path),
                "course_title": outline.get("course_title", ""),
                "lessons_count": len(outline.get("lessons", [])),
                "lesson_payloads_count": len(lessons) if isinstance(lessons, list) else 0,
                "pretest_count": len(data.get("pretest", [])) if isinstance(data.get("pretest"), list) else 0,
                "quiz_count": len(data.get("final_quiz", [])) if isinstance(data.get("final_quiz"), list) else 0,
                "modified_at": path.stat().st_mtime,
            })
        except Exception:
            continue
    return bundles
