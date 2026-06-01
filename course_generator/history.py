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
