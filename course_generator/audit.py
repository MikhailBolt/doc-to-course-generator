from pathlib import Path
from typing import Dict, List


def audit_output_paths(paths: Dict[str, str], *, required: List[str] | None = None) -> List[str]:
    """Return human-readable issues for missing expected output files."""
    required = required or ["course_html", "outline", "report"]
    issues: List[str] = []
    for key in required:
        file_path = paths.get(key)
        if not file_path:
            issues.append(f"Missing path entry: {key}")
            continue
        if not Path(file_path).is_file():
            issues.append(f"File not found [{key}]: {file_path}")
    return issues
