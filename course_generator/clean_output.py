from pathlib import Path
from typing import Any, Dict, List, Tuple

# Group keyed by logical artifact name; files match *{suffix}.
ARTIFACT_SUFFIXES: Tuple[str, ...] = (
    "course_bundle.json",
    "generation_report.json",
    "run_manifest.json",
    "course_summary.json",
    "course.html",
    "course_outline.json",
    "course_delivery.zip",
    "OUTPUT_INDEX.md",
    "course_full.md",
    "course_summary.md",
    "flashcards.json",
    "flashcards_anki.txt",
    "quizzes.csv",
    "quizzes.gift",
    "quiz.json",
    "pretest.json",
    "lesson_summaries.json",
    "course_metadata.json",
    "course_summary.docx",
    "course_summary.pdf",
)


def _group_artifacts(output_dir: Path) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {suffix: [] for suffix in ARTIFACT_SUFFIXES}
    if not output_dir.is_dir():
        return groups
    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        name = path.name
        for suffix in ARTIFACT_SUFFIXES:
            if name == suffix or name.endswith(f"_{suffix}"):
                groups[suffix].append(path)
                break
    for suffix in groups:
        groups[suffix] = sorted(groups[suffix], key=lambda p: p.stat().st_mtime, reverse=True)
    return groups


def plan_clean_output(output_dir: str, *, keep_last: int = 1) -> Dict[str, Any]:
    """Return files to delete vs keep. keep_last=0 deletes all matched artifacts."""
    root = Path(output_dir)
    keep_last = max(0, int(keep_last))
    groups = _group_artifacts(root)
    keep: List[str] = []
    delete: List[str] = []
    for paths in groups.values():
        if keep_last == 0:
            delete.extend(str(p) for p in paths)
            continue
        keep.extend(str(p) for p in paths[:keep_last])
        delete.extend(str(p) for p in paths[keep_last:])
    return {
        "output_dir": str(root),
        "keep_last": keep_last,
        "keep_count": len(keep),
        "delete_count": len(delete),
        "keep": keep,
        "delete": delete,
    }


def clean_output_dir(output_dir: str, *, keep_last: int = 1, dry_run: bool = True) -> Dict[str, Any]:
    """Clean generator artifacts. dry_run=True only plans deletions."""
    plan = plan_clean_output(output_dir, keep_last=keep_last)
    plan["dry_run"] = dry_run
    deleted: List[str] = []
    errors: List[str] = []
    if not dry_run:
        for path_str in plan["delete"]:
            path = Path(path_str)
            try:
                path.unlink()
                deleted.append(path_str)
            except OSError as exc:
                errors.append(f"{path_str}: {exc}")
    plan["deleted"] = deleted
    plan["errors"] = errors
    plan["ok"] = len(errors) == 0
    return plan
