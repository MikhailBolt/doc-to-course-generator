from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from course_generator import __version__


def build_output_index_markdown(
    paths: Dict[str, str],
    *,
    course_title: str = "",
    quality: Dict[str, Any] | None = None,
    elapsed_seconds: float | None = None,
) -> str:
    """Human-readable index of generated artifacts."""
    lines = [
        "# Course output index",
        "",
        f"- **Generator:** Doc-to-Course Generator v{__version__}",
        f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    if course_title:
        lines.append(f"- **Course:** {course_title}")
    if elapsed_seconds is not None:
        lines.append(f"- **Duration:** {elapsed_seconds:.1f}s")
    if quality:
        lines.append(
            f"- **Quality:** {quality.get('overall_score', '—')}/100 (grade {quality.get('grade', '—')})"
        )
    lines.extend(["", "## Files", ""])

    descriptions = {
        "course_html": "Interactive HTML course (open in browser)",
        "outline": "Course structure JSON",
        "pretest": "Pre-test questions JSON",
        "quiz": "Final quiz JSON",
        "summaries": "Per-lesson summaries JSON",
        "markdown": "Short Markdown summary",
        "course_full_md": "Full Markdown export with lesson bodies",
        "metadata": "Generation metadata JSON",
        "bundle": "Combined bundle JSON",
        "report": "Generation report with quality breakdown",
        "docx": "DOCX summary",
        "pdf": "PDF summary",
        "flashcards": "Flashcards JSON",
        "anki_tsv": "Anki import (TSV)",
        "quizzes_csv": "Quizzes spreadsheet (CSV)",
        "quizzes_gift": "Moodle GIFT quiz import",
        "delivery_zip": "All artifacts in one ZIP",
        "output_index": "This index file",
        "run_manifest": "Compact run summary JSON",
    }

    for key in sorted(paths.keys()):
        file_path = paths[key]
        path = Path(file_path)
        if not path.is_file():
            continue
        desc = descriptions.get(key, key)
        size_kb = path.stat().st_size / 1024
        lines.append(f"- **{path.name}** — {desc} ({size_kb:.1f} KB)")

    lines.append("")
    lines.append("Start with `course.html` for the full learning experience.")
    return "\n".join(lines) + "\n"
