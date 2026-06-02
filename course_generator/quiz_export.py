import csv
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List


def quiz_rows_for_csv(questions: List[Dict[str, Any]], *, quiz_name: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for idx, item in enumerate(questions, start=1):
        options = item.get("options", [])
        if not isinstance(options, list):
            options = []
        padded = [str(o) for o in options] + [""] * (4 - len(options))
        rows.append(
            {
                "quiz": quiz_name,
                "number": str(idx),
                "question": str(item.get("question", "")).strip(),
                "type": str(item.get("type", "single_choice")),
                "option_a": padded[0] if len(padded) > 0 else "",
                "option_b": padded[1] if len(padded) > 1 else "",
                "option_c": padded[2] if len(padded) > 2 else "",
                "option_d": padded[3] if len(padded) > 3 else "",
                "correct_answer": str(item.get("correct_answer", "")).strip(),
                "explanation": str(item.get("explanation", "")).strip(),
                "lesson_title": str(item.get("lesson_title", "")).strip(),
            }
        )
    return rows


def write_quiz_csv(path: Path, questions: List[Dict[str, Any]], *, quiz_name: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "quiz",
        "number",
        "question",
        "type",
        "option_a",
        "option_b",
        "option_c",
        "option_d",
        "correct_answer",
        "explanation",
        "lesson_title",
    ]
    rows = quiz_rows_for_csv(questions, quiz_name=quiz_name)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def combined_quiz_csv_text(pretest: List[Dict[str, Any]], final_quiz: List[Dict[str, Any]]) -> str:
    """UTF-8 BOM CSV string with pre-test and final quiz rows."""
    buffer = StringIO()
    fieldnames = [
        "quiz",
        "number",
        "question",
        "type",
        "option_a",
        "option_b",
        "option_c",
        "option_d",
        "correct_answer",
        "explanation",
        "lesson_title",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(quiz_rows_for_csv(pretest, quiz_name="pretest"))
    writer.writerows(quiz_rows_for_csv(final_quiz, quiz_name="final"))
    return buffer.getvalue()
