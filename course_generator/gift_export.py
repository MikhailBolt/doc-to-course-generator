from typing import Any, Dict, List


def _escape_gift(text: str) -> str:
    return str(text).replace("{", "\\{").replace("}", "\\}")


def questions_to_gift(questions: List[Dict[str, Any]], *, category: str) -> str:
    """Moodle GIFT format for multiple-choice / true-false questions."""
    blocks: List[str] = []
    cat = _escape_gift(category)
    blocks.append(f"$CATEGORY: {cat}")

    for idx, item in enumerate(questions, start=1):
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        options = item.get("options", [])
        if not isinstance(options, list) or not options:
            continue
        correct = str(item.get("correct_answer", "")).strip()
        title = _escape_gift(question[:60])
        body = _escape_gift(question)
        lines = [f"::{title}:: {body} {{"]
        for opt in options:
            opt_s = str(opt).strip()
            prefix = "=" if opt_s == correct else "~"
            lines.append(f"{prefix}{_escape_gift(opt_s)}")
        lines.append("}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks) + ("\n" if blocks else "")


def combined_gift_export(pretest: List[Dict[str, Any]], final_quiz: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    if pretest:
        parts.append(questions_to_gift(pretest, category="Pre-test"))
    if final_quiz:
        parts.append(questions_to_gift(final_quiz, category="Final quiz"))
    return "\n\n".join(parts)
