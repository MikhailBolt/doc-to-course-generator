from typing import Any, Dict, List


def build_flashcards(outline: Dict[str, Any], lesson_payloads: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Build study flashcards from glossary terms and lesson key points."""
    cards: List[Dict[str, str]] = []

    for entry in outline.get("glossary", []):
        if not isinstance(entry, dict):
            continue
        term = str(entry.get("term", "")).strip()
        definition = str(entry.get("definition", "")).strip()
        if term and definition:
            cards.append({"front": term, "back": definition, "source": "glossary"})

    lessons = outline.get("lessons", [])
    for idx, lesson in enumerate(lessons):
        if not isinstance(lesson, dict):
            continue
        title = str(lesson.get("title", f"Lesson {idx + 1}")).strip()
        goal = str(lesson.get("goal", "")).strip()
        payload = lesson_payloads[idx] if idx < len(lesson_payloads) else {}
        summary = str(payload.get("summary", "")).strip() or goal

        for point in lesson.get("key_points", []):
            point = str(point).strip()
            if not point:
                continue
            cards.append(
                {
                    "front": f"{title}: {point}",
                    "back": summary[:500],
                    "source": "lesson_key_point",
                }
            )

        takeaways = payload.get("key_takeaways", []) if isinstance(payload.get("key_takeaways"), list) else []
        for tw in takeaways:
            tw = str(tw).strip()
            if tw:
                cards.append({"front": f"{title} — takeaway", "back": tw, "source": "lesson_takeaway"})

    return cards[:200]


def flashcards_to_anki_tsv(cards: List[Dict[str, str]]) -> str:
    """Tab-separated deck for Anki import (front, back, tags)."""
    lines: List[str] = []
    for card in cards:
        front = str(card.get("front", "")).replace("\t", " ").replace("\n", " ").strip()
        back = str(card.get("back", "")).replace("\t", " ").replace("\n", " ").strip()
        tag = str(card.get("source", "course")).replace(" ", "_")
        if front and back:
            lines.append(f"{front}\t{back}\t{tag}")
    return "\n".join(lines) + ("\n" if lines else "")

