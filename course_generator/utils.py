import json
import re
from typing import Any, Dict, List

from langchain_ollama import OllamaLLM


def clean_text(text: str) -> str:
    text = re.sub(r"\r", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def normalize_json_block(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def extract_json_from_text(text: str) -> Any:
    text = normalize_json_block(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise


def sanitize_lesson_html(section_html: str) -> str:
    section_html = re.sub(r"<script.*?>.*?</script>", "", section_html, flags=re.IGNORECASE | re.DOTALL)
    section_html = re.sub(r"<style.*?>.*?</style>", "", section_html, flags=re.IGNORECASE | re.DOTALL)
    section_html = re.sub(r"\son\w+\s*=\s*(['\"]).*?\1", "", section_html, flags=re.IGNORECASE | re.DOTALL)
    return section_html.strip()


def deduplicate_questions(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique = []
    seen = set()
    for item in questions:
        question = str(item.get("question", "")).strip().lower()
        question = re.sub(r"\s+", " ", question)
        options = tuple(str(x).strip().lower() for x in item.get("options", []))
        key = (question, options)
        if not question or key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def deduplicate_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for src in sources:
        key = (
            str(src.get("document_name", "")),
            str(src.get("page", "")),
            str(src.get("chunk_id", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(src)
    return unique


def estimate_duration_minutes(outline: Dict[str, Any]) -> int:
    lessons = outline.get("lessons", [])
    lesson_count = len(lessons) if isinstance(lessons, list) else 0
    return max(20, lesson_count * 12 + 10)


def ensure_minimum_quiz_coverage(quiz_data: List[Dict[str, Any]], lesson_titles: List[str]) -> Dict[str, Any]:
    covered = set()
    for q in quiz_data:
        title = str(q.get("lesson_title", "")).strip()
        if title:
            covered.add(title)

    missing = [title for title in lesson_titles if title not in covered]
    return {
        "covered_lessons": sorted(list(covered)),
        "missing_lessons": missing,
        "coverage_ratio": round(len(covered) / len(lesson_titles), 2) if lesson_titles else 1.0,
    }


def call_llm(llm: OllamaLLM, prompt: str) -> str:
    response = llm.invoke(prompt)
    return response.strip() if isinstance(response, str) else str(response).strip()
