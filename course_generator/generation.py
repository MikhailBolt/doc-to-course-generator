import html
import json
from typing import Any, Dict, List

from langchain_ollama import OllamaLLM

from course_generator.utils import (
    call_llm,
    clean_text,
    deduplicate_questions,
    deduplicate_sources,
    extract_json_from_text,
    sanitize_lesson_html,
)


def retry_llm_json(llm: OllamaLLM, prompt: str, max_attempts: int = 3) -> Any:
    last_error = None
    current_prompt = prompt
    for _ in range(max_attempts):
        raw = call_llm(llm, current_prompt)
        try:
            return extract_json_from_text(raw)
        except Exception as exc:
            last_error = exc
            current_prompt = "Return ONLY valid JSON with no markdown fences.\n\n" + prompt
    raise ValueError(f"Failed to get valid JSON after {max_attempts} attempts: {last_error}")


def validate_outline(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Outline is not a JSON object.")

    required_str_fields = ["course_title", "course_description", "target_audience"]
    for field in required_str_fields:
        if not isinstance(data.get(field), str) or not data.get(field).strip():
            raise ValueError(f"Outline field '{field}' is missing or invalid.")

    for list_field in ["prerequisites", "learning_outcomes", "glossary", "lessons"]:
        if not isinstance(data.get(list_field), list):
            raise ValueError(f"Outline field '{list_field}' must be a list.")

    if not data["lessons"]:
        raise ValueError("Outline contains no lessons.")

    normalized_lessons = []
    for lesson in data["lessons"]:
        if not isinstance(lesson, dict):
            continue
        title = str(lesson.get("title", "")).strip()
        goal = str(lesson.get("goal", "")).strip()
        key_points = lesson.get("key_points", [])
        if not title or not goal or not isinstance(key_points, list):
            continue
        cleaned_points = [str(x).strip() for x in key_points if str(x).strip()]
        if not cleaned_points:
            continue
        normalized_lessons.append({
            "title": title,
            "goal": goal,
            "key_points": cleaned_points[:5],
        })

    if not normalized_lessons:
        raise ValueError("No valid lessons found in outline.")

    normalized_glossary = []
    for item in data.get("glossary", []):
        if not isinstance(item, dict):
            continue
        term = str(item.get("term", "")).strip()
        definition = str(item.get("definition", "")).strip()
        if term and definition:
            normalized_glossary.append({"term": term, "definition": definition})

    data["lessons"] = normalized_lessons
    data["glossary"] = normalized_glossary[:8]
    data["prerequisites"] = [str(x).strip() for x in data.get("prerequisites", []) if str(x).strip()]
    data["learning_outcomes"] = [str(x).strip() for x in data.get("learning_outcomes", []) if str(x).strip()]
    return data


def get_language_instruction(language: str) -> str:
    return "Generate all content in Russian." if language == "ru" else "Generate all content in English."


def fallback_lesson_html(lesson_title: str, lesson_goal: str, key_points: List[str], retrieved_docs: List[Any]) -> Dict[str, Any]:
    paragraphs = []
    for doc in retrieved_docs[:2]:
        text = clean_text(doc.page_content)
        if text:
            paragraphs.append(html.escape(text[:700]))

    body_html = "".join(f"<p>{p}</p>" for p in paragraphs[:2])
    takeaway_html = "".join(f"<li>{html.escape(point)}</li>" for point in key_points[:5])

    lesson_html = f"""
<section class="lesson-section">
  <h2>{html.escape(lesson_title)}</h2>
  <p>{html.escape(lesson_goal)}</p>
  {body_html}
  <h3>Key takeaways</h3>
  <ul>
    {takeaway_html}
  </ul>
  <h3>In practice</h3>
  <p>Review the source material and connect the main ideas from this lesson to a real task or workflow.</p>
</section>
""".strip()

    return {
        "lesson_html": lesson_html,
        "summary": lesson_goal,
        "key_takeaways": key_points[:5],
    }


def validate_question_item(item: Dict[str, Any], allow_true_false: bool) -> Dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    question = str(item.get("question", "")).strip()
    q_type = str(item.get("type", "single_choice")).strip()
    options = item.get("options", [])
    correct_answer = str(item.get("correct_answer", "")).strip()
    explanation = str(item.get("explanation", "")).strip()

    if not question or not isinstance(options, list):
        return None

    if allow_true_false and q_type == "true_false":
        if options != ["True", "False"]:
            return None
        if correct_answer not in options:
            return None
        return {
            "question": question,
            "type": "true_false",
            "options": options,
            "correct_answer": correct_answer,
            "explanation": explanation,
        }

    if len(options) != 4 or correct_answer not in options:
        return None

    return {
        "question": question,
        "type": "single_choice",
        "options": [str(x).strip() for x in options],
        "correct_answer": correct_answer,
        "explanation": explanation,
    }


def generate_course_outline(
    llm: OllamaLLM,
    preview_text: str,
    rag_context: str,
    difficulty: str,
    language: str,
    min_lessons: int,
    max_lessons: int,
) -> Dict[str, Any]:
    rag_section = ""
    if rag_context.strip():
        rag_section = f"""
SEMANTIC RETRIEVAL CONTEXT (representative chunks from the vector index — use together with previews):
{rag_context}
"""

    prompt = f"""
You are an expert instructional designer.

{get_language_instruction(language)}

Based ONLY on the materials below (semantic retrieval chunks and/or document previews), generate a structured training course outline.

Requirements:
- The course must be practical, clear, and useful for real learning.
- Difficulty level: {difficulty}.
- Return ONLY valid JSON.
- The JSON schema must be:

{{
  "course_title": "string",
  "course_description": "string",
  "target_audience": "string",
  "prerequisites": ["string", "string"],
  "learning_outcomes": ["string", "string"],
  "glossary": [
    {{
      "term": "string",
      "definition": "string"
    }}
  ],
  "lessons": [
    {{
      "title": "string",
      "goal": "string",
      "key_points": ["string", "string", "string"]
    }}
  ]
}}

Rules:
- Create {min_lessons} to {max_lessons} lessons.
- Each lesson should have 3 to 5 key points.
- Create 3 to 8 glossary items.
- Do not invent topics that are not supported by the materials below.
- Be specific.
- Keep target_audience short.

{rag_section}
DOCUMENT PREVIEW (beginning of each file; may be truncated):
{preview_text}
"""
    data = retry_llm_json(llm, prompt)
    return validate_outline(data)


def review_outline(llm: OllamaLLM, outline: Dict[str, Any], language: str, min_lessons: int, max_lessons: int) -> Dict[str, Any]:
    prompt = f"""
You are reviewing a generated training course outline.

{get_language_instruction(language)}

Return ONLY valid JSON with exactly the same schema as the input.

Your job:
- improve clarity
- remove redundancy
- make lesson progression more logical
- keep only information grounded in the source-derived outline
- do not add unrelated topics
- preserve {min_lessons} to {max_lessons} lessons

INPUT OUTLINE:
{json.dumps(outline, ensure_ascii=False, indent=2)}
"""
    reviewed = retry_llm_json(llm, prompt)
    return validate_outline(reviewed)


def generate_lesson_html_section(
    llm: OllamaLLM,
    lesson_title: str,
    lesson_goal: str,
    key_points: List[str],
    retrieved_docs: List[Any],
    language: str,
    include_source_excerpts: bool,
) -> Dict[str, Any]:
    context_blocks = []
    sources = []
    source_excerpts = []

    for doc in retrieved_docs:
        page = doc.metadata.get("page")
        page_num = page + 1 if isinstance(page, int) else "N/A"
        doc_name = doc.metadata.get("document_name", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "N/A")
        excerpt = clean_text(doc.page_content)[:300]
        sources.append({"document_name": doc_name, "page": page_num, "chunk_id": chunk_id})
        source_excerpts.append({"document_name": doc_name, "page": page_num, "chunk_id": chunk_id, "excerpt": excerpt})
        context_blocks.append(f"[Document: {doc_name} | Page: {page_num} | Chunk: {chunk_id}]\n{doc.page_content}")

    context_text = "\n\n".join(context_blocks)
    prompt = f"""
You are creating one lesson section for an HTML training course.

{get_language_instruction(language)}

Write the lesson using ONLY the provided context.

Lesson title: {lesson_title}
Lesson goal: {lesson_goal}
Key points:
{json.dumps(key_points, ensure_ascii=False, indent=2)}

Return ONLY valid JSON with this schema:
{{
  "lesson_html": "<section>...</section>",
  "summary": "string",
  "key_takeaways": ["string", "string", "string"]
}}

Requirements for lesson_html:
- Use semantic HTML.
- Start with <section class="lesson-section"> and end with </section>.
- Include:
  - <h2> lesson title
  - <p> lesson goal/introduction
  - 2 to 4 short paragraphs
  - <ul> with key takeaways
  - one subsection titled "In practice"
- Do not include full HTML page, only section HTML.
- Do not use markdown.
- Keep it readable and concise.
- Do not invent facts beyond the context.

CONTEXT:
{context_text}
"""
    try:
        data = retry_llm_json(llm, prompt)
        if not isinstance(data, dict):
            raise ValueError("Lesson JSON is not an object.")
        lesson_html = sanitize_lesson_html(str(data.get("lesson_html", "")).strip())
        summary = str(data.get("summary", "")).strip()
        key_takeaways = data.get("key_takeaways", [])
        if not lesson_html:
            raise ValueError("Empty lesson HTML.")
        if not isinstance(key_takeaways, list):
            key_takeaways = []
        return {
            "lesson_html": lesson_html,
            "summary": summary or lesson_goal,
            "key_takeaways": [str(x).strip() for x in key_takeaways if str(x).strip()][:5],
            "sources": deduplicate_sources(sources),
            "source_excerpts": source_excerpts[: min(3, len(source_excerpts))] if include_source_excerpts else [],
        }
    except Exception:
        fallback = fallback_lesson_html(lesson_title, lesson_goal, key_points, retrieved_docs)
        fallback["sources"] = deduplicate_sources(sources)
        fallback["source_excerpts"] = source_excerpts[: min(3, len(source_excerpts))] if include_source_excerpts else []
        return fallback


def generate_pretest(llm: OllamaLLM, outline: Dict[str, Any], pretest_questions: int, difficulty: str, language: str) -> List[Dict[str, Any]]:
    prompt = f"""
You are generating a short diagnostic pre-test for a course.

{get_language_instruction(language)}

Difficulty: {difficulty}
Generate exactly {pretest_questions} questions.

Return ONLY valid JSON as an array.
Each item must follow this schema:

[
  {{
    "question": "string",
    "type": "single_choice",
    "options": ["A", "B", "C", "D"],
    "correct_answer": "one of the options exactly",
    "explanation": "string"
  }}
]

Rules:
- Use ONLY topics that appear in the course outline below.
- Keep questions broad and diagnostic.
- Each question must have exactly 4 options.
- Only 1 option must be correct.

COURSE OUTLINE:
{json.dumps(outline, ensure_ascii=False, indent=2)}
"""
    data = retry_llm_json(llm, prompt)
    if not isinstance(data, list):
        raise ValueError("Pre-test output is not a JSON array.")
    cleaned = []
    for item in data:
        validated = validate_question_item(item, allow_true_false=False)
        if validated:
            cleaned.append(validated)
    cleaned = deduplicate_questions(cleaned)
    if not cleaned:
        raise ValueError("No valid pre-test questions were generated.")
    return cleaned[:pretest_questions]


def generate_quiz(
    llm: OllamaLLM,
    outline: Dict[str, Any],
    lesson_payloads: List[Dict[str, Any]],
    difficulty: str,
    quiz_questions: int,
    language: str,
) -> List[Dict[str, Any]]:
    lesson_summaries = []
    lesson_titles = []

    for idx, lesson in enumerate(outline.get("lessons", []), start=1):
        lesson_titles.append(lesson.get("title", ""))
        summary = lesson_payloads[idx - 1].get("summary", "")
        lesson_summaries.append(
            {
                "lesson_number": idx,
                "title": lesson.get("title", ""),
                "goal": lesson.get("goal", ""),
                "key_points": lesson.get("key_points", []),
                "summary": summary,
                "key_takeaways": lesson_payloads[idx - 1].get("key_takeaways", []),
            }
        )

    prompt = f"""
You are generating a final quiz for a training course.

{get_language_instruction(language)}

Difficulty: {difficulty}
Generate exactly {quiz_questions} questions.

Return ONLY valid JSON as an array.
Each item must follow this schema:

[
  {{
    "question": "string",
    "type": "single_choice" or "true_false",
    "options": ["A", "B", "C", "D"] OR ["True", "False"],
    "correct_answer": "one of the options exactly",
    "explanation": "string",
    "lesson_title": "string"
  }}
]

Rules:
- Use ONLY the information from the lesson summaries below.
- At least 70% of questions should be single_choice.
- true_false questions must use exactly ["True", "False"].
- single_choice questions must have exactly 4 options.
- Only 1 option must be correct.
- Options must be plausible.
- Explanations should be concise.
- Avoid duplicate questions.
- Cover multiple lessons, not only one lesson.
- lesson_title must match one of the provided lesson titles exactly.

LESSON TITLES:
{json.dumps(lesson_titles, ensure_ascii=False, indent=2)}

LESSON SUMMARIES:
{json.dumps(lesson_summaries, ensure_ascii=False, indent=2)}
"""
    data = retry_llm_json(llm, prompt)
    if not isinstance(data, list):
        raise ValueError("Quiz output is not a JSON array.")

    cleaned_questions = []
    for item in data:
        validated = validate_question_item(item, allow_true_false=True)
        if validated:
            lesson_title = str(item.get("lesson_title", "")).strip()
            validated["lesson_title"] = lesson_title if lesson_title in lesson_titles else ""
            cleaned_questions.append(validated)

    cleaned_questions = deduplicate_questions(cleaned_questions)
    if not cleaned_questions:
        raise ValueError("No valid quiz questions were generated.")
    return cleaned_questions[:quiz_questions]


def review_quiz(llm: OllamaLLM, quiz_data: List[Dict[str, Any]], lesson_titles: List[str], language: str) -> List[Dict[str, Any]]:
    prompt = f"""
You are reviewing a generated training quiz.

{get_language_instruction(language)}

Return ONLY valid JSON as an array.

Your job:
- remove duplicates
- improve wording
- make distractors more plausible
- keep the number of questions the same or slightly smaller
- preserve schema:
[
  {{
    "question": "string",
    "type": "single_choice" or "true_false",
    "options": [...],
    "correct_answer": "string",
    "explanation": "string",
    "lesson_title": "string"
  }}
]

Allowed lesson titles:
{json.dumps(lesson_titles, ensure_ascii=False, indent=2)}

QUIZ:
{json.dumps(quiz_data, ensure_ascii=False, indent=2)}
"""
    reviewed = retry_llm_json(llm, prompt)
    if not isinstance(reviewed, list):
        return quiz_data

    cleaned_questions = []
    for item in reviewed:
        validated = validate_question_item(item, allow_true_false=True)
        if validated:
            lesson_title = str(item.get("lesson_title", "")).strip()
            validated["lesson_title"] = lesson_title if lesson_title in lesson_titles else ""
            cleaned_questions.append(validated)

    cleaned_questions = deduplicate_questions(cleaned_questions)
    return cleaned_questions if cleaned_questions else quiz_data
