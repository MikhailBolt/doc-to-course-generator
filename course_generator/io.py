import json
import re
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from course_generator.utils import ensure_minimum_quiz_coverage, estimate_duration_minutes


def sanitize_filename_part(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def build_output_path(output_dir: str, filename: str, output_prefix: str = "") -> Path:
    if output_prefix:
        prefix = sanitize_filename_part(output_prefix)
        filename = f"{prefix}_{filename}"
    return Path(output_dir) / filename


def save_json(path: Path, data: Any) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return str(path)


def save_text(path: Path, text: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return str(path)


def save_course_html(output_dir: str, html_text: str, output_prefix: str = "") -> str:
    return save_text(build_output_path(output_dir, "course.html", output_prefix), html_text)


def save_quiz_json(output_dir: str, quiz_data: List[Dict[str, Any]], output_prefix: str = "") -> str:
    return save_json(build_output_path(output_dir, "quiz.json", output_prefix), quiz_data)


def save_pretest_json(output_dir: str, pretest_data: List[Dict[str, Any]], output_prefix: str = "") -> str:
    return save_json(build_output_path(output_dir, "pretest.json", output_prefix), pretest_data)


def save_outline_json(output_dir: str, outline: Dict[str, Any], output_prefix: str = "") -> str:
    return save_json(build_output_path(output_dir, "course_outline.json", output_prefix), outline)


def save_markdown_summary(output_dir: str, markdown_text: str, output_prefix: str = "") -> str:
    return save_text(build_output_path(output_dir, "course_summary.md", output_prefix), markdown_text)


def save_lesson_summaries(output_dir: str, lesson_payloads: List[Dict[str, Any]], outline: Dict[str, Any], output_prefix: str = "") -> str:
    path = build_output_path(output_dir, "lesson_summaries.json", output_prefix)
    lessons = outline.get("lessons", [])
    payload = []
    for idx, lesson in enumerate(lessons):
        lesson_payload = lesson_payloads[idx] if idx < len(lesson_payloads) else {}
        payload.append({
            "lesson_number": idx + 1,
            "title": lesson.get("title", ""),
            "goal": lesson.get("goal", ""),
            "key_points": lesson.get("key_points", []),
            "summary": lesson_payload.get("summary", ""),
            "key_takeaways": lesson_payload.get("key_takeaways", []),
            "sources": lesson_payload.get("sources", []),
            "source_excerpts": lesson_payload.get("source_excerpts", []),
        })
    return save_json(path, payload)


def save_generation_report(
    output_dir: str,
    outline: Dict[str, Any],
    docs_info: List[Dict[str, Any]],
    pretest_data: List[Dict[str, Any]],
    quiz_data: List[Dict[str, Any]],
    args: Namespace,
    elapsed_seconds: float,
    outline_rag_used: bool,
) -> str:
    path = build_output_path(output_dir, "generation_report.json", args.output_prefix)
    report = {
        "generated_at": datetime.now().isoformat(),
        "model": args.model,
        "embedding_model": args.embedding_model,
        "difficulty": args.difficulty,
        "retrieval_type": args.retrieval_type,
        "language": args.language,
        "output_prefix": args.output_prefix,
        "top_k": args.top_k,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "documents_count": len(docs_info),
        "documents": docs_info,
        "lessons_count": len(outline.get("lessons", [])),
        "pretest_questions_count": len(pretest_data),
        "final_quiz_questions_count": len(quiz_data),
        "estimated_duration_minutes": estimate_duration_minutes(outline),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "quiz_coverage": ensure_minimum_quiz_coverage(quiz_data, [lesson.get("title", "") for lesson in outline.get("lessons", [])]) if quiz_data else {"covered_lessons": [], "missing_lessons": [], "coverage_ratio": 0},
        "source_extensions": sorted(list({doc.get("type", "unknown") for doc in docs_info})),
        "skip_pretest": args.skip_pretest,
        "skip_final_quiz": args.skip_final_quiz,
        "include_source_excerpts": args.include_source_excerpts,
        "outline_rag_used": outline_rag_used,
        "skip_outline_rag": getattr(args, "skip_outline_rag", False),
    }
    return save_json(path, report)


def save_course_metadata(output_dir: str, outline: Dict[str, Any], docs_info: List[Dict[str, Any]], args: Namespace) -> str:
    path = build_output_path(output_dir, "course_metadata.json", args.output_prefix)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "model": args.model,
        "embedding_model": args.embedding_model,
        "difficulty": args.difficulty,
        "retrieval_type": args.retrieval_type,
        "language": args.language,
        "output_prefix": args.output_prefix,
        "top_k": args.top_k,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "documents": docs_info,
        "outline": outline,
    }
    return save_json(path, payload)


def save_course_bundle(output_dir: str, outline: Dict[str, Any], docs_info: List[Dict[str, Any]], lesson_payloads: List[Dict[str, Any]], pretest_data: List[Dict[str, Any]], quiz_data: List[Dict[str, Any]], args: Namespace) -> str:
    path = build_output_path(output_dir, "course_bundle.json", args.output_prefix)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "model": args.model,
            "embedding_model": args.embedding_model,
            "difficulty": args.difficulty,
            "retrieval_type": args.retrieval_type,
            "language": args.language,
            "output_prefix": args.output_prefix,
        },
        "documents": docs_info,
        "outline": outline,
        "lessons": lesson_payloads,
        "pretest": pretest_data,
        "final_quiz": quiz_data,
    }
    return save_json(path, payload)
