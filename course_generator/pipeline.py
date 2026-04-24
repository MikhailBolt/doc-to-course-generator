import time
from argparse import Namespace
from typing import Any, Dict, List

from langchain_ollama import OllamaLLM

from course_generator.documents import collect_source_files, get_combined_preview_text
from course_generator.generation import (
    generate_course_outline,
    generate_lesson_html_section,
    generate_pretest,
    generate_quiz,
    review_outline,
    review_quiz,
)
from course_generator.html_export import build_course_html, build_markdown_summary
from course_generator.io import (
    save_course_bundle,
    save_course_html,
    save_course_metadata,
    save_generation_report,
    save_lesson_summaries,
    save_markdown_summary,
    save_outline_json,
    save_pretest_json,
    save_quiz_json,
)
from course_generator.rag import load_or_create_vectorstore, retrieve_lesson_context, retrieve_outline_context


def run_pipeline(args: Namespace) -> Dict[str, Any]:
    """
    Run the full generation pipeline and return paths + artifacts.

    Unlike CLI, this raises exceptions instead of calling sys.exit().
    """
    started = time.time()

    source_files = collect_source_files(args.docs_path)
    vectorstore, docs_info = load_or_create_vectorstore(args, source_files)
    llm = OllamaLLM(model=args.model)

    preview_text = get_combined_preview_text(source_files, max_chars_per_file=args.max_preview_chars_per_file)
    rag_context = ""
    outline_rag_used = False
    if not args.skip_outline_rag:
        rag_context = retrieve_outline_context(
            vectorstore,
            source_files,
            retrieval_type=args.retrieval_type,
            max_chunks=args.outline_rag_max_chunks,
            max_chars=args.outline_rag_max_chars,
        )
        outline_rag_used = bool(str(rag_context).strip())

    outline = generate_course_outline(
        llm,
        preview_text,
        rag_context,
        args.difficulty,
        args.language,
        args.min_lessons,
        args.max_lessons,
    )
    if not args.disable_review_pass:
        outline = review_outline(llm, outline, args.language, args.min_lessons, args.max_lessons)

    lesson_payloads: List[Dict[str, Any]] = []
    lessons = outline.get("lessons", [])
    for idx, lesson in enumerate(lessons, start=1):
        lesson_title = str(lesson.get("title", f"Lesson {idx}")).strip()
        lesson_goal = str(lesson.get("goal", "")).strip()
        key_points = lesson.get("key_points", []) if isinstance(lesson.get("key_points", []), list) else []
        retrieved_docs = retrieve_lesson_context(vectorstore, lesson_title, key_points, args.top_k, args.retrieval_type)
        lesson_payloads.append(
            generate_lesson_html_section(
                llm,
                lesson_title,
                lesson_goal,
                key_points,
                retrieved_docs,
                args.language,
                args.include_source_excerpts,
            )
        )

    pretest_data: List[Dict[str, Any]] = []
    if not args.skip_pretest:
        pretest_data = generate_pretest(llm, outline, args.pretest_questions, args.difficulty, args.language)

    quiz_data: List[Dict[str, Any]] = []
    if not args.skip_final_quiz:
        quiz_data = generate_quiz(llm, outline, lesson_payloads, args.difficulty, args.quiz_questions, args.language)
        if not args.disable_review_pass:
            quiz_data = review_quiz(llm, quiz_data, [lesson.get("title", "") for lesson in outline.get("lessons", [])], args.language)

    course_html = build_course_html(outline, lesson_payloads, docs_info, pretest_data, quiz_data, args.language, args.include_source_excerpts)
    markdown_summary = build_markdown_summary(outline, docs_info, lesson_payloads, args.language)

    course_path = save_course_html(args.output_dir, course_html, args.output_prefix)
    outline_path = save_outline_json(args.output_dir, outline, args.output_prefix)
    quiz_path = save_quiz_json(args.output_dir, quiz_data, args.output_prefix)
    pretest_path = save_pretest_json(args.output_dir, pretest_data, args.output_prefix)
    metadata_path = save_course_metadata(args.output_dir, outline, docs_info, args)
    summaries_path = save_lesson_summaries(args.output_dir, lesson_payloads, outline, args.output_prefix)
    markdown_path = save_markdown_summary(args.output_dir, markdown_summary, args.output_prefix)
    bundle_path = save_course_bundle(args.output_dir, outline, docs_info, lesson_payloads, pretest_data, quiz_data, args)

    elapsed = time.time() - started
    report_path = save_generation_report(
        args.output_dir,
        outline,
        docs_info,
        pretest_data,
        quiz_data,
        args,
        elapsed,
        outline_rag_used=outline_rag_used,
    )

    return {
        "docs_info": docs_info,
        "outline": outline,
        "lesson_payloads": lesson_payloads,
        "pretest": pretest_data,
        "quiz": quiz_data,
        "course_html": course_html,
        "markdown_summary": markdown_summary,
        "paths": {
            "course_html": course_path,
            "outline": outline_path,
            "pretest": pretest_path,
            "quiz": quiz_path,
            "summaries": summaries_path,
            "markdown": markdown_path,
            "metadata": metadata_path,
            "bundle": bundle_path,
            "report": report_path,
        },
        "elapsed_seconds": round(elapsed, 2),
        "outline_rag_used": outline_rag_used,
    }

