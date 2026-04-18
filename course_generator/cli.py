import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_ollama import OllamaLLM

from course_generator.constants import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DB_FAISS_PATH,
    DEFAULT_DOCS_PATH,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_LESSONS,
    DEFAULT_MAX_PREVIEW_CHARS_PER_FILE,
    DEFAULT_MIN_LESSONS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_PREFIX,
    DEFAULT_PRETEST_QUESTIONS,
    DEFAULT_QUIZ_QUESTIONS,
    DEFAULT_LLM_MODEL,
    DEFAULT_LOG_DIR,
    DEFAULT_MANIFEST_FILE,
    DEFAULT_OUTLINE_RAG_MAX_CHARS,
    DEFAULT_OUTLINE_RAG_MAX_CHUNKS,
    DEFAULT_RETRIEVAL_TYPE,
    DEFAULT_TOP_K,
    SUPPORTED_EXTENSIONS,
)
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

load_dotenv()


def ensure_directories(docs_path: str, db_path: str, manifest_file: str, output_dir: str, log_dir: str) -> None:
    docs = Path(docs_path)
    if docs.suffix.lower() in SUPPORTED_EXTENSIONS:
        docs.parent.mkdir(parents=True, exist_ok=True)
    else:
        docs.mkdir(parents=True, exist_ok=True)

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate HTML course and quizzes from local documents using LLM + RAG"
    )
    parser.add_argument("--docs-path", default=os.getenv("DOCS_PATH", DEFAULT_DOCS_PATH))
    parser.add_argument("--db", default=os.getenv("DB_FAISS_PATH", DEFAULT_DB_FAISS_PATH))
    parser.add_argument("--manifest-file", default=os.getenv("MANIFEST_FILE", DEFAULT_MANIFEST_FILE))
    parser.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    parser.add_argument("--log-dir", default=os.getenv("LOG_DIR", DEFAULT_LOG_DIR))
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL))
    parser.add_argument("--chunk-size", type=int, default=int(os.getenv("CHUNK_SIZE", DEFAULT_CHUNK_SIZE)))
    parser.add_argument("--chunk-overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)))
    parser.add_argument("--top-k", type=int, default=int(os.getenv("TOP_K", DEFAULT_TOP_K)))
    parser.add_argument("--quiz-questions", type=int, default=int(os.getenv("QUIZ_QUESTIONS", DEFAULT_QUIZ_QUESTIONS)))
    parser.add_argument("--pretest-questions", type=int, default=int(os.getenv("PRETEST_QUESTIONS", DEFAULT_PRETEST_QUESTIONS)))
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default=os.getenv("DIFFICULTY", "medium"))
    parser.add_argument("--retrieval-type", choices=["similarity", "mmr"], default=os.getenv("RETRIEVAL_TYPE", DEFAULT_RETRIEVAL_TYPE))
    parser.add_argument("--language", choices=["en", "ru"], default=os.getenv("LANGUAGE", DEFAULT_LANGUAGE))
    parser.add_argument("--max-preview-chars-per-file", type=int, default=int(os.getenv("MAX_PREVIEW_CHARS_PER_FILE", DEFAULT_MAX_PREVIEW_CHARS_PER_FILE)))
    parser.add_argument("--output-prefix", default=os.getenv("OUTPUT_PREFIX", DEFAULT_OUTPUT_PREFIX))
    parser.add_argument("--min-lessons", type=int, default=int(os.getenv("MIN_LESSONS", DEFAULT_MIN_LESSONS)))
    parser.add_argument("--max-lessons", type=int, default=int(os.getenv("MAX_LESSONS", DEFAULT_MAX_LESSONS)))
    parser.add_argument("--disable-review-pass", action="store_true")
    parser.add_argument("--skip-pretest", action="store_true")
    parser.add_argument("--skip-final-quiz", action="store_true")
    parser.add_argument("--include-source-excerpts", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--skip-outline-rag", action="store_true", help="Do not attach FAISS retrieval chunks to the outline prompt.")
    parser.add_argument(
        "--outline-rag-max-chunks",
        type=int,
        default=int(os.getenv("OUTLINE_RAG_MAX_CHUNKS", DEFAULT_OUTLINE_RAG_MAX_CHUNKS)),
    )
    parser.add_argument(
        "--outline-rag-max-chars",
        type=int,
        default=int(os.getenv("OUTLINE_RAG_MAX_CHARS", DEFAULT_OUTLINE_RAG_MAX_CHARS)),
    )
    return parser.parse_args()


def log_message(log_dir: str, message: str) -> None:
    log_file = Path(log_dir) / f"run_{datetime.now().strftime('%Y-%m-%d')}.log"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")


def main() -> None:
    args = parse_args()
    ensure_directories(args.docs_path, args.db, args.manifest_file, args.output_dir, args.log_dir)

    if args.min_lessons < 1 or args.max_lessons < args.min_lessons:
        print("(X) Invalid lesson range. Check --min-lessons and --max-lessons.")
        sys.exit(1)

    started = time.time()
    log_message(args.log_dir, "Starting course and quiz generation pipeline")

    source_files = collect_source_files(args.docs_path)
    print(f"Found {len(source_files)} source file(s).")

    try:
        vectorstore, docs_info = load_or_create_vectorstore(args, source_files)
        log_message(args.log_dir, f"Vector store ready. Documents loaded: {len(docs_info)}")
    except Exception as exc:
        print(f"(X) Failed to prepare vector store: {exc}")
        log_message(args.log_dir, f"Vector store error: {exc}")
        sys.exit(1)

    try:
        llm = OllamaLLM(model=args.model)
        log_message(args.log_dir, f"Initialized Ollama model: {args.model}")
    except Exception as exc:
        print(f"(X) Failed to initialize LLM: {exc}")
        log_message(args.log_dir, f"LLM init error: {exc}")
        sys.exit(1)

    outline_rag_used = False
    try:
        print("--- Generating course outline... ---")
        preview_text = get_combined_preview_text(source_files, max_chars_per_file=args.max_preview_chars_per_file)
        rag_context = ""
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
        log_message(args.log_dir, "Course outline generated successfully")
    except Exception as exc:
        print(f"(X) Failed to generate course outline: {exc}")
        log_message(args.log_dir, f"Course outline error: {exc}")
        sys.exit(1)

    lesson_payloads: List[Dict[str, Any]] = []
    lessons = outline.get("lessons", [])
    try:
        for idx, lesson in enumerate(lessons, start=1):
            lesson_title = str(lesson.get("title", f"Lesson {idx}")).strip()
            lesson_goal = str(lesson.get("goal", "")).strip()
            key_points = lesson.get("key_points", []) if isinstance(lesson.get("key_points", []), list) else []
            print(f"--- Generating lesson {idx}/{len(lessons)}: {lesson_title} ---")
            log_message(args.log_dir, f"Generating lesson: {lesson_title}")
            retrieved_docs = retrieve_lesson_context(vectorstore, lesson_title, key_points, args.top_k, args.retrieval_type)
            lesson_payloads.append(
                generate_lesson_html_section(llm, lesson_title, lesson_goal, key_points, retrieved_docs, args.language, args.include_source_excerpts)
            )
        log_message(args.log_dir, "All lesson sections generated successfully")
    except Exception as exc:
        print(f"(X) Failed during lesson generation: {exc}")
        log_message(args.log_dir, f"Lesson generation error: {exc}")
        sys.exit(1)

    pretest_data: List[Dict[str, Any]] = []
    if not args.skip_pretest:
        try:
            print("--- Generating pre-test... ---")
            pretest_data = generate_pretest(llm, outline, args.pretest_questions, args.difficulty, args.language)
            log_message(args.log_dir, f"Pre-test generated successfully with {len(pretest_data)} questions")
        except Exception as exc:
            print(f"(X) Failed to generate pre-test: {exc}")
            log_message(args.log_dir, f"Pre-test generation error: {exc}")
            sys.exit(1)

    quiz_data: List[Dict[str, Any]] = []
    if not args.skip_final_quiz:
        try:
            print("--- Generating final quiz... ---")
            quiz_data = generate_quiz(llm, outline, lesson_payloads, args.difficulty, args.quiz_questions, args.language)
            if not args.disable_review_pass:
                quiz_data = review_quiz(llm, quiz_data, [lesson.get("title", "") for lesson in outline.get("lessons", [])], args.language)
            log_message(args.log_dir, f"Quiz generated successfully with {len(quiz_data)} questions")
        except Exception as exc:
            print(f"(X) Failed to generate quiz: {exc}")
            log_message(args.log_dir, f"Quiz generation error: {exc}")
            sys.exit(1)

    try:
        print("--- Building final HTML course... ---")
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

        print("\n[SUCCESS] Generation complete!")
        print(f"Course HTML:        {course_path}")
        print(f"Course outline:     {outline_path}")
        print(f"Pre-test JSON:      {pretest_path}")
        print(f"Final quiz JSON:    {quiz_path}")
        print(f"Lesson summaries:   {summaries_path}")
        print(f"Markdown summary:   {markdown_path}")
        print(f"Metadata:           {metadata_path}")
        print(f"Bundle:             {bundle_path}")
        print(f"Report:             {report_path}")
        print(f"Time:               {elapsed:.2f}s")

        log_message(
            args.log_dir,
            f"Generation complete. course={course_path}, outline={outline_path}, pretest={pretest_path}, quiz={quiz_path}, summaries={summaries_path}, markdown={markdown_path}, metadata={metadata_path}, bundle={bundle_path}, report={report_path}, elapsed={elapsed:.2f}s",
        )
    except Exception as exc:
        print(f"(X) Failed to save outputs: {exc}")
        log_message(args.log_dir, f"Save outputs error: {exc}")
        sys.exit(1)
