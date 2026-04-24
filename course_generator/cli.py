import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

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
from course_generator.documents import collect_source_files
from course_generator.pipeline import run_pipeline

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

    log_message(args.log_dir, "Starting course and quiz generation pipeline")
    try:
        # NOTE: We keep CLI output roughly the same, but delegate work to the shared pipeline.
        source_files = collect_source_files(args.docs_path)
        print(f"Found {len(source_files)} source file(s).")
        print("--- Running generation pipeline... ---")
        result = run_pipeline(args)
    except Exception as exc:
        print(f"(X) Generation failed: {exc}")
        log_message(args.log_dir, f"Generation error: {exc}")
        sys.exit(1)

    paths = result["paths"]
    print("\n[SUCCESS] Generation complete!")
    print(f"Course HTML:        {paths['course_html']}")
    print(f"Course outline:     {paths['outline']}")
    print(f"Pre-test JSON:      {paths['pretest']}")
    print(f"Final quiz JSON:    {paths['quiz']}")
    print(f"Lesson summaries:   {paths['summaries']}")
    print(f"Markdown summary:   {paths['markdown']}")
    print(f"Metadata:           {paths['metadata']}")
    print(f"Bundle:             {paths['bundle']}")
    print(f"Report:             {paths['report']}")
    print(f"Time:               {result['elapsed_seconds']:.2f}s")

    log_message(
        args.log_dir,
        "Generation complete. "
        f"course={paths['course_html']}, outline={paths['outline']}, pretest={paths['pretest']}, quiz={paths['quiz']}, "
        f"summaries={paths['summaries']}, markdown={paths['markdown']}, metadata={paths['metadata']}, "
        f"bundle={paths['bundle']}, report={paths['report']}, elapsed={result['elapsed_seconds']:.2f}s",
    )
