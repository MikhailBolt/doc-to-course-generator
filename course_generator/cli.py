import argparse
import os
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from course_generator import __version__
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
from course_generator.config_loader import CONFIG_KEYS, apply_config_file
from course_generator.scaffold import init_project_directories
from course_generator.audit import audit_output_paths
from course_generator.batch import run_batch
from course_generator.health import check_embeddings, check_ollama, format_ollama_message, list_ollama_models
from course_generator.report_diff import diff_generation_reports
from course_generator.history import list_recent_reports
from course_generator.io import load_outline_json
from course_generator.pipeline import run_pipeline
from course_generator.presets import PRESET_CLI_SLUGS, apply_cli_preset

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate HTML course and quizzes from local documents using LLM + RAG",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"doc-to-course-generator {__version__}",
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
    parser.add_argument("--check-ollama", action="store_true", help="Check Ollama connectivity and model availability, then exit.")
    parser.add_argument("--no-delivery-zip", action="store_true", help="Do not create course_delivery.zip after generation.")
    parser.add_argument(
        "--preset",
        choices=list(PRESET_CLI_SLUGS.keys()),
        default=None,
        help="Apply a named preset (quick, full, outline) before other flags.",
    )
    parser.add_argument("--list-models", action="store_true", help="List Ollama models and exit.")
    parser.add_argument("--export-docx", action="store_true", help="Also export course_summary.docx.")
    parser.add_argument("--export-pdf", action="store_true", help="Also export course_summary.pdf.")
    parser.add_argument("--open", action="store_true", help="Open course.html in the default browser after a successful run.")
    parser.add_argument(
        "--recursive-docs",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("DOCS_RECURSIVE", "").lower() in {"1", "true", "yes"},
        help="Include supported files in docs subfolders (default: from DOCS_RECURSIVE env).",
    )
    parser.add_argument(
        "--quality-llm-review",
        action="store_true",
        help="Run an extra LLM pass for narrative quality feedback in the report.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List source files and estimated LLM work without building the index or calling Ollama.",
    )
    parser.add_argument(
        "--outline-only",
        action="store_true",
        help="Generate and save course_outline.json only (still builds FAISS index).",
    )
    parser.add_argument(
        "--from-outline",
        metavar="PATH",
        default=None,
        help="Resume from an existing course_outline.json (skips outline generation).",
    )
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="Save progress to output/.checkpoints/ after outline and each lesson.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        metavar="PATH",
        default=None,
        help="Resume lesson generation from a checkpoint.json file.",
    )
    parser.add_argument(
        "--ollama-timeout",
        type=float,
        default=float(os.getenv("OLLAMA_TIMEOUT", "120")),
        help="Timeout in seconds for each Ollama request (default: OLLAMA_TIMEOUT env or 120).",
    )
    parser.add_argument("--list-runs", action="store_true", help="List recent generation reports and exit.")
    parser.add_argument(
        "--validate-outline",
        metavar="PATH",
        default=None,
        help="Validate a course_outline.json file and exit.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=int(os.getenv("MAX_SOURCE_FILES", "0")) or None,
        help="Limit number of source files (0 = no limit). Useful for large folders.",
    )
    parser.add_argument(
        "--export-quiz-csv",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("EXPORT_QUIZ_CSV", "true").lower() in {"1", "true", "yes"},
        help="Export quizzes.csv for LMS import (default: on).",
    )
    parser.add_argument(
        "--export-flashcards",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("EXPORT_FLASHCARDS", "true").lower() in {"1", "true", "yes"},
        help="Export flashcards.json study deck (default: on).",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="JSON file with generation options (CLI flags override config values).",
    )
    parser.add_argument(
        "--export-gift",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("EXPORT_GIFT", "true").lower() in {"1", "true", "yes"},
        help="Export quizzes.gift for Moodle import (default: on).",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print effective generation settings as JSON and exit.",
    )
    parser.add_argument(
        "--init-dirs",
        action="store_true",
        help="Create docs/, output/, logs/, vectorstore/ and a sample doc, then exit.",
    )
    parser.add_argument(
        "--batch-dir",
        metavar="DIR",
        default=None,
        help="Run generation for each *.json config file in a directory (sequential).",
    )
    parser.add_argument(
        "--check-embeddings",
        action="store_true",
        help="Verify the embedding model loads (may download weights on first run), then exit.",
    )
    parser.add_argument(
        "--json-result",
        action="store_true",
        help="On success, print a JSON summary to stdout (for scripts/CI).",
    )
    parser.add_argument(
        "--diff-reports",
        nargs=2,
        metavar=("REPORT_A", "REPORT_B"),
        default=None,
        help="Compare two generation_report.json files and exit.",
    )
    return parser


def default_args() -> argparse.Namespace:
    return build_parser().parse_args([])


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def _cli_progress(stage: str, detail: str) -> None:
    if detail:
        print(f"--- [{stage}] {detail} ---")
    else:
        print(f"--- [{stage}] ---")


def log_message(log_dir: str, message: str) -> None:
    log_file = Path(log_dir) / f"run_{datetime.now().strftime('%Y-%m-%d')}.log"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")


def main() -> None:
    defaults = default_args()
    args = parse_args()

    if args.config:
        try:
            apply_config_file(args, args.config, defaults)
        except Exception as exc:
            print(f"(X) Config error: {exc}")
            sys.exit(1)

    if args.init_dirs:
        created = init_project_directories()
        if created:
            print("[OK] Created:")
            for item in created:
                print(f"  - {item}")
        else:
            print("[OK] Project folders already exist.")
        sys.exit(0)

    if args.print_config:
        import json

        payload = {k: getattr(args, k) for k in sorted(CONFIG_KEYS) if hasattr(args, k)}
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        sys.exit(0)

    if args.validate_outline:
        try:
            outline = load_outline_json(args.validate_outline)
        except Exception as exc:
            print(f"(X) Invalid outline: {exc}")
            sys.exit(1)
        print(f"[OK] Valid outline: {outline.get('course_title', '')}")
        print(f"     Lessons: {len(outline.get('lessons', []))}")
        print(f"     Glossary: {len(outline.get('glossary', []))}")
        sys.exit(0)

    if args.list_runs:
        runs = list_recent_reports(args.output_dir, limit=12)
        if not runs:
            print(f"No generation reports found in '{args.output_dir}'.")
            sys.exit(0)
        print(f"Recent runs in {args.output_dir}:")
        for row in runs:
            quality = row.get("quality_score")
            q_txt = f"{quality}/100 ({row.get('grade', '-')})" if quality is not None else "n/a"
            print(
                f"  {row.get('generated_at', '?')} | {row.get('model', '?')} | "
                f"{row.get('lessons_count', 0)} lessons | quality {q_txt} | {row.get('elapsed_seconds', '?')}s"
            )
            print(f"    {row.get('path', '')}")
        sys.exit(0)

    if args.list_models:
        try:
            models = list_ollama_models()
        except Exception as exc:
            print(f"(X) Could not list models: {exc}")
            sys.exit(1)
        if not models:
            print("(X) No models returned from Ollama.")
            sys.exit(1)
        print("Available Ollama models:")
        for name in models:
            print(f"  - {name}")
        sys.exit(0)

    if args.check_ollama:
        info = check_ollama(args.model)
        if not info.get("ok") or not info.get("model_available"):
            print(f"(X) {format_ollama_message(info)}")
            sys.exit(1)
        print(f"[OK] {format_ollama_message(info)}")
        sys.exit(0)

    if args.check_embeddings:
        info = check_embeddings(args.embedding_model)
        if not info.get("ok"):
            print(f"(X) Embedding model failed: {info.get('error', 'unknown')}")
            sys.exit(1)
        print(f"[OK] Embedding model '{args.embedding_model}' loaded ({info.get('dimensions', '?')} dims).")
        sys.exit(0)

    if args.diff_reports:
        import json

        try:
            diff = diff_generation_reports(args.diff_reports[0], args.diff_reports[1])
        except Exception as exc:
            print(f"(X) {exc}")
            sys.exit(1)
        print(json.dumps(diff, ensure_ascii=False, indent=2))
        sys.exit(0)

    try:
        apply_cli_preset(args)
    except ValueError as exc:
        print(f"(X) {exc}")
        sys.exit(1)

    ensure_directories(args.docs_path, args.db, args.manifest_file, args.output_dir, args.log_dir)

    if args.min_lessons < 1 or args.max_lessons < args.min_lessons:
        print("(X) Invalid lesson range. Check --min-lessons and --max-lessons.")
        sys.exit(1)

    if getattr(args, "outline_only", False) and getattr(args, "from_outline", None):
        print("(X) Use either --outline-only or --from-outline, not both.")
        sys.exit(1)
    if getattr(args, "resume_checkpoint", None) and getattr(args, "from_outline", None):
        print("(X) Use either --resume-checkpoint or --from-outline, not both.")
        sys.exit(1)
    if getattr(args, "resume_checkpoint", None) and getattr(args, "outline_only", False):
        print("(X) Use either --resume-checkpoint or --outline-only, not both.")
        sys.exit(1)

    log_message(args.log_dir, "Starting course and quiz generation pipeline")
    try:
        if getattr(args, "batch_dir", None):
            print(f"--- Batch mode: {args.batch_dir} ---")
            batch_results = run_batch(args.batch_dir, args, defaults, progress=_cli_progress)
            failed = [r for r in batch_results if not r.get("ok")]
            print(f"\n[BATCH] {len(batch_results) - len(failed)}/{len(batch_results)} succeeded.")
            for row in batch_results:
                status = "OK" if row.get("ok") else "FAIL"
                print(f"  [{status}] {row.get('config')}")
                if not row.get("ok"):
                    print(f"         {row.get('error', '')}")
            if getattr(args, "json_result", False):
                import json

                print(json.dumps({"batch": batch_results}, ensure_ascii=False, indent=2, default=str))
            sys.exit(1 if failed else 0)

        print("--- Running generation pipeline... ---")
        result = run_pipeline(args, progress=_cli_progress)
    except DocumentSourceError as exc:
        print(f"(X) {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"(X) Generation failed: {exc}")
        log_message(args.log_dir, f"Generation error: {exc}")
        sys.exit(1)

    if result.get("dry_run"):
        plan = result["plan"]
        print("\n[DRY RUN] No index build, no LLM calls.")
        print(f"Documents ({plan['document_count']}, {plan.get('total_size_human', '?')} total):")
        for item in plan.get("document_details", []):
            print(f"  - {item['name']} ({item.get('size_human', '?')}, {item.get('type', '')})")
        print(f"Recursive scan: {plan['recursive_docs']}")
        print(f"Estimated lessons: ~{plan['estimated_lessons']}")
        print(f"Estimated LLM calls: ~{plan['estimated_llm_calls']}")
        print(f"Estimated runtime:   ~{plan.get('estimated_runtime_minutes', '?')} min")
        print(f"Model: {plan['model']}")
        print("Pipeline steps:")
        for step in plan["pipeline_steps"]:
            print(f"  • {step}")
        sys.exit(0)

    if result.get("outline_only"):
        paths = result["paths"]
        print("\n[SUCCESS] Outline-only run complete.")
        print(f"Course outline: {paths.get('outline', '')}")
        print(f"Time: {result['elapsed_seconds']:.2f}s")
        print(f"Outline RAG used: {result.get('outline_rag_used', False)}")
        sys.exit(0)

    paths = result["paths"]
    n_docs = len(result.get("docs_info", []))
    print("\n[SUCCESS] Generation complete!")
    print(f"Documents indexed: {n_docs} file(s)")
    print(f"Course HTML:        {paths['course_html']}")
    print(f"Course outline:     {paths['outline']}")
    print(f"Pre-test JSON:      {paths['pretest']}")
    print(f"Final quiz JSON:    {paths['quiz']}")
    print(f"Lesson summaries:   {paths['summaries']}")
    print(f"Markdown summary:   {paths['markdown']}")
    if paths.get("course_full_md"):
        print(f"Full course MD:     {paths['course_full_md']}")
    print(f"Metadata:           {paths['metadata']}")
    print(f"Bundle:             {paths['bundle']}")
    print(f"Report:             {paths['report']}")
    if paths.get("docx"):
        print(f"Course DOCX:        {paths['docx']}")
    if paths.get("pdf"):
        print(f"Course PDF:         {paths['pdf']}")
    if paths.get("flashcards"):
        print(f"Flashcards JSON:    {paths['flashcards']}")
    if paths.get("anki_tsv"):
        print(f"Anki deck (TSV):    {paths['anki_tsv']}")
    if paths.get("quizzes_csv"):
        print(f"Quizzes CSV:        {paths['quizzes_csv']}")
    if paths.get("quizzes_gift"):
        print(f"Moodle GIFT:        {paths['quizzes_gift']}")
    if paths.get("output_index"):
        print(f"Output index:       {paths['output_index']}")
    if paths.get("delivery_zip"):
        print(f"Delivery ZIP:       {paths['delivery_zip']}")
    if result.get("checkpoint"):
        print(f"Checkpoint:         {result['checkpoint']}")
    print(f"Time:               {result['elapsed_seconds']:.2f}s")
    quality = result.get("quality", {})
    if quality:
        print(f"Quality score:      {quality.get('overall_score', 'n/a')}/100 (grade {quality.get('grade', '-')})")
        for tip in quality.get("recommendations", []):
            print(f"  → {tip}")
        if quality.get("llm_review"):
            print("\nLLM quality review:")
            print(quality["llm_review"])

    if getattr(args, "open", False) and paths.get("course_html"):
        html_path = Path(paths["course_html"]).resolve()
        if html_path.is_file():
            webbrowser.open(html_path.as_uri())
            print(f"Opened in browser: {html_path}")

    audit_issues = audit_output_paths(paths)
    if audit_issues:
        print("\n(!) Output audit warnings:")
        for issue in audit_issues:
            print(f"  - {issue}")

    if getattr(args, "json_result", False):
        import json

        payload = {
            "ok": True,
            "version": __version__,
            "elapsed_seconds": result.get("elapsed_seconds"),
            "quality": quality,
            "paths": paths,
            "checkpoint": result.get("checkpoint"),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    log_message(
        args.log_dir,
        "Generation complete. "
        f"course={paths['course_html']}, outline={paths['outline']}, pretest={paths['pretest']}, quiz={paths['quiz']}, "
        f"summaries={paths['summaries']}, markdown={paths['markdown']}, metadata={paths['metadata']}, "
        f"bundle={paths['bundle']}, report={paths['report']}, elapsed={result['elapsed_seconds']:.2f}s",
    )
