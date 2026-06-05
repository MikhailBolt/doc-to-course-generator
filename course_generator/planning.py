from argparse import Namespace
from typing import Any, Dict, List

from course_generator.documents import DocCollection, document_display_name


def _human_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def estimate_runtime_minutes(llm_calls: int, lesson_count: int) -> float:
    """Rough wall-clock estimate for local Ollama (seconds per call vary by hardware)."""
    lesson_calls = max(0, lesson_count)
    other_calls = max(0, llm_calls - lesson_calls)
    seconds = lesson_calls * 75 + other_calls * 40
    return round(seconds / 60, 1)


def estimate_llm_calls(args: Namespace, lesson_count: int) -> int:
    calls = 1  # outline
    if not args.disable_review_pass:
        calls += 1
    calls += max(0, lesson_count)
    if not args.skip_pretest:
        calls += 1
    if not args.skip_final_quiz:
        calls += 1
        if not args.disable_review_pass:
            calls += 1
    if getattr(args, "quality_llm_review", False):
        calls += 1
    return calls


def build_run_plan(dc: DocCollection, args: Namespace) -> Dict[str, Any]:
    max_files = getattr(args, "max_files", None)
    if max_files is not None and int(max_files) <= 0:
        max_files = None
    lesson_est = (int(args.min_lessons) + int(args.max_lessons)) // 2
    documents = [document_display_name(p, dc.root) for p in dc.files]
    document_details = [
        {
            "name": document_display_name(p, dc.root),
            "size_bytes": p.stat().st_size,
            "size_human": _human_size(p.stat().st_size),
            "type": p.suffix.lower().lstrip("."),
        }
        for p in dc.files
    ]
    total_bytes = sum(d["size_bytes"] for d in document_details)

    steps: List[str] = ["Build or load FAISS index"]
    if getattr(args, "from_outline", None):
        steps.append(f"Load outline from {args.from_outline}")
    else:
        steps.append("Generate course outline")
        if not args.skip_outline_rag:
            steps.append("Retrieve RAG chunks for outline")
        if not args.disable_review_pass:
            steps.append("Review outline")

    if getattr(args, "checkpoint", False):
        steps.append("Save checkpoints after outline and each lesson")
    if getattr(args, "resume_checkpoint", None):
        steps.insert(1, f"Resume from checkpoint {args.resume_checkpoint}")

    if not getattr(args, "outline_only", False):
        steps.append(f"Generate ~{lesson_est} lesson sections (RAG per lesson)")
        if not args.skip_pretest:
            steps.append("Generate pre-test")
        if not args.skip_final_quiz:
            steps.append("Generate final quiz")
            if not args.disable_review_pass:
                steps.append("Review quiz")
        steps.append("Build HTML and save artifacts")
        if getattr(args, "export_docx", False):
            steps.append("Export DOCX summary")
        if getattr(args, "export_pdf", False):
            steps.append("Export PDF summary")
        if getattr(args, "export_flashcards", True):
            steps.append("Export flashcards.json")
        if getattr(args, "export_quiz_csv", True):
            steps.append("Export quizzes.csv")
        if getattr(args, "export_gift", True):
            steps.append("Export quizzes.gift (Moodle)")
        steps.append("Write OUTPUT_INDEX.md")
        if not getattr(args, "no_delivery_zip", False):
            steps.append("Package delivery ZIP")

    return {
        "documents": documents,
        "document_details": document_details,
        "total_size_bytes": total_bytes,
        "total_size_human": _human_size(total_bytes),
        "document_count": len(documents),
        "documents_truncated": dc.truncated,
        "documents_total_found": dc.total_found or len(documents),
        "max_files": int(max_files) if max_files else None,
        "recursive_docs": bool(getattr(args, "recursive_docs", False)),
        "docs_root": str(dc.root),
        "estimated_lessons": lesson_est,
        "estimated_llm_calls": estimate_llm_calls(args, lesson_est),
        "estimated_runtime_minutes": estimate_runtime_minutes(
            estimate_llm_calls(args, lesson_est), lesson_est
        ),
        "model": args.model,
        "embedding_model": args.embedding_model,
        "pipeline_steps": steps,
        "outline_only": bool(getattr(args, "outline_only", False)),
        "from_outline": getattr(args, "from_outline", None),
    }
