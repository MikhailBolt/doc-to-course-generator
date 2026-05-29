from argparse import Namespace
from typing import Any, Dict, List

from course_generator.documents import DocCollection, document_display_name


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
    lesson_est = (int(args.min_lessons) + int(args.max_lessons)) // 2
    documents = [document_display_name(p, dc.root) for p in dc.files]

    steps: List[str] = ["Build or load FAISS index"]
    if getattr(args, "from_outline", None):
        steps.append(f"Load outline from {args.from_outline}")
    else:
        steps.append("Generate course outline")
        if not args.skip_outline_rag:
            steps.append("Retrieve RAG chunks for outline")
        if not args.disable_review_pass:
            steps.append("Review outline")

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
        if not getattr(args, "no_delivery_zip", False):
            steps.append("Package delivery ZIP")

    return {
        "documents": documents,
        "document_count": len(documents),
        "recursive_docs": bool(getattr(args, "recursive_docs", False)),
        "docs_root": str(dc.root),
        "estimated_lessons": lesson_est,
        "estimated_llm_calls": estimate_llm_calls(args, lesson_est),
        "model": args.model,
        "embedding_model": args.embedding_model,
        "pipeline_steps": steps,
        "outline_only": bool(getattr(args, "outline_only", False)),
        "from_outline": getattr(args, "from_outline", None),
    }
