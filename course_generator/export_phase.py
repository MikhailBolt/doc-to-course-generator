import time
from argparse import Namespace
from typing import Any, Callable, Dict, List, Optional

from course_generator.export import create_delivery_zip
from course_generator.flashcards import build_flashcards, flashcards_to_anki_tsv
from course_generator.gift_export import combined_gift_export
from course_generator.html_export import build_course_html, build_full_course_markdown, build_markdown_summary
from course_generator.io import (
    save_anki_tsv,
    save_course_bundle,
    save_course_docx,
    save_course_html,
    save_course_metadata,
    save_course_pdf,
    save_flashcards_json,
    save_full_course_markdown,
    save_generation_report,
    save_lesson_summaries,
    save_markdown_summary,
    save_outline_json,
    save_output_index,
    save_pretest_json,
    save_quiz_json,
    save_quizzes_csv,
    save_quizzes_gift,
    save_run_manifest,
)
from course_generator.llm_review import llm_quality_review
from course_generator.output_index import build_output_index_markdown
from course_generator.quality import compute_quality_score

ProgressCallback = Callable[[str, str], None]


def run_export_phase(
    args: Namespace,
    *,
    outline: Dict[str, Any],
    lesson_payloads: List[Dict[str, Any]],
    docs_info: List[Dict[str, Any]],
    pretest_data: List[Dict[str, Any]],
    quiz_data: List[Dict[str, Any]],
    outline_rag_used: bool,
    llm: Any,
    progress: Optional[ProgressCallback],
    started: float,
    checkpoint_path: str = "",
) -> Dict[str, Any]:
    from course_generator import __version__
    from course_generator.checkpoints import checkpoint_path_for_args, save_checkpoint

    def _notify(stage: str, detail: str = "") -> None:
        if progress:
            progress(stage, detail)

    def _save_checkpoint_if_enabled(
        outline: Dict[str, Any],
        lessons: List[Dict[str, Any]],
        rag_used: bool,
        stage: str,
    ) -> str:
        if not getattr(args, "checkpoint", False):
            return ""
        return save_checkpoint(
            checkpoint_path_for_args(args),
            outline=outline,
            lesson_payloads=lessons,
            outline_rag_used=rag_used,
            stage=stage,
            args=args,
        )

    _notify("export", "Building HTML and saving outputs")
    course_html = build_course_html(
        outline, lesson_payloads, docs_info, pretest_data, quiz_data, args.language, args.include_source_excerpts
    )
    markdown_summary = build_markdown_summary(outline, docs_info, lesson_payloads, args.language)
    full_markdown = build_full_course_markdown(
        outline, docs_info, lesson_payloads, pretest_data, quiz_data, args.language
    )

    quality = compute_quality_score(
        outline, lesson_payloads, pretest_data, quiz_data, args, outline_rag_used=outline_rag_used
    )
    if getattr(args, "quality_llm_review", False) and llm is not None:
        _notify("quality_llm_review", "LLM narrative quality review")
        quality["llm_review"] = llm_quality_review(llm, outline, quality, args.language)

    course_path = save_course_html(args.output_dir, course_html, args.output_prefix)
    outline_path = save_outline_json(args.output_dir, outline, args.output_prefix)
    quiz_path = save_quiz_json(args.output_dir, quiz_data, args.output_prefix)
    pretest_path = save_pretest_json(args.output_dir, pretest_data, args.output_prefix)
    metadata_path = save_course_metadata(args.output_dir, outline, docs_info, args)
    summaries_path = save_lesson_summaries(args.output_dir, lesson_payloads, outline, args.output_prefix)
    markdown_path = save_markdown_summary(args.output_dir, markdown_summary, args.output_prefix)
    full_md_path = save_full_course_markdown(args.output_dir, full_markdown, args.output_prefix)
    docx_path = ""
    if getattr(args, "export_docx", False):
        _notify("docx", "Exporting DOCX summary")
        docx_path = save_course_docx(args.output_dir, markdown_summary, args.output_prefix)
    pdf_path = ""
    if getattr(args, "export_pdf", False):
        _notify("pdf", "Exporting PDF summary")
        pdf_path = save_course_pdf(args.output_dir, markdown_summary, args.output_prefix)
    bundle_path = save_course_bundle(args.output_dir, outline, docs_info, lesson_payloads, pretest_data, quiz_data, args)

    flashcards_path = ""
    anki_path = ""
    flashcards_count = 0
    if getattr(args, "export_flashcards", True):
        cards = build_flashcards(outline, lesson_payloads)
        flashcards_count = len(cards)
        if cards:
            _notify("flashcards", f"Saving {flashcards_count} flashcards")
            flashcards_path = save_flashcards_json(args.output_dir, cards, args.output_prefix)
            anki_path = save_anki_tsv(args.output_dir, flashcards_to_anki_tsv(cards), args.output_prefix)

    quizzes_csv_path = ""
    if getattr(args, "export_quiz_csv", True) and (pretest_data or quiz_data):
        _notify("quiz_csv", "Exporting quizzes.csv")
        quizzes_csv_path = save_quizzes_csv(args.output_dir, pretest_data, quiz_data, args.output_prefix)

    gift_path = ""
    if getattr(args, "export_gift", True) and (pretest_data or quiz_data):
        gift_text = combined_gift_export(pretest_data, quiz_data)
        if gift_text.strip():
            _notify("gift", "Exporting Moodle GIFT quizzes")
            gift_path = save_quizzes_gift(args.output_dir, gift_text, args.output_prefix)

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
        quality_score=quality,
        lesson_payloads=lesson_payloads,
        flashcards_count=flashcards_count,
    )

    paths: Dict[str, str] = {
        "course_html": course_path,
        "outline": outline_path,
        "pretest": pretest_path,
        "quiz": quiz_path,
        "summaries": summaries_path,
        "markdown": markdown_path,
        "course_full_md": full_md_path,
        "metadata": metadata_path,
        "bundle": bundle_path,
        "report": report_path,
    }
    if docx_path:
        paths["docx"] = docx_path
    if pdf_path:
        paths["pdf"] = pdf_path
    if flashcards_path:
        paths["flashcards"] = flashcards_path
    if anki_path:
        paths["anki_tsv"] = anki_path
    if quizzes_csv_path:
        paths["quizzes_csv"] = quizzes_csv_path
    if gift_path:
        paths["quizzes_gift"] = gift_path

    index_md = build_output_index_markdown(
        paths,
        course_title=str(outline.get("course_title", "")),
        quality=quality,
        elapsed_seconds=elapsed,
    )
    index_path = save_output_index(args.output_dir, index_md, args.output_prefix)
    paths["output_index"] = index_path

    manifest_path = save_run_manifest(
        args.output_dir,
        {
            "generator_version": __version__,
            "course_title": outline.get("course_title", ""),
            "elapsed_seconds": round(elapsed, 2),
            "quality_score": quality.get("overall_score"),
            "grade": quality.get("grade"),
            "lessons_count": len(outline.get("lessons", [])),
            "lessons_fallback_count": sum(
                1 for p in lesson_payloads if p.get("generation_mode") == "fallback"
            ),
            "paths": paths,
        },
        args.output_prefix,
    )
    paths["run_manifest"] = manifest_path

    zip_path = ""
    if not getattr(args, "no_delivery_zip", False):
        _notify("zip", "Packaging delivery ZIP")
        zip_path = create_delivery_zip(
            paths,
            args.output_dir,
            args.output_prefix,
            course_title=str(outline.get("course_title", "")),
            quality=quality,
            elapsed_seconds=elapsed,
        )
        paths["delivery_zip"] = zip_path

    checkpoint_path = _save_checkpoint_if_enabled(outline, lesson_payloads, outline_rag_used, "complete") or checkpoint_path
    _notify("done", f"Finished in {elapsed:.1f}s — quality {quality['overall_score']}/100")

    result: Dict[str, Any] = {
        "docs_info": docs_info,
        "outline": outline,
        "lesson_payloads": lesson_payloads,
        "pretest": pretest_data,
        "quiz": quiz_data,
        "course_html": course_html,
        "markdown_summary": markdown_summary,
        "quality": quality,
        "paths": paths,
        "elapsed_seconds": round(elapsed, 2),
        "outline_rag_used": outline_rag_used,
    }
    if checkpoint_path:
        result["checkpoint"] = checkpoint_path
    return result
