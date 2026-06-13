import os
import time
from argparse import Namespace
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_ollama import OllamaLLM

from course_generator.bundle_io import load_bundle_json, parse_lesson_indices
from course_generator.checkpoints import checkpoint_path_for_args, load_checkpoint, save_checkpoint
from course_generator.export_phase import run_export_phase

from course_generator.documents import DocCollection, collect_source_files, get_combined_preview_text
from course_generator.generation import (
    generate_course_outline,
    generate_lesson_html_section,
    generate_pretest,
    generate_quiz,
    review_outline,
    review_quiz,
)
from course_generator.preflight import require_ollama
from course_generator.io import load_outline_json, save_outline_json
from course_generator.planning import build_run_plan
from course_generator.rag import load_or_create_vectorstore, retrieve_lesson_context, retrieve_outline_context

ProgressCallback = Callable[[str, str], None]


def _notify(progress: Optional[ProgressCallback], stage: str, detail: str = "") -> None:
    if progress:
        progress(stage, detail)


def _build_llm(args: Namespace) -> OllamaLLM:
    timeout = float(getattr(args, "ollama_timeout", None) or os.getenv("OLLAMA_TIMEOUT", "120"))
    return OllamaLLM(model=args.model, timeout=timeout)


def _save_checkpoint_if_enabled(
    args: Namespace,
    outline: Dict[str, Any],
    lesson_payloads: List[Dict[str, Any]],
    outline_rag_used: bool,
    stage: str,
) -> str:
    if not getattr(args, "checkpoint", False):
        return ""
    path = checkpoint_path_for_args(args)
    return save_checkpoint(
        path,
        outline=outline,
        lesson_payloads=lesson_payloads,
        outline_rag_used=outline_rag_used,
        stage=stage,
        args=args,
    )


def _generate_outline(
    args: Namespace,
    llm: OllamaLLM,
    vectorstore: Any,
    source_files: List[Any],
    labels_base: Any,
    progress: Optional[ProgressCallback],
) -> Tuple[Dict[str, Any], bool]:
    _notify(progress, "outline", "Generating course outline")
    preview_text = get_combined_preview_text(
        source_files,
        labels_base=labels_base,
        max_chars_per_file=args.max_preview_chars_per_file,
    )
    rag_context = ""
    outline_rag_used = False
    if not args.skip_outline_rag:
        _notify(progress, "outline_rag", "Retrieving chunks for outline grounding")
        rag_context = retrieve_outline_context(
            vectorstore,
            source_files,
            labels_base=labels_base,
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
        _notify(progress, "review_outline", "Reviewing outline")
        outline = review_outline(llm, outline, args.language, args.min_lessons, args.max_lessons)
    return outline, outline_rag_used


def _generate_lessons(
    args: Namespace,
    llm: OllamaLLM,
    vectorstore: Any,
    outline: Dict[str, Any],
    progress: Optional[ProgressCallback],
    *,
    initial_payloads: Optional[List[Dict[str, Any]]] = None,
    outline_rag_used: bool = False,
) -> List[Dict[str, Any]]:
    lesson_payloads: List[Dict[str, Any]] = list(initial_payloads or [])
    lessons = outline.get("lessons", [])
    lesson_total = len(lessons)
    start = len(lesson_payloads)
    if start >= lesson_total:
        return lesson_payloads

    for idx, lesson in enumerate(lessons[start:], start=start + 1):
        lesson_title = str(lesson.get("title", f"Lesson {idx}")).strip()
        lesson_goal = str(lesson.get("goal", "")).strip()
        key_points = lesson.get("key_points", []) if isinstance(lesson.get("key_points", []), list) else []
        _notify(progress, "lesson", f"{idx}/{lesson_total}: {lesson_title}")
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
        _save_checkpoint_if_enabled(args, outline, lesson_payloads, outline_rag_used, "lessons")
    return lesson_payloads


def _regenerate_fallback_lessons(
    args: Namespace,
    llm: OllamaLLM,
    vectorstore: Any,
    outline: Dict[str, Any],
    lesson_payloads: List[Dict[str, Any]],
    progress: Optional[ProgressCallback],
    *,
    outline_rag_used: bool = False,
) -> List[Dict[str, Any]]:
    lessons = outline.get("lessons", [])
    for idx, payload in enumerate(lesson_payloads):
        if payload.get("generation_mode") not in {"fallback"}:
            continue
        if idx >= len(lessons):
            continue
        lesson = lessons[idx]
        lesson_title = str(lesson.get("title", f"Lesson {idx + 1}")).strip()
        lesson_goal = str(lesson.get("goal", "")).strip()
        key_points = lesson.get("key_points", []) if isinstance(lesson.get("key_points", []), list) else []
        _notify(progress, "lesson_retry", f"{idx + 1}: {lesson_title} (fallback retry)")
        retrieved_docs = retrieve_lesson_context(vectorstore, lesson_title, key_points, args.top_k, args.retrieval_type)
        lesson_payloads[idx] = generate_lesson_html_section(
            llm,
            lesson_title,
            lesson_goal,
            key_points,
            retrieved_docs,
            args.language,
            args.include_source_excerpts,
        )
        _save_checkpoint_if_enabled(args, outline, lesson_payloads, outline_rag_used, "lessons")
    return lesson_payloads


def _regenerate_lesson_indices(
    args: Namespace,
    llm: OllamaLLM,
    vectorstore: Any,
    outline: Dict[str, Any],
    lesson_payloads: List[Dict[str, Any]],
    indices: List[int],
    progress: Optional[ProgressCallback],
    *,
    outline_rag_used: bool = False,
) -> List[Dict[str, Any]]:
    lessons = outline.get("lessons", [])
    while len(lesson_payloads) < len(lessons):
        lesson_payloads.append({})
    for one_based in indices:
        idx = one_based - 1
        if idx < 0 or idx >= len(lessons):
            continue
        lesson = lessons[idx]
        lesson_title = str(lesson.get("title", f"Lesson {one_based}")).strip()
        lesson_goal = str(lesson.get("goal", "")).strip()
        key_points = lesson.get("key_points", []) if isinstance(lesson.get("key_points", []), list) else []
        _notify(progress, "lesson_regen", f"{one_based}: {lesson_title}")
        retrieved_docs = retrieve_lesson_context(vectorstore, lesson_title, key_points, args.top_k, args.retrieval_type)
        lesson_payloads[idx] = generate_lesson_html_section(
            llm,
            lesson_title,
            lesson_goal,
            key_points,
            retrieved_docs,
            args.language,
            args.include_source_excerpts,
        )
        _save_checkpoint_if_enabled(args, outline, lesson_payloads, outline_rag_used, "lessons")
    return lesson_payloads


def run_pipeline(args: Namespace, progress: Optional[ProgressCallback] = None) -> Dict[str, Any]:
    """
    Run the generation pipeline and return paths + artifacts.

    Modes (via args):
    - dry_run: list documents and estimated work, no LLM/index build.
    - outline_only: stop after saving course_outline.json.
    - from_outline: load outline JSON path instead of generating one.
    - checkpoint: save progress after outline and each lesson.
    - resume_checkpoint: continue lessons from a checkpoint file.
    - from_bundle: load outline/lessons/quizzes from course_bundle.json.
    - artifacts_only: rebuild exports from bundle without LLM calls.
    """
    started = time.time()

    from_bundle_path = getattr(args, "from_bundle", None) or None
    if getattr(args, "artifacts_only", False):
        if not from_bundle_path:
            raise ValueError("--artifacts-only requires --from-bundle")
        bundle = load_bundle_json(from_bundle_path)
        _notify(progress, "artifacts_only", "Rebuilding exports from bundle (no LLM)")
        return run_export_phase(
            args,
            outline=bundle["outline"],
            lesson_payloads=bundle.get("lessons", []),
            docs_info=bundle.get("documents", []),
            pretest_data=bundle.get("pretest", []),
            quiz_data=bundle.get("final_quiz", []),
            outline_rag_used=False,
            llm=None,
            progress=progress,
            started=started,
        )

    _notify(progress, "load_documents", "Collecting source files")
    max_files = getattr(args, "max_files", None) or None
    if max_files is not None and int(max_files) <= 0:
        max_files = None
    dc = collect_source_files(
        args.docs_path,
        recursive=getattr(args, "recursive_docs", False),
        max_files=int(max_files) if max_files else None,
    )
    source_files = dc.files
    labels_base = dc.root

    if getattr(args, "dry_run", False):
        plan = build_run_plan(dc, args)
        elapsed = time.time() - started
        _notify(progress, "dry_run", f"{plan['document_count']} file(s), ~{plan['estimated_llm_calls']} LLM calls")
        return {
            "dry_run": True,
            "plan": plan,
            "docs_info": [],
            "paths": {},
            "elapsed_seconds": round(elapsed, 2),
        }

    _notify(progress, "preflight", f"Checking Ollama model: {args.model}")
    require_ollama(args.model)

    _notify(progress, "vectorstore", f"Building or loading FAISS index ({len(source_files)} file(s))")
    vectorstore, docs_info = load_or_create_vectorstore(args, source_files, labels_base=labels_base)

    from_outline_path = getattr(args, "from_outline", None) or None
    resume_path = getattr(args, "resume_checkpoint", None) or None
    outline_rag_used = False
    initial_lessons: List[Dict[str, Any]] = []
    bundle_pretest: List[Dict[str, Any]] = []
    bundle_quiz: List[Dict[str, Any]] = []
    llm: Optional[OllamaLLM] = None
    checkpoint_path = ""

    if from_bundle_path:
        _notify(progress, "bundle", f"Loading bundle from {from_bundle_path}")
        bundle = load_bundle_json(from_bundle_path)
        outline = bundle["outline"]
        initial_lessons = list(bundle.get("lessons", []))
        bundle_pretest = list(bundle.get("pretest", []))
        bundle_quiz = list(bundle.get("final_quiz", []))
        if bundle.get("documents"):
            docs_info = bundle["documents"]
        _notify(progress, "bundle", f"{len(initial_lessons)} lesson payload(s) loaded from bundle")
    elif resume_path:
        _notify(progress, "checkpoint", f"Resuming from {resume_path}")
        cp = load_checkpoint(resume_path)
        outline = cp["outline"]
        initial_lessons = cp.get("lesson_payloads", [])
        outline_rag_used = bool(cp.get("outline_rag_used", False))
        _notify(progress, "checkpoint", f"{len(initial_lessons)}/{len(outline.get('lessons', []))} lessons already done")
    elif from_outline_path:
        _notify(progress, "outline", f"Loading outline from {from_outline_path}")
        outline = load_outline_json(from_outline_path)
    else:
        _notify(progress, "llm", f"Connecting to Ollama model: {args.model}")
        llm = _build_llm(args)
        outline, outline_rag_used = _generate_outline(args, llm, vectorstore, source_files, labels_base, progress)
        checkpoint_path = _save_checkpoint_if_enabled(args, outline, [], outline_rag_used, "outline")

        if getattr(args, "outline_only", False):
            _notify(progress, "export", "Saving outline only")
            outline_path = save_outline_json(args.output_dir, outline, args.output_prefix)
            elapsed = time.time() - started
            _notify(progress, "done", f"Outline saved in {elapsed:.1f}s")
            return {
                "outline_only": True,
                "docs_info": docs_info,
                "outline": outline,
                "lesson_payloads": [],
                "pretest": [],
                "quiz": [],
                "course_html": "",
                "markdown_summary": "",
                "quality": {},
                "paths": {"outline": outline_path},
                "elapsed_seconds": round(elapsed, 2),
                "outline_rag_used": outline_rag_used,
                "checkpoint": checkpoint_path,
            }

    if llm is None:
        _notify(progress, "llm", f"Connecting to Ollama model: {args.model}")
        llm = _build_llm(args)

    lesson_payloads = _generate_lessons(
        args,
        llm,
        vectorstore,
        outline,
        progress,
        initial_payloads=initial_lessons,
        outline_rag_used=outline_rag_used,
    )
    checkpoint_path = _save_checkpoint_if_enabled(args, outline, lesson_payloads, outline_rag_used, "lessons") or checkpoint_path

    if getattr(args, "regenerate_fallback", False):
        before = sum(1 for p in lesson_payloads if p.get("generation_mode") == "fallback")
        if before:
            _notify(progress, "fallback_retry", f"Regenerating {before} fallback lesson(s)")
            lesson_payloads = _regenerate_fallback_lessons(
                args, llm, vectorstore, outline, lesson_payloads, progress, outline_rag_used=outline_rag_used
            )

    regen_indices = parse_lesson_indices(getattr(args, "regenerate_lessons", None))
    if regen_indices:
        _notify(progress, "lesson_regen", f"Regenerating lesson(s): {regen_indices}")
        lesson_payloads = _regenerate_lesson_indices(
            args,
            llm,
            vectorstore,
            outline,
            lesson_payloads,
            regen_indices,
            progress,
            outline_rag_used=outline_rag_used,
        )

    pretest_data: List[Dict[str, Any]] = []
    if bundle_pretest and not args.skip_pretest:
        pretest_data = bundle_pretest
    elif not args.skip_pretest:
        _notify(progress, "pretest", "Generating diagnostic pre-test")
        pretest_data = generate_pretest(llm, outline, args.pretest_questions, args.difficulty, args.language)

    quiz_data: List[Dict[str, Any]] = []
    if bundle_quiz and not args.skip_final_quiz:
        quiz_data = bundle_quiz
    elif not args.skip_final_quiz:
        _notify(progress, "quiz", "Generating final quiz")
        quiz_data = generate_quiz(llm, outline, lesson_payloads, args.difficulty, args.quiz_questions, args.language)
        if not args.disable_review_pass:
            _notify(progress, "review_quiz", "Reviewing quiz")
            quiz_data = review_quiz(
                llm,
                quiz_data,
                [lesson.get("title", "") for lesson in outline.get("lessons", [])],
                args.language,
            )

    return run_export_phase(
        args,
        outline=outline,
        lesson_payloads=lesson_payloads,
        docs_info=docs_info,
        pretest_data=pretest_data,
        quiz_data=quiz_data,
        outline_rag_used=outline_rag_used,
        llm=llm,
        progress=progress,
        started=started,
        checkpoint_path=checkpoint_path,
    )
