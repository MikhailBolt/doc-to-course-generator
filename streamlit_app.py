import os
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Any, List, Optional

import streamlit as st

from course_generator import __version__
from course_generator.cli import ensure_directories
from course_generator.health import check_ollama, format_ollama_message, list_ollama_models
from course_generator.presets import PRESET_NAMES, PRESETS
from course_generator.constants import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DB_FAISS_PATH,
    DEFAULT_DOCS_PATH,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LANGUAGE,
    DEFAULT_LLM_MODEL,
    DEFAULT_LOG_DIR,
    DEFAULT_MANIFEST_FILE,
    DEFAULT_MAX_LESSONS,
    DEFAULT_MAX_PREVIEW_CHARS_PER_FILE,
    DEFAULT_MIN_LESSONS,
    DEFAULT_OUTLINE_RAG_MAX_CHARS,
    DEFAULT_OUTLINE_RAG_MAX_CHUNKS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_PREFIX,
    DEFAULT_PRETEST_QUESTIONS,
    DEFAULT_QUIZ_QUESTIONS,
    DEFAULT_RETRIEVAL_TYPE,
    DEFAULT_TOP_K,
    SUPPORTED_EXTENSIONS,
)
from course_generator.checkpoints import find_recent_checkpoints
from course_generator.history import list_recent_reports
from course_generator.pipeline import run_pipeline
from course_generator.rag import load_existing_vectorstore, retrieve_lesson_context
from course_generator.rag import load_existing_vectorstore, retrieve_lesson_context
from course_generator.user_settings import load_user_settings, save_user_settings


def _save_uploads(uploaded_files: List[Any], target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for uf in uploaded_files:
        name = Path(uf.name).name
        if not name:
            continue
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            continue
        (target_dir / name).write_bytes(uf.getbuffer())


def _make_args(
    *,
    docs_path: str,
    output_prefix: str,
    model: str,
    embedding_model: str,
    language: str,
    difficulty: str,
    retrieval_type: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    min_lessons: int,
    max_lessons: int,
    quiz_questions: int,
    pretest_questions: int,
    skip_pretest: bool,
    skip_final_quiz: bool,
    include_source_excerpts: bool,
    disable_review_pass: bool,
    rebuild: bool,
    skip_outline_rag: bool,
    outline_rag_max_chunks: int,
    outline_rag_max_chars: int,
    max_preview_chars_per_file: int,
    export_docx: bool,
    export_pdf: bool,
    quality_llm_review: bool,
    recursive_docs: bool,
    dry_run: bool,
    outline_only: bool,
    from_outline: Optional[str],
    checkpoint: bool,
    resume_checkpoint: Optional[str],
    ollama_timeout: float,
    export_quiz_csv: bool,
    export_flashcards: bool,
    export_gift: bool,
    regenerate_fallback: bool,
    max_files: int,
) -> Namespace:
    return Namespace(
        docs_path=docs_path,
        db=DEFAULT_DB_FAISS_PATH,
        manifest_file=DEFAULT_MANIFEST_FILE,
        output_dir=DEFAULT_OUTPUT_DIR,
        log_dir=DEFAULT_LOG_DIR,
        embedding_model=embedding_model,
        model=model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        quiz_questions=quiz_questions,
        pretest_questions=pretest_questions,
        difficulty=difficulty,
        retrieval_type=retrieval_type,
        language=language,
        max_preview_chars_per_file=max_preview_chars_per_file,
        output_prefix=output_prefix,
        min_lessons=min_lessons,
        max_lessons=max_lessons,
        disable_review_pass=disable_review_pass,
        skip_pretest=skip_pretest,
        skip_final_quiz=skip_final_quiz,
        include_source_excerpts=include_source_excerpts,
        rebuild=rebuild,
        skip_outline_rag=skip_outline_rag,
        outline_rag_max_chunks=outline_rag_max_chunks,
        outline_rag_max_chars=outline_rag_max_chars,
        export_docx=export_docx,
        export_pdf=export_pdf,
        quality_llm_review=quality_llm_review,
        no_delivery_zip=False,
        preset=None,
        recursive_docs=recursive_docs,
        dry_run=dry_run,
        outline_only=outline_only,
        from_outline=from_outline or None,
        checkpoint=checkpoint,
        resume_checkpoint=resume_checkpoint or None,
        ollama_timeout=ollama_timeout,
        export_quiz_csv=export_quiz_csv,
        export_flashcards=export_flashcards,
        export_gift=export_gift,
        regenerate_fallback=regenerate_fallback,
        max_files=max_files if max_files > 0 else None,
    )


def main() -> None:
    st.set_page_config(page_title="Doc-to-Course Generator", layout="wide")

    st.title("Doc-to-Course Generator")
    st.caption(
        f"v{__version__} — Generate an HTML training course + quizzes from PDF/TXT/MD "
        "using local Ollama + FAISS RAG."
    )

    saved = load_user_settings()

    with st.sidebar:
        st.header("Ollama")
        model_default = saved.get("model") or os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
        ollama_models: List[str] = []
        try:
            ollama_models = list_ollama_models(timeout=3)
        except Exception:
            ollama_models = []

        if ollama_models:
            default_idx = 0
            for i, name in enumerate(ollama_models):
                if name == model_default or name.startswith(f"{model_default}:"):
                    default_idx = i
                    break
            model = st.selectbox("Ollama model", ollama_models, index=default_idx, key="ollama_model")
        else:
            model = st.text_input("Ollama model", value=model_default, key="ollama_model")
            st.caption("Start Ollama to load models into the list automatically.")

        if st.button("Check Ollama", use_container_width=True):
            info = check_ollama(model)
            if not info.get("ok") or not info.get("model_available"):
                st.error(format_ollama_message(info))
            else:
                st.success(format_ollama_message(info))

        st.header("Inputs")
        source_mode = st.radio("Source", ["Upload files", "Use local docs folder"], index=0)

        uploaded_files: List[Any] = []
        docs_path: Optional[str] = None
        recursive_docs = False

        if source_mode == "Upload files":
            uploaded_files = st.file_uploader(
                "Upload PDF / TXT / MD",
                type=[ext.lstrip(".") for ext in sorted(SUPPORTED_EXTENSIONS)],
                accept_multiple_files=True,
            )
        else:
            docs_default = saved.get("docs_path") or os.getenv("DOCS_PATH", DEFAULT_DOCS_PATH)
            docs_path = st.text_input("Docs path", value=docs_default)
            recursive_docs = st.checkbox(
                "Scan subfolders",
                value=saved.get(
                    "recursive_docs",
                    os.getenv("DOCS_RECURSIVE", "").lower() in {"1", "true", "yes"},
                ),
            )

        st.header("Generation")
        preset_default = saved.get("preset", PRESET_NAMES[0])
        preset_index = PRESET_NAMES.index(preset_default) if preset_default in PRESET_NAMES else 0
        preset_name = st.selectbox("Preset", PRESET_NAMES, index=preset_index)
        preset_cfg = PRESETS.get(preset_name, {}) if preset_name != "Custom" else {}

        lang_default = saved.get("language", DEFAULT_LANGUAGE)
        language = st.selectbox("Language", ["en", "ru"], index=0 if lang_default == "en" else 1)
        diff_default = saved.get("difficulty", "medium")
        diff_index = ["easy", "medium", "hard"].index(diff_default) if diff_default in ("easy", "medium", "hard") else 1
        difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"], index=diff_index)
        embedding_model = st.text_input(
            "Embedding model",
            value=saved.get("embedding_model") or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        )
        ret_default = saved.get("retrieval_type", DEFAULT_RETRIEVAL_TYPE)
        retrieval_type = st.selectbox("Retrieval", ["similarity", "mmr"], index=0 if ret_default == "similarity" else 1)

        chunk_size = st.number_input(
            "Chunk size",
            min_value=200,
            max_value=4000,
            value=int(saved.get("chunk_size", DEFAULT_CHUNK_SIZE)),
            step=50,
        )
        chunk_overlap = st.number_input(
            "Chunk overlap",
            min_value=0,
            max_value=2000,
            value=int(saved.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)),
            step=50,
        )
        top_k = st.number_input(
            "Top-k chunks per lesson",
            min_value=1,
            max_value=20,
            value=int(preset_cfg.get("top_k", DEFAULT_TOP_K)),
            step=1,
        )
        max_preview_chars_per_file = st.number_input(
            "Max preview chars per file",
            min_value=500,
            max_value=50000,
            value=DEFAULT_MAX_PREVIEW_CHARS_PER_FILE,
            step=500,
        )

        st.header("Course shape")
        min_lessons = st.number_input(
            "Min lessons",
            min_value=1,
            max_value=30,
            value=int(preset_cfg.get("min_lessons", DEFAULT_MIN_LESSONS)),
            step=1,
        )
        max_lessons = st.number_input(
            "Max lessons",
            min_value=1,
            max_value=50,
            value=int(preset_cfg.get("max_lessons", DEFAULT_MAX_LESSONS)),
            step=1,
        )

        st.header("Quizzes")
        quiz_questions = st.number_input(
            "Final quiz questions",
            min_value=0,
            max_value=50,
            value=int(preset_cfg.get("quiz_questions", DEFAULT_QUIZ_QUESTIONS)),
            step=1,
        )
        pretest_questions = st.number_input(
            "Pre-test questions",
            min_value=0,
            max_value=50,
            value=int(preset_cfg.get("pretest_questions", DEFAULT_PRETEST_QUESTIONS)),
            step=1,
        )
        skip_pretest = st.checkbox("Skip pre-test", value=bool(preset_cfg.get("skip_pretest", False)))
        skip_final_quiz = st.checkbox("Skip final quiz", value=bool(preset_cfg.get("skip_final_quiz", False)))

        st.header("Quality / diagnostics")
        include_source_excerpts = st.checkbox(
            "Include source excerpts",
            value=bool(preset_cfg.get("include_source_excerpts", False)),
        )
        disable_review_pass = st.checkbox(
            "Disable review pass",
            value=bool(preset_cfg.get("disable_review_pass", False)),
        )
        rebuild = st.checkbox("Force rebuild FAISS index", value=False)
        export_docx = st.checkbox("Export DOCX summary", value=False)
        export_pdf = st.checkbox("Export PDF summary", value=False)
        quality_llm_review = st.checkbox("LLM quality review (extra LLM call)", value=False)

        st.header("Outline grounding (RAG)")
        skip_outline_rag = st.checkbox("Skip outline RAG", value=bool(preset_cfg.get("skip_outline_rag", False)))
        outline_rag_max_chunks = st.number_input("Outline RAG max chunks", min_value=0, max_value=200, value=DEFAULT_OUTLINE_RAG_MAX_CHUNKS, step=1)
        outline_rag_max_chars = st.number_input("Outline RAG max chars", min_value=1000, max_value=100000, value=DEFAULT_OUTLINE_RAG_MAX_CHARS, step=1000)

        output_prefix = st.text_input(
            "Output prefix",
            value=saved.get("output_prefix") or os.getenv("OUTPUT_PREFIX", DEFAULT_OUTPUT_PREFIX),
        )

        st.header("Run mode")
        dry_run = st.checkbox("Dry run (list files, no LLM)", value=False)
        outline_only = st.checkbox(
            "Outline only",
            value=bool(preset_cfg.get("outline_only", False)),
        )
        from_outline = st.text_input(
            "Resume from outline JSON (optional)",
            value=saved.get("from_outline", ""),
            placeholder="output/course_outline.json",
        )
        checkpoint = st.checkbox("Save checkpoints (resume if interrupted)", value=False)
        resume_checkpoint = st.text_input(
            "Resume from checkpoint JSON (optional)",
            value=saved.get("resume_checkpoint", ""),
            placeholder="output/.checkpoints/default/checkpoint.json",
        )
        ollama_timeout = st.number_input(
            "Ollama timeout (seconds)",
            min_value=30,
            max_value=900,
            value=int(saved.get("ollama_timeout", 120)),
            step=30,
        )
        max_files = st.number_input(
            "Max source files (0 = unlimited)",
            min_value=0,
            max_value=500,
            value=int(saved.get("max_files", 0)),
            step=1,
        )
        export_quiz_csv = st.checkbox("Export quizzes.csv", value=True)
        export_flashcards = st.checkbox("Export flashcards.json", value=True)
        export_gift = st.checkbox("Export Moodle GIFT (quizzes.gift)", value=True)
        regenerate_fallback = st.checkbox("Retry fallback lessons", value=False)

        run_btn = st.button("Generate", type="primary", use_container_width=True)

    if not run_btn:
        st.info("Pick a source (upload or docs folder), then click **Generate**.")
        with st.expander("Explore saved FAISS index (no generation)"):
            st.caption("Search chunks from the last built index in `vectorstore/`.")
            rag_query = st.text_input("Search query", placeholder="e.g. main concepts from chapter 2")
            rag_top_k = st.slider("Top-k chunks", 1, 12, 4)
            if st.button("Search index", use_container_width=True) and rag_query.strip():
                vs = load_existing_vectorstore(DEFAULT_DB_FAISS_PATH, embedding_model)
                if vs is None:
                    st.warning("No FAISS index found. Run generation once or place index under vectorstore/.")
                else:
                    hits = retrieve_lesson_context(vs, rag_query.strip(), [], int(rag_top_k), retrieval_type)
                    for i, doc in enumerate(hits, start=1):
                        name = doc.metadata.get("document_name", "?")
                        page = doc.metadata.get("page")
                        page_n = page + 1 if isinstance(page, int) else "?"
                        st.markdown(f"**{i}.** `{name}` p.{page_n} · chunk {doc.metadata.get('chunk_id', '?')}")
                        st.text(doc.page_content[:1200])
        checkpoints = find_recent_checkpoints(DEFAULT_OUTPUT_DIR, limit=5)
        if checkpoints:
            st.subheader("Saved checkpoints")
            st.caption("Paste a path into **Resume from checkpoint** in the sidebar to continue.")
            st.dataframe(checkpoints, use_container_width=True, hide_index=True)

        recent = list_recent_reports(DEFAULT_OUTPUT_DIR, limit=6)
        if recent:
            st.subheader("Recent runs")
            st.dataframe(recent, use_container_width=True, hide_index=True)
        return

    if int(min_lessons) > int(max_lessons):
        st.error("Min lessons must be ≤ max lessons.")
        return

    if outline_only and from_outline and from_outline.strip():
        st.error("Choose either **Outline only** or **Resume from outline**, not both.")
        return
    if resume_checkpoint and resume_checkpoint.strip() and from_outline and from_outline.strip():
        st.error("Choose either **Resume from checkpoint** or **Resume from outline**, not both.")
        return

    if not dry_run:
        ollama_info = check_ollama(model)
        if not ollama_info.get("ok") or not ollama_info.get("model_available"):
            st.error(format_ollama_message(ollama_info))
            return

    if source_mode == "Upload files":
        if not uploaded_files:
            st.error("Upload at least one file.")
            return
        tmp_root = Path(".tmp_uploads")
        tmp_root.mkdir(exist_ok=True)
        upload_dir = Path(tempfile.mkdtemp(prefix="dtcg_", dir=str(tmp_root)))
        _save_uploads(uploaded_files, upload_dir)
        docs_path_final = str(upload_dir)
    else:
        if not docs_path or not docs_path.strip():
            st.error("Provide a docs path.")
            return
        docs_path_final = docs_path.strip()

    args = _make_args(
        docs_path=docs_path_final,
        output_prefix=output_prefix,
        model=model,
        embedding_model=embedding_model,
        language=language,
        difficulty=difficulty,
        retrieval_type=retrieval_type,
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
        top_k=int(top_k),
        min_lessons=int(min_lessons),
        max_lessons=int(max_lessons),
        quiz_questions=int(quiz_questions),
        pretest_questions=int(pretest_questions),
        skip_pretest=bool(skip_pretest),
        skip_final_quiz=bool(skip_final_quiz),
        include_source_excerpts=bool(include_source_excerpts),
        disable_review_pass=bool(disable_review_pass),
        rebuild=bool(rebuild),
        skip_outline_rag=bool(skip_outline_rag),
        outline_rag_max_chunks=int(outline_rag_max_chunks),
        outline_rag_max_chars=int(outline_rag_max_chars),
        max_preview_chars_per_file=int(max_preview_chars_per_file),
        export_docx=bool(export_docx),
        export_pdf=bool(export_pdf),
        quality_llm_review=bool(quality_llm_review),
        recursive_docs=bool(recursive_docs),
        dry_run=bool(dry_run),
        outline_only=bool(outline_only),
        from_outline=from_outline.strip() if from_outline and from_outline.strip() else None,
        checkpoint=bool(checkpoint),
        resume_checkpoint=resume_checkpoint.strip() if resume_checkpoint and resume_checkpoint.strip() else None,
        ollama_timeout=float(ollama_timeout),
        export_quiz_csv=bool(export_quiz_csv),
        export_flashcards=bool(export_flashcards),
        export_gift=bool(export_gift),
        regenerate_fallback=bool(regenerate_fallback),
        max_files=int(max_files),
    )

    ensure_directories(args.docs_path, args.db, args.manifest_file, args.output_dir, args.log_dir)

    save_user_settings(
        {
            "model": model,
            "embedding_model": embedding_model,
            "language": language,
            "difficulty": difficulty,
            "retrieval_type": retrieval_type,
            "chunk_size": int(chunk_size),
            "chunk_overlap": int(chunk_overlap),
            "preset": preset_name,
            "docs_path": docs_path if source_mode == "Use local docs folder" else "",
            "recursive_docs": bool(recursive_docs),
            "output_prefix": output_prefix,
            "from_outline": from_outline.strip() if from_outline else "",
            "resume_checkpoint": resume_checkpoint.strip() if resume_checkpoint else "",
            "ollama_timeout": int(ollama_timeout),
        }
    )

    lesson_progress = st.progress(0.0, text="Waiting to start…")

    with st.status("Generating…", expanded=True) as status:
        status.write("Pipeline started…")

        def on_progress(stage: str, detail: str) -> None:
            line = f"**{stage}** — {detail}" if detail else f"**{stage}**"
            status.write(line)
            if stage == "lesson" and "/" in detail:
                try:
                    left, rest = detail.split("/", 1)
                    current = int(left.strip())
                    total = int(rest.split(":", 1)[0].strip())
                    if total > 0:
                        lesson_progress.progress(min(1.0, current / total), text=f"Lesson {current}/{total}")
                except ValueError:
                    pass

        try:
            result = run_pipeline(args, progress=on_progress)
        except Exception as exc:
            status.update(label="Failed", state="error")
            st.exception(exc)
            return
        status.update(label="Done", state="complete")
        lesson_progress.progress(1.0, text="Complete")

    if result.get("dry_run"):
        plan = result["plan"]
        st.success(f"Dry run in {result['elapsed_seconds']}s — no LLM calls.")
        st.markdown(
            f"**{plan['document_count']}** file(s) · **{plan.get('total_size_human', '?')}** · "
            f"~**{plan['estimated_llm_calls']}** LLM calls · ~**{plan.get('estimated_runtime_minutes', '?')}** min · "
            f"model `{plan['model']}`"
        )
        if plan.get("documents_truncated"):
            st.warning(
                f"File list truncated: using {plan['document_count']} of {plan.get('documents_total_found', '?')} "
                f"(max-files={plan.get('max_files')})."
            )
        details = plan.get("document_details") or []
        if details:
            st.dataframe(
                [{"file": d["name"], "size": d["size_human"], "type": d["type"]} for d in details],
                use_container_width=True,
                hide_index=True,
            )
        elif plan["documents"]:
            st.code("\n".join(plan["documents"]))
        st.markdown("**Planned steps**")
        for step in plan["pipeline_steps"]:
            st.markdown(f"- {step}")
        return

    paths = result["paths"]
    quality = result.get("quality", {})

    if result.get("outline_only"):
        st.success(
            f"Outline saved in {result['elapsed_seconds']}s · outline RAG: {result.get('outline_rag_used', False)}"
        )
        outline_path = paths.get("outline")
        if outline_path and Path(outline_path).exists():
            st.download_button(
                "Course outline (JSON)",
                data=Path(outline_path).read_bytes(),
                file_name=Path(outline_path).name,
                mime="application/json",
            )
        outline = result.get("outline", {})
        if outline:
            st.markdown("**Course title**")
            st.write(outline.get("course_title", ""))
            st.markdown("**Description**")
            st.write(outline.get("course_description", ""))
        return

    success_msg = (
        f"Done in {result['elapsed_seconds']}s · outline RAG: {result.get('outline_rag_used', False)} · "
        f"quality **{quality.get('overall_score', '—')}/100** (grade **{quality.get('grade', '—')}**)"
    )
    if result.get("checkpoint"):
        success_msg += f" · checkpoint `{result['checkpoint']}`"
    st.success(success_msg)

    if quality.get("checks"):
        with st.expander("Quality breakdown", expanded=True):
            st.progress(min(1.0, float(quality.get("overall_score", 0)) / 100.0))
            for check in quality["checks"]:
                icon = "✅" if check.get("passed") else "⚠️"
                st.markdown(
                    f"{icon} **{check.get('label')}** — {check.get('score')}/{check.get('max')}  \n"
                    f"<span style='color:#64748b'>{check.get('detail', '')}</span>",
                    unsafe_allow_html=True,
                )

    recs = quality.get("recommendations", [])
    if recs:
        with st.expander("Suggestions to improve quality"):
            for tip in recs:
                st.markdown(f"- {tip}")

    if quality.get("llm_review"):
        with st.expander("LLM quality review", expanded=True):
            st.markdown(quality["llm_review"])

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.subheader("Downloads")
        for label, key, mime in [
            ("Course HTML", "course_html", "text/html"),
            ("Course outline (JSON)", "outline", "application/json"),
            ("Pre-test (JSON)", "pretest", "application/json"),
            ("Final quiz (JSON)", "quiz", "application/json"),
            ("Bundle (JSON)", "bundle", "application/json"),
            ("Generation report (JSON)", "report", "application/json"),
            ("Markdown summary", "markdown", "text/markdown"),
            ("Full course (Markdown)", "course_full_md", "text/markdown"),
            ("Flashcards (JSON)", "flashcards", "application/json"),
            ("Anki deck (TSV)", "anki_tsv", "text/plain"),
            ("Quizzes (CSV)", "quizzes_csv", "text/csv"),
            ("Moodle GIFT", "quizzes_gift", "text/plain"),
            ("Output index", "output_index", "text/markdown"),
            ("Run manifest", "run_manifest", "application/json"),
        ]:
            p = paths.get(key)
            if p and Path(p).exists():
                st.download_button(label, data=Path(p).read_bytes(), file_name=Path(p).name, mime=mime)

        docx_p = paths.get("docx")
        if docx_p and Path(docx_p).exists():
            st.download_button(
                "Course summary (DOCX)",
                data=Path(docx_p).read_bytes(),
                file_name=Path(docx_p).name,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

        pdf_p = paths.get("pdf")
        if pdf_p and Path(pdf_p).exists():
            st.download_button(
                "Course summary (PDF)",
                data=Path(pdf_p).read_bytes(),
                file_name=Path(pdf_p).name,
                mime="application/pdf",
            )

        zip_p = paths.get("delivery_zip")
        if zip_p and Path(zip_p).exists():
            st.download_button(
                "All outputs (ZIP)",
                data=Path(zip_p).read_bytes(),
                file_name=Path(zip_p).name,
                mime="application/zip",
            )

        cards_path = paths.get("flashcards")
        if cards_path and Path(cards_path).exists():
            import json

            cards = json.loads(Path(cards_path).read_text(encoding="utf-8"))
            if cards:
                with st.expander(f"Flashcards preview ({len(cards)})", expanded=False):
                    st.dataframe(cards[:12], use_container_width=True, hide_index=True)

        outline = result.get("outline", {})
        if outline:
            st.markdown("**Course title**")
            st.write(outline.get("course_title", ""))
            st.markdown("**Description**")
            st.write(outline.get("course_description", ""))

    with col_b:
        st.subheader("Preview")
        st.caption("Rendered `course.html` (may take a second to load).")
        st.components.v1.html(result["course_html"], height=800, scrolling=True)


if __name__ == "__main__":
    main()

