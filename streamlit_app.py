import os
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Any, List, Optional

import streamlit as st

from course_generator.cli import ensure_directories
from course_generator.health import check_ollama, list_ollama_models
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
from course_generator.history import list_recent_reports
from course_generator.pipeline import run_pipeline
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
    quality_llm_review: bool,
    recursive_docs: bool,
    dry_run: bool,
    outline_only: bool,
    from_outline: Optional[str],
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
        quality_llm_review=quality_llm_review,
        no_delivery_zip=False,
        preset=None,
        recursive_docs=recursive_docs,
        dry_run=dry_run,
        outline_only=outline_only,
        from_outline=from_outline or None,
    )


def main() -> None:
    st.set_page_config(page_title="Doc-to-Course Generator", layout="wide")

    st.title("Doc-to-Course Generator")
    st.caption("Generate an HTML training course + quizzes from PDF/TXT/MD using local Ollama + FAISS RAG.")

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
            if not info.get("ok"):
                st.error(f"Ollama unreachable: {info.get('error', 'unknown')}")
            elif not info.get("model_available"):
                st.warning(f"Model `{model}` not found. Try: {', '.join(info.get('models_sample', [])[:5])}")
            else:
                st.success(f"Ollama OK — `{model}` is available.")

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

        run_btn = st.button("Generate", type="primary", use_container_width=True)

    if not run_btn:
        st.info("Pick a source (upload or docs folder), then click **Generate**.")
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

    if not dry_run:
        ollama_info = check_ollama(model)
        if not ollama_info.get("ok"):
            st.error(f"Start Ollama first (could not reach {ollama_info.get('host')}): {ollama_info.get('error', '')}")
            return
        if not ollama_info.get("model_available"):
            st.error(f"Model `{model}` is not available in Ollama. Run `ollama pull {model}` or pick another model.")
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
        quality_llm_review=bool(quality_llm_review),
        recursive_docs=bool(recursive_docs),
        dry_run=bool(dry_run),
        outline_only=bool(outline_only),
        from_outline=from_outline.strip() if from_outline and from_outline.strip() else None,
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
        }
    )

    with st.status("Generating…", expanded=True) as status:
        status.write("Pipeline started…")

        def on_progress(stage: str, detail: str) -> None:
            line = f"**{stage}** — {detail}" if detail else f"**{stage}**"
            status.write(line)

        try:
            result = run_pipeline(args, progress=on_progress)
        except Exception as exc:
            status.update(label="Failed", state="error")
            st.exception(exc)
            return
        status.update(label="Done", state="complete")

    if result.get("dry_run"):
        plan = result["plan"]
        st.success(f"Dry run in {result['elapsed_seconds']}s — no LLM calls.")
        st.markdown(f"**{plan['document_count']}** file(s) · ~**{plan['estimated_llm_calls']}** LLM calls · model `{plan['model']}`")
        if plan["documents"]:
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

    st.success(
        f"Done in {result['elapsed_seconds']}s · outline RAG: {result.get('outline_rag_used', False)} · "
        f"quality **{quality.get('overall_score', '—')}/100** (grade **{quality.get('grade', '—')}**)"
    )

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
        ]:
            p = Path(paths[key])
            if p.exists():
                st.download_button(label, data=p.read_bytes(), file_name=p.name, mime=mime)

        docx_p = paths.get("docx")
        if docx_p and Path(docx_p).exists():
            st.download_button(
                "Course summary (DOCX)",
                data=Path(docx_p).read_bytes(),
                file_name=Path(docx_p).name,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

        zip_p = paths.get("delivery_zip")
        if zip_p and Path(zip_p).exists():
            st.download_button(
                "All outputs (ZIP)",
                data=Path(zip_p).read_bytes(),
                file_name=Path(zip_p).name,
                mime="application/zip",
            )

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

