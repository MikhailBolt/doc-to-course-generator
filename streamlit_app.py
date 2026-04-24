import os
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import List, Optional

import streamlit as st

from course_generator.cli import ensure_directories
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
from course_generator.pipeline import run_pipeline


def _save_uploads(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile], target_dir: Path) -> None:
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
    )


def main() -> None:
    st.set_page_config(page_title="Doc-to-Course Generator", layout="wide")

    st.title("Doc-to-Course Generator")
    st.caption("Generate an HTML training course + quizzes from PDF/TXT/MD using local Ollama + FAISS RAG.")

    with st.sidebar:
        st.header("Inputs")
        source_mode = st.radio("Source", ["Upload files", "Use local docs folder"], index=0)

        uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile] = []
        docs_path: Optional[str] = None

        if source_mode == "Upload files":
            uploaded_files = st.file_uploader(
                "Upload PDF / TXT / MD",
                type=[ext.lstrip(".") for ext in sorted(SUPPORTED_EXTENSIONS)],
                accept_multiple_files=True,
            )
        else:
            docs_path = st.text_input("Docs path", value=os.getenv("DOCS_PATH", DEFAULT_DOCS_PATH))

        st.header("Generation")
        language = st.selectbox("Language", ["en", "ru"], index=0 if DEFAULT_LANGUAGE == "en" else 1)
        difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"], index=1)
        model = st.text_input("Ollama model", value=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL))
        embedding_model = st.text_input("Embedding model", value=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
        retrieval_type = st.selectbox("Retrieval", ["similarity", "mmr"], index=0 if DEFAULT_RETRIEVAL_TYPE == "similarity" else 1)

        chunk_size = st.number_input("Chunk size", min_value=200, max_value=4000, value=DEFAULT_CHUNK_SIZE, step=50)
        chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=2000, value=DEFAULT_CHUNK_OVERLAP, step=50)
        top_k = st.number_input("Top-k chunks per lesson", min_value=1, max_value=20, value=DEFAULT_TOP_K, step=1)
        max_preview_chars_per_file = st.number_input(
            "Max preview chars per file",
            min_value=500,
            max_value=50000,
            value=DEFAULT_MAX_PREVIEW_CHARS_PER_FILE,
            step=500,
        )

        st.header("Course shape")
        min_lessons = st.number_input("Min lessons", min_value=1, max_value=30, value=DEFAULT_MIN_LESSONS, step=1)
        max_lessons = st.number_input("Max lessons", min_value=1, max_value=50, value=DEFAULT_MAX_LESSONS, step=1)

        st.header("Quizzes")
        quiz_questions = st.number_input("Final quiz questions", min_value=0, max_value=50, value=DEFAULT_QUIZ_QUESTIONS, step=1)
        pretest_questions = st.number_input("Pre-test questions", min_value=0, max_value=50, value=DEFAULT_PRETEST_QUESTIONS, step=1)
        skip_pretest = st.checkbox("Skip pre-test", value=False)
        skip_final_quiz = st.checkbox("Skip final quiz", value=False)

        st.header("Quality / diagnostics")
        include_source_excerpts = st.checkbox("Include source excerpts", value=False)
        disable_review_pass = st.checkbox("Disable review pass", value=False)
        rebuild = st.checkbox("Force rebuild FAISS index", value=False)

        st.header("Outline grounding (RAG)")
        skip_outline_rag = st.checkbox("Skip outline RAG", value=False)
        outline_rag_max_chunks = st.number_input("Outline RAG max chunks", min_value=0, max_value=200, value=DEFAULT_OUTLINE_RAG_MAX_CHUNKS, step=1)
        outline_rag_max_chars = st.number_input("Outline RAG max chars", min_value=1000, max_value=100000, value=DEFAULT_OUTLINE_RAG_MAX_CHARS, step=1000)

        output_prefix = st.text_input("Output prefix", value=os.getenv("OUTPUT_PREFIX", DEFAULT_OUTPUT_PREFIX))

        run_btn = st.button("Generate", type="primary", use_container_width=True)

    if not run_btn:
        st.info("Pick a source (upload or docs folder), then click **Generate**.")
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
    )

    ensure_directories(args.docs_path, args.db, args.manifest_file, args.output_dir, args.log_dir)

    with st.status("Generating…", expanded=True) as status:
        status.write("Running pipeline (this may take a while depending on model + docs).")
        try:
            result = run_pipeline(args)
        except Exception as exc:
            status.update(label="Failed", state="error")
            st.exception(exc)
            return
        status.update(label="Done", state="complete")

    paths = result["paths"]
    st.success(f"Done in {result['elapsed_seconds']}s. Outline RAG used: {result['outline_rag_used']}")

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
        ]:
            p = Path(paths[key])
            if p.exists():
                st.download_button(label, data=p.read_bytes(), file_name=p.name, mime=mime)

    with col_b:
        st.subheader("Preview")
        st.caption("Rendered `course.html` (may take a second to load).")
        st.components.v1.html(result["course_html"], height=800, scrolling=True)


if __name__ == "__main__":
    main()

