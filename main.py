import os
import re
import sys
import json
import html
import time
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM


load_dotenv()


# =========================
# Default settings
# =========================
DEFAULT_DOCS_PATH = "docs"
DEFAULT_DB_FAISS_PATH = "vectorstore/db_faiss"
DEFAULT_MANIFEST_FILE = "vectorstore/index_manifest.json"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_LOG_DIR = "logs"

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "llama3"

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 6
DEFAULT_QUIZ_QUESTIONS = 10


# =========================
# Utils
# =========================
def ensure_directories(
    docs_path: str,
    db_path: str,
    manifest_file: str,
    output_dir: str,
    log_dir: str,
) -> None:
    docs = Path(docs_path)
    if docs.suffix.lower() == ".pdf":
        docs.parent.mkdir(parents=True, exist_ok=True)
    else:
        docs.mkdir(parents=True, exist_ok=True)

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate HTML course and quiz from PDF documents using local LLM + RAG"
    )

    parser.add_argument(
        "--docs-path",
        default=os.getenv("DOCS_PATH", DEFAULT_DOCS_PATH),
        help="Path to a PDF file or folder with PDF files",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("DB_FAISS_PATH", DEFAULT_DB_FAISS_PATH),
        help="Path to FAISS vector store",
    )
    parser.add_argument(
        "--manifest-file",
        default=os.getenv("MANIFEST_FILE", DEFAULT_MANIFEST_FILE),
        help="Path to vector index manifest JSON",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
        help="Output directory for generated files",
    )
    parser.add_argument(
        "--log-dir",
        default=os.getenv("LOG_DIR", DEFAULT_LOG_DIR),
        help="Directory for logs",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        help="Embedding model name",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
        help="Ollama model name",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.getenv("CHUNK_SIZE", DEFAULT_CHUNK_SIZE)),
        help="Chunk size for text splitting",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=int(os.getenv("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)),
        help="Chunk overlap for text splitting",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(os.getenv("TOP_K", DEFAULT_TOP_K)),
        help="How many chunks to retrieve per lesson",
    )
    parser.add_argument(
        "--quiz-questions",
        type=int,
        default=int(os.getenv("QUIZ_QUESTIONS", DEFAULT_QUIZ_QUESTIONS)),
        help="Number of quiz questions to generate",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default=os.getenv("DIFFICULTY", "medium"),
        help="Quiz/course difficulty level",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild of vector index",
    )

    return parser.parse_args()


def log_message(log_dir: str, message: str) -> None:
    log_file = Path(log_dir) / f"run_{datetime.now().strftime('%Y-%m-%d')}.log"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")


def clean_text(text: str) -> str:
    text = re.sub(r"\r", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def safe_filename(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    return name.strip("_") or "output"


def normalize_json_block(text: str) -> str:
    text = text.strip()

    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return text


def extract_json_from_text(text: str) -> Any:
    text = normalize_json_block(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise


# =========================
# PDF collection + manifest
# =========================
def collect_pdf_files(docs_path: str) -> List[Path]:
    path = Path(docs_path)

    if not path.exists():
        print(f"(X) Error: '{docs_path}' does not exist.")
        sys.exit(1)

    if path.is_file():
        if path.suffix.lower() != ".pdf":
            print(f"(X) Error: '{docs_path}' is not a PDF.")
            sys.exit(1)
        return [path]

    pdf_files = sorted([p for p in path.glob("*.pdf") if p.is_file()])
    if not pdf_files:
        print(f"(X) Error: No PDF files found in '{docs_path}'.")
        sys.exit(1)

    return pdf_files


def file_fingerprint(file_path: Path) -> str:
    stat = file_path.stat()
    raw = f"{file_path.resolve()}|{stat.st_size}|{stat.st_mtime}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_manifest_data(pdf_files: List[Path]) -> Dict[str, Any]:
    return {
        "files": [
            {
                "name": pdf.name,
                "path": str(pdf.resolve()),
                "fingerprint": file_fingerprint(pdf),
            }
            for pdf in pdf_files
        ]
    }


def load_manifest(manifest_file: str) -> Dict[str, Any]:
    path = Path(manifest_file)
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_manifest(manifest_file: str, data: Dict[str, Any]) -> None:
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def is_index_stale(pdf_files: List[Path], db_path: str, manifest_file: str) -> bool:
    db_dir = Path(db_path)
    index_file = db_dir / "index.faiss"
    meta_file = db_dir / "index.pkl"

    if not db_dir.exists():
        return True
    if not index_file.exists() or not meta_file.exists():
        return True

    current_manifest = build_manifest_data(pdf_files)
    saved_manifest = load_manifest(manifest_file)
    return current_manifest != saved_manifest


# =========================
# Vector store
# =========================
def build_vectorstore(
    pdf_files: List[Path],
    db_path: str,
    manifest_file: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Tuple[FAISS, List[Dict[str, Any]]]:
    print("--- Processing PDF files and building vector DB... ---")

    all_documents = []
    docs_info = []

    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()

        if not documents:
            print(f"(!) Skipping '{pdf_path.name}': no text extracted.")
            continue

        docs_info.append(
            {
                "name": pdf_path.name,
                "path": str(pdf_path.resolve()),
                "pages": len(documents),
            }
        )

        for doc in documents:
            doc.metadata["document_name"] = pdf_path.name
            doc.metadata["document_path"] = str(pdf_path.resolve())

        all_documents.extend(documents)

    if not all_documents:
        raise ValueError("No text could be extracted from provided PDF files.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splits = splitter.split_documents(all_documents)

    if not splits:
        raise ValueError("No chunks were created from the PDF files.")

    for idx, split in enumerate(splits):
        split.metadata["chunk_id"] = idx

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local(db_path)
    save_manifest(manifest_file, build_manifest_data(pdf_files))

    return vectorstore, docs_info


def load_or_create_vectorstore(args: argparse.Namespace, pdf_files: List[Path]) -> Tuple[FAISS, List[Dict[str, Any]]]:
    should_rebuild = args.rebuild or is_index_stale(pdf_files, args.db, args.manifest_file)
    docs_info = []

    for pdf in pdf_files:
        try:
            page_count = len(PyPDFLoader(str(pdf)).load())
        except Exception:
            page_count = 0

        docs_info.append(
            {
                "name": pdf.name,
                "path": str(pdf.resolve()),
                "pages": page_count,
            }
        )

    if should_rebuild:
        if args.rebuild:
            print("(!) Force rebuild requested.")
        else:
            print("(!) Document set changed or index missing. Rebuilding vector DB...")

        return build_vectorstore(
            pdf_files=pdf_files,
            db_path=args.db,
            manifest_file=args.manifest_file,
            embedding_model=args.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

    print("--- Loading existing vector DB... ---")
    embeddings = HuggingFaceEmbeddings(model_name=args.embedding_model)

    try:
        vectorstore = FAISS.load_local(
            args.db,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return vectorstore, docs_info
    except Exception:
        print("(!) Existing vector DB is corrupted or incompatible. Rebuilding...")
        return build_vectorstore(
            pdf_files=pdf_files,
            db_path=args.db,
            manifest_file=args.manifest_file,
            embedding_model=args.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )


# =========================
# LLM helpers
# =========================
def call_llm(llm: OllamaLLM, prompt: str) -> str:
    response = llm.invoke(prompt)
    if isinstance(response, str):
        return response.strip()
    return str(response).strip()


def get_combined_preview_text(pdf_files: List[Path], max_chars_per_file: int = 6000) -> str:
    parts = []

    for pdf_path in pdf_files:
        try:
            pages = PyPDFLoader(str(pdf_path)).load()
            joined = "\n".join(page.page_content for page in pages)
            joined = clean_text(joined)
            joined = joined[:max_chars_per_file]

            parts.append(
                f"\n===== DOCUMENT: {pdf_path.name} =====\n"
                f"{joined}\n"
            )
        except Exception as e:
            parts.append(f"\n===== DOCUMENT: {pdf_path.name} =====\nFailed to read document: {e}\n")

    return "\n".join(parts)


def generate_course_outline(
    llm: OllamaLLM,
    preview_text: str,
    difficulty: str,
) -> Dict[str, Any]:
    prompt = f"""
You are an instructional designer.

Based ONLY on the document content below, generate a structured training course outline.

Requirements:
- The course must be practical and clear.
- Difficulty level: {difficulty}.
- Return ONLY valid JSON.
- The JSON schema must be:

{{
  "course_title": "string",
  "course_description": "string",
  "target_audience": "string",
  "learning_outcomes": ["string", "string"],
  "lessons": [
    {{
      "title": "string",
      "goal": "string",
      "key_points": ["string", "string", "string"]
    }}
  ]
}}

Rules:
- Create 4 to 7 lessons.
- Each lesson should have 3 to 5 key points.
- Do not invent topics that are not supported by the documents.
- Be specific.

DOCUMENT CONTENT:
{preview_text}
"""
    raw = call_llm(llm, prompt)
    data = extract_json_from_text(raw)

    if not isinstance(data, dict):
        raise ValueError("Course outline LLM output is not a JSON object.")

    lessons = data.get("lessons", [])
    if not isinstance(lessons, list) or not lessons:
        raise ValueError("Course outline contains no lessons.")

    return data


def retrieve_lesson_context(vectorstore: FAISS, lesson_title: str, key_points: List[str], top_k: int) -> List[Any]:
    query = lesson_title
    if key_points:
        query += "\n" + "\n".join(key_points)

    docs = vectorstore.similarity_search(query, k=top_k)
    return docs


def generate_lesson_html_section(
    llm: OllamaLLM,
    lesson_title: str,
    lesson_goal: str,
    key_points: List[str],
    retrieved_docs: List[Any],
) -> Dict[str, Any]:
    context_blocks = []

    sources = []
    for doc in retrieved_docs:
        page = doc.metadata.get("page")
        page_num = page + 1 if isinstance(page, int) else "N/A"
        doc_name = doc.metadata.get("document_name", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "N/A")

        sources.append(
            {
                "document_name": doc_name,
                "page": page_num,
                "chunk_id": chunk_id,
            }
        )

        context_blocks.append(
            f"[Document: {doc_name} | Page: {page_num} | Chunk: {chunk_id}]\n{doc.page_content}"
        )

    context_text = "\n\n".join(context_blocks)

    prompt = f"""
You are creating a lesson section for an HTML training course.

Write the content for this lesson using ONLY the provided context.

Lesson title: {lesson_title}
Lesson goal: {lesson_goal}
Key points:
{json.dumps(key_points, ensure_ascii=False, indent=2)}

Return ONLY valid JSON with this schema:
{{
  "lesson_html": "<section>...</section>",
  "summary": "string"
}}

Requirements for lesson_html:
- Use semantic HTML.
- Start with <section class="lesson-section"> and end with </section>.
- Include:
  - <h2> lesson title
  - <p> lesson goal/introduction
  - a short explanatory body
  - <ul> with key takeaways
  - a small "In practice" subsection
- Do not include full HTML page, only section HTML.
- Do not use markdown.
- Keep it readable and concise.
- Do not invent facts beyond the context.

CONTEXT:
{context_text}
"""
    raw = call_llm(llm, prompt)
    data = extract_json_from_text(raw)

    if not isinstance(data, dict):
        raise ValueError(f"Lesson generation failed for lesson: {lesson_title}")

    lesson_html = data.get("lesson_html", "").strip()
    summary = data.get("summary", "").strip()

    if not lesson_html:
        raise ValueError(f"Empty lesson_html for lesson: {lesson_title}")

    return {
        "lesson_html": lesson_html,
        "summary": summary,
        "sources": sources,
    }


def generate_quiz(
    llm: OllamaLLM,
    outline: Dict[str, Any],
    lesson_payloads: List[Dict[str, Any]],
    difficulty: str,
    quiz_questions: int,
) -> List[Dict[str, Any]]:
    lesson_summaries = []
    for idx, lesson in enumerate(outline.get("lessons", []), start=1):
        summary = lesson_payloads[idx - 1].get("summary", "")
        lesson_summaries.append(
            {
                "lesson_number": idx,
                "title": lesson.get("title", ""),
                "goal": lesson.get("goal", ""),
                "key_points": lesson.get("key_points", []),
                "summary": summary,
            }
        )

    prompt = f"""
You are generating a multiple-choice quiz for a training course.

Difficulty: {difficulty}
Generate exactly {quiz_questions} questions.

Return ONLY valid JSON as an array.
Each item must follow this schema:

[
  {{
    "question": "string",
    "options": ["A", "B", "C", "D"],
    "correct_answer": "one of the options exactly",
    "explanation": "string"
  }}
]

Rules:
- Use ONLY the information from the lesson summaries below.
- Each question must have exactly 4 options.
- Only 1 option must be correct.
- Options must be plausible.
- Explanations should be concise.
- Avoid duplicate questions.

LESSON SUMMARIES:
{json.dumps(lesson_summaries, ensure_ascii=False, indent=2)}
"""
    raw = call_llm(llm, prompt)
    data = extract_json_from_text(raw)

    if not isinstance(data, list):
        raise ValueError("Quiz output is not a JSON array.")

    cleaned_questions = []
    for item in data:
        if not isinstance(item, dict):
            continue

        question = str(item.get("question", "")).strip()
        options = item.get("options", [])
        correct_answer = str(item.get("correct_answer", "")).strip()
        explanation = str(item.get("explanation", "")).strip()

        if not question or not isinstance(options, list) or len(options) != 4:
            continue
        if correct_answer not in options:
            continue

        cleaned_questions.append(
            {
                "question": question,
                "options": options,
                "correct_answer": correct_answer,
                "explanation": explanation,
            }
        )

    if not cleaned_questions:
        raise ValueError("No valid quiz questions were generated.")

    return cleaned_questions


# =========================
# HTML builder
# =========================
def build_course_html(
    outline: Dict[str, Any],
    lesson_payloads: List[Dict[str, Any]],
    docs_info: List[Dict[str, Any]],
) -> str:
    course_title = html.escape(outline.get("course_title", "Generated Course"))
    course_description = html.escape(outline.get("course_description", ""))
    target_audience = html.escape(outline.get("target_audience", ""))

    learning_outcomes = outline.get("learning_outcomes", [])
    if not isinstance(learning_outcomes, list):
        learning_outcomes = []

    overview_items = "\n".join(
        f"<li>{html.escape(str(item))}</li>" for item in learning_outcomes
    )

    docs_items = "\n".join(
        f"<li><strong>{html.escape(doc['name'])}</strong> — {doc['pages']} pages</li>"
        for doc in docs_info
    )

    toc_items = []
    for i, lesson in enumerate(outline.get("lessons", []), start=1):
        title = html.escape(lesson.get("title", f"Lesson {i}"))
        toc_items.append(f'<li><a href="#lesson-{i}">{title}</a></li>')
    toc_html = "\n".join(toc_items)

    lesson_sections = []
    for i, payload in enumerate(lesson_payloads, start=1):
        raw_section = payload["lesson_html"]

        section_with_anchor = raw_section.replace(
            '<section class="lesson-section">',
            f'<section class="lesson-section" id="lesson-{i}">',
            1
        )

        sources = payload.get("sources", [])
        source_items = "\n".join(
            f"<li>{html.escape(str(src['document_name']))} — page {html.escape(str(src['page']))}, chunk {html.escape(str(src['chunk_id']))}</li>"
            for src in sources
        )

        if source_items:
            section_with_anchor += f"""
<div class="lesson-sources">
  <h3>Sources used</h3>
  <ul>
    {source_items}
  </ul>
</div>
"""
        lesson_sections.append(section_with_anchor)

    lesson_sections_html = "\n\n".join(lesson_sections)

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{course_title}</title>
  <style>
    body {{
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.6;
      color: #1f2937;
      background: #f8fafc;
      margin: 0;
      padding: 0;
    }}
    .container {{
      max-width: 980px;
      margin: 0 auto;
      padding: 32px 20px 64px;
    }}
    .hero {{
      background: white;
      border-radius: 16px;
      padding: 28px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.06);
      margin-bottom: 24px;
    }}
    .meta, .toc, .lesson-section, .lesson-sources {{
      background: white;
      border-radius: 16px;
      padding: 24px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.05);
      margin-bottom: 20px;
    }}
    h1, h2, h3 {{
      color: #0f172a;
    }}
    h1 {{
      margin-top: 0;
      font-size: 2rem;
    }}
    ul {{
      padding-left: 20px;
    }}
    .muted {{
      color: #64748b;
    }}
    a {{
      color: #2563eb;
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    .badge {{
      display: inline-block;
      background: #dbeafe;
      color: #1d4ed8;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 0.9rem;
      margin-right: 8px;
    }}
    .footer {{
      margin-top: 28px;
      color: #64748b;
      font-size: 0.95rem;
      text-align: center;
    }}
    code {{
      background: #f1f5f9;
      padding: 2px 6px;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="hero">
      <span class="badge">AI-generated course</span>
      <span class="badge">HTML</span>
      <h1>{course_title}</h1>
      <p>{course_description}</p>
      <p><strong>Target audience:</strong> {target_audience}</p>
    </div>

    <div class="meta">
      <h2>Learning outcomes</h2>
      <ul>
        {overview_items}
      </ul>
    </div>

    <div class="meta">
      <h2>Documents used</h2>
      <ul>
        {docs_items}
      </ul>
    </div>

    <div class="toc">
      <h2>Course contents</h2>
      <ol>
        {toc_html}
      </ol>
    </div>

    {lesson_sections_html}

    <div class="footer">
      Generated on {generated_at}
    </div>
  </div>
</body>
</html>
"""


# =========================
# Save outputs
# =========================
def save_course_html(output_dir: str, html_text: str) -> str:
    path = Path(output_dir) / "course.html"
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_text)
    return str(path)


def save_quiz_json(output_dir: str, quiz_data: List[Dict[str, Any]]) -> str:
    path = Path(output_dir) / "quiz.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(quiz_data, f, ensure_ascii=False, indent=2)
    return str(path)


def save_course_metadata(
    output_dir: str,
    outline: Dict[str, Any],
    docs_info: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    path = Path(output_dir) / "course_metadata.json"
    payload = {
        "generated_at": datetime.now().isoformat(),
        "model": args.model,
        "embedding_model": args.embedding_model,
        "difficulty": args.difficulty,
        "top_k": args.top_k,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "documents": docs_info,
        "outline": outline,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return str(path)


# =========================
# Main
# =========================
def main() -> None:
    args = parse_args()

    ensure_directories(
        docs_path=args.docs_path,
        db_path=args.db,
        manifest_file=args.manifest_file,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
    )

    started = time.time()
    log_message(args.log_dir, "Starting course and quiz generation pipeline")

    pdf_files = collect_pdf_files(args.docs_path)
    print(f"Found {len(pdf_files)} PDF file(s).")

    try:
        vectorstore, docs_info = load_or_create_vectorstore(args, pdf_files)
        log_message(args.log_dir, f"Vector store ready. Documents loaded: {len(docs_info)}")
    except Exception as e:
        print(f"(X) Failed to prepare vector store: {e}")
        log_message(args.log_dir, f"Vector store error: {e}")
        sys.exit(1)

    try:
        llm = OllamaLLM(model=args.model)
        log_message(args.log_dir, f"Initialized Ollama model: {args.model}")
    except Exception as e:
        print(f"(X) Failed to initialize LLM: {e}")
        log_message(args.log_dir, f"LLM init error: {e}")
        sys.exit(1)

    try:
        print("--- Generating course outline... ---")
        preview_text = get_combined_preview_text(pdf_files)
        outline = generate_course_outline(
            llm=llm,
            preview_text=preview_text,
            difficulty=args.difficulty,
        )
        log_message(args.log_dir, "Course outline generated successfully")
    except Exception as e:
        print(f"(X) Failed to generate course outline: {e}")
        log_message(args.log_dir, f"Course outline error: {e}")
        sys.exit(1)

    lesson_payloads = []
    lessons = outline.get("lessons", [])

    try:
        for idx, lesson in enumerate(lessons, start=1):
            lesson_title = str(lesson.get("title", f"Lesson {idx}")).strip()
            lesson_goal = str(lesson.get("goal", "")).strip()
            key_points = lesson.get("key_points", [])
            if not isinstance(key_points, list):
                key_points = []

            print(f"--- Generating lesson {idx}/{len(lessons)}: {lesson_title} ---")
            log_message(args.log_dir, f"Generating lesson: {lesson_title}")

            retrieved_docs = retrieve_lesson_context(
                vectorstore=vectorstore,
                lesson_title=lesson_title,
                key_points=key_points,
                top_k=args.top_k,
            )

            lesson_payload = generate_lesson_html_section(
                llm=llm,
                lesson_title=lesson_title,
                lesson_goal=lesson_goal,
                key_points=key_points,
                retrieved_docs=retrieved_docs,
            )
            lesson_payloads.append(lesson_payload)

        log_message(args.log_dir, "All lesson sections generated successfully")
    except Exception as e:
        print(f"(X) Failed during lesson generation: {e}")
        log_message(args.log_dir, f"Lesson generation error: {e}")
        sys.exit(1)

    try:
        print("--- Generating quiz... ---")
        quiz_data = generate_quiz(
            llm=llm,
            outline=outline,
            lesson_payloads=lesson_payloads,
            difficulty=args.difficulty,
            quiz_questions=args.quiz_questions,
        )
        log_message(args.log_dir, f"Quiz generated successfully with {len(quiz_data)} questions")
    except Exception as e:
        print(f"(X) Failed to generate quiz: {e}")
        log_message(args.log_dir, f"Quiz generation error: {e}")
        sys.exit(1)

    try:
        print("--- Building final HTML course... ---")
        course_html = build_course_html(
            outline=outline,
            lesson_payloads=lesson_payloads,
            docs_info=docs_info,
        )

        course_path = save_course_html(args.output_dir, course_html)
        quiz_path = save_quiz_json(args.output_dir, quiz_data)
        metadata_path = save_course_metadata(args.output_dir, outline, docs_info, args)

        elapsed = time.time() - started

        print("\n[SUCCESS] Generation complete!")
        print(f"Course HTML: {course_path}")
        print(f"Quiz JSON:   {quiz_path}")
        print(f"Metadata:    {metadata_path}")
        print(f"Time:        {elapsed:.2f}s")

        log_message(
            args.log_dir,
            f"Generation complete. course={course_path}, quiz={quiz_path}, metadata={metadata_path}, elapsed={elapsed:.2f}s",
        )
    except Exception as e:
        print(f"(X) Failed to save outputs: {e}")
        log_message(args.log_dir, f"Save outputs error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()