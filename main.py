import argparse
import html
import json
import os
import re
import sys
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

# =========================
# Defaults
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
DEFAULT_PRETEST_QUESTIONS = 5
DEFAULT_RETRIEVAL_TYPE = "similarity"
DEFAULT_LANGUAGE = "en"
DEFAULT_MAX_PREVIEW_CHARS_PER_FILE = 6000
DEFAULT_OUTPUT_PREFIX = ""
DEFAULT_MAX_LESSONS = 7
DEFAULT_MIN_LESSONS = 4

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


# =========================
# Utils
# =========================
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


def sanitize_lesson_html(section_html: str) -> str:
    section_html = re.sub(r"<script.*?>.*?</script>", "", section_html, flags=re.IGNORECASE | re.DOTALL)
    section_html = re.sub(r"<style.*?>.*?</style>", "", section_html, flags=re.IGNORECASE | re.DOTALL)
    section_html = re.sub(r"\son\w+\s*=\s*(['\"]).*?\1", "", section_html, flags=re.IGNORECASE | re.DOTALL)
    return section_html.strip()


def deduplicate_questions(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique = []
    seen = set()
    for item in questions:
        question = str(item.get("question", "")).strip().lower()
        question = re.sub(r"\s+", " ", question)
        options = tuple(str(x).strip().lower() for x in item.get("options", []))
        key = (question, options)
        if not question or key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def deduplicate_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for src in sources:
        key = (
            str(src.get("document_name", "")),
            str(src.get("page", "")),
            str(src.get("chunk_id", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(src)
    return unique


def estimate_duration_minutes(outline: Dict[str, Any]) -> int:
    lessons = outline.get("lessons", [])
    lesson_count = len(lessons) if isinstance(lessons, list) else 0
    return max(20, lesson_count * 12 + 10)


def retry_llm_json(llm: OllamaLLM, prompt: str, max_attempts: int = 3) -> Any:
    last_error = None
    current_prompt = prompt
    for _ in range(max_attempts):
        raw = call_llm(llm, current_prompt)
        try:
            return extract_json_from_text(raw)
        except Exception as exc:
            last_error = exc
            current_prompt = "Return ONLY valid JSON with no markdown fences.\n\n" + prompt
    raise ValueError(f"Failed to get valid JSON after {max_attempts} attempts: {last_error}")


def validate_outline(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Outline is not a JSON object.")

    required_str_fields = ["course_title", "course_description", "target_audience"]
    for field in required_str_fields:
        if not isinstance(data.get(field), str) or not data.get(field).strip():
            raise ValueError(f"Outline field '{field}' is missing or invalid.")

    for list_field in ["prerequisites", "learning_outcomes", "glossary", "lessons"]:
        if not isinstance(data.get(list_field), list):
            raise ValueError(f"Outline field '{list_field}' must be a list.")

    if not data["lessons"]:
        raise ValueError("Outline contains no lessons.")

    normalized_lessons = []
    for lesson in data["lessons"]:
        if not isinstance(lesson, dict):
            continue
        title = str(lesson.get("title", "")).strip()
        goal = str(lesson.get("goal", "")).strip()
        key_points = lesson.get("key_points", [])
        if not title or not goal or not isinstance(key_points, list):
            continue
        cleaned_points = [str(x).strip() for x in key_points if str(x).strip()]
        if not cleaned_points:
            continue
        normalized_lessons.append({
            "title": title,
            "goal": goal,
            "key_points": cleaned_points[:5],
        })

    if not normalized_lessons:
        raise ValueError("No valid lessons found in outline.")

    normalized_glossary = []
    for item in data.get("glossary", []):
        if not isinstance(item, dict):
            continue
        term = str(item.get("term", "")).strip()
        definition = str(item.get("definition", "")).strip()
        if term and definition:
            normalized_glossary.append({"term": term, "definition": definition})

    data["lessons"] = normalized_lessons
    data["glossary"] = normalized_glossary[:8]
    data["prerequisites"] = [str(x).strip() for x in data.get("prerequisites", []) if str(x).strip()]
    data["learning_outcomes"] = [str(x).strip() for x in data.get("learning_outcomes", []) if str(x).strip()]
    return data


def fallback_lesson_html(lesson_title: str, lesson_goal: str, key_points: List[str], retrieved_docs: List[Any]) -> Dict[str, Any]:
    paragraphs = []
    for doc in retrieved_docs[:2]:
        text = clean_text(doc.page_content)
        if text:
            paragraphs.append(html.escape(text[:700]))

    body_html = "".join(f"<p>{p}</p>" for p in paragraphs[:2])
    takeaway_html = "".join(f"<li>{html.escape(point)}</li>" for point in key_points[:5])

    lesson_html = f"""
<section class="lesson-section">
  <h2>{html.escape(lesson_title)}</h2>
  <p>{html.escape(lesson_goal)}</p>
  {body_html}
  <h3>Key takeaways</h3>
  <ul>
    {takeaway_html}
  </ul>
  <h3>In practice</h3>
  <p>Review the source material and connect the main ideas from this lesson to a real task or workflow.</p>
</section>
""".strip()

    return {
        "lesson_html": lesson_html,
        "summary": lesson_goal,
        "key_takeaways": key_points[:5],
    }


def validate_question_item(item: Dict[str, Any], allow_true_false: bool) -> Dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    question = str(item.get("question", "")).strip()
    q_type = str(item.get("type", "single_choice")).strip()
    options = item.get("options", [])
    correct_answer = str(item.get("correct_answer", "")).strip()
    explanation = str(item.get("explanation", "")).strip()

    if not question or not isinstance(options, list):
        return None

    if allow_true_false and q_type == "true_false":
        if options != ["True", "False"]:
            return None
        if correct_answer not in options:
            return None
        return {
            "question": question,
            "type": "true_false",
            "options": options,
            "correct_answer": correct_answer,
            "explanation": explanation,
        }

    if len(options) != 4 or correct_answer not in options:
        return None

    return {
        "question": question,
        "type": "single_choice",
        "options": [str(x).strip() for x in options],
        "correct_answer": correct_answer,
        "explanation": explanation,
    }


def sanitize_filename_part(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def build_output_path(output_dir: str, filename: str, output_prefix: str = "") -> Path:
    if output_prefix:
        prefix = sanitize_filename_part(output_prefix)
        filename = f"{prefix}_{filename}"
    return Path(output_dir) / filename


# =========================
# Document collection + manifest
# =========================
def collect_source_files(docs_path: str) -> List[Path]:
    path = Path(docs_path)

    if not path.exists():
        print(f"(X) Error: '{docs_path}' does not exist.")
        sys.exit(1)

    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"(X) Error: '{docs_path}' is not a supported file type.")
            sys.exit(1)
        return [path]

    source_files = sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS])
    if not source_files:
        print(f"(X) Error: No supported files found in '{docs_path}'.")
        sys.exit(1)

    return source_files


def file_fingerprint(file_path: Path) -> str:
    stat = file_path.stat()
    raw = f"{file_path.resolve()}|{stat.st_size}|{stat.st_mtime}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_manifest_data(source_files: List[Path]) -> Dict[str, Any]:
    return {
        "files": [
            {
                "name": f.name,
                "path": str(f.resolve()),
                "fingerprint": file_fingerprint(f),
            }
            for f in source_files
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


def is_index_stale(source_files: List[Path], db_path: str, manifest_file: str) -> bool:
    db_dir = Path(db_path)
    index_file = db_dir / "index.faiss"
    meta_file = db_dir / "index.pkl"

    if not db_dir.exists() or not index_file.exists() or not meta_file.exists():
        return True

    current_manifest = build_manifest_data(source_files)
    saved_manifest = load_manifest(manifest_file)
    return current_manifest != saved_manifest


def load_file_documents(file_path: Path) -> List[Any]:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
    elif suffix in {".txt", ".md"}:
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    for doc in docs:
        doc.metadata["document_name"] = file_path.name
        doc.metadata["document_path"] = str(file_path.resolve())
        doc.metadata["document_type"] = suffix.lstrip(".")
    return docs


# =========================
# Vector store
# =========================
def build_vectorstore(
    source_files: List[Path],
    db_path: str,
    manifest_file: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Tuple[FAISS, List[Dict[str, Any]]]:
    print("--- Processing source files and building vector DB... ---")
    all_documents = []
    docs_info = []

    for file_path in source_files:
        documents = load_file_documents(file_path)
        if not documents:
            print(f"(!) Skipping '{file_path.name}': no text extracted.")
            continue

        docs_info.append({
            "name": file_path.name,
            "path": str(file_path.resolve()),
            "pages": len(documents),
            "type": file_path.suffix.lower().lstrip("."),
        })
        all_documents.extend(documents)

    if not all_documents:
        raise ValueError("No text could be extracted from provided files.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splits = splitter.split_documents(all_documents)

    if not splits:
        raise ValueError("No chunks were created from source files.")

    for idx, split in enumerate(splits):
        split.metadata["chunk_id"] = idx

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local(db_path)
    save_manifest(manifest_file, build_manifest_data(source_files))
    return vectorstore, docs_info


def load_or_create_vectorstore(args: argparse.Namespace, source_files: List[Path]) -> Tuple[FAISS, List[Dict[str, Any]]]:
    should_rebuild = args.rebuild or is_index_stale(source_files, args.db, args.manifest_file)
    docs_info = []

    for f in source_files:
        try:
            docs_count = len(load_file_documents(f))
        except Exception:
            docs_count = 0
        docs_info.append({
            "name": f.name,
            "path": str(f.resolve()),
            "pages": docs_count,
            "type": f.suffix.lower().lstrip("."),
        })

    if should_rebuild:
        if args.rebuild:
            print("(!) Force rebuild requested.")
        else:
            print("(!) Source file set changed or index missing. Rebuilding vector DB...")
        return build_vectorstore(
            source_files=source_files,
            db_path=args.db,
            manifest_file=args.manifest_file,
            embedding_model=args.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

    print("--- Loading existing vector DB... ---")
    embeddings = HuggingFaceEmbeddings(model_name=args.embedding_model)
    try:
        vectorstore = FAISS.load_local(args.db, embeddings, allow_dangerous_deserialization=True)
        return vectorstore, docs_info
    except Exception:
        print("(!) Existing vector DB is corrupted or incompatible. Rebuilding...")
        return build_vectorstore(
            source_files=source_files,
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
    return response.strip() if isinstance(response, str) else str(response).strip()


def get_language_instruction(language: str) -> str:
    return "Generate all content in Russian." if language == "ru" else "Generate all content in English."


def get_combined_preview_text(source_files: List[Path], max_chars_per_file: int = 6000) -> str:
    parts = []
    for file_path in source_files:
        try:
            docs = load_file_documents(file_path)
            joined = "\n".join(doc.page_content for doc in docs)
            joined = clean_text(joined)[:max_chars_per_file]
            parts.append(f"\n===== DOCUMENT: {file_path.name} =====\n{joined}\n")
        except Exception as exc:
            parts.append(f"\n===== DOCUMENT: {file_path.name} =====\nFailed to read document: {exc}\n")
    return "\n".join(parts)


def generate_course_outline(llm: OllamaLLM, preview_text: str, difficulty: str, language: str, min_lessons: int, max_lessons: int) -> Dict[str, Any]:
    prompt = f"""
You are an expert instructional designer.

{get_language_instruction(language)}

Based ONLY on the document content below, generate a structured training course outline.

Requirements:
- The course must be practical, clear, and useful for real learning.
- Difficulty level: {difficulty}.
- Return ONLY valid JSON.
- The JSON schema must be:

{{
  "course_title": "string",
  "course_description": "string",
  "target_audience": "string",
  "prerequisites": ["string", "string"],
  "learning_outcomes": ["string", "string"],
  "glossary": [
    {{
      "term": "string",
      "definition": "string"
    }}
  ],
  "lessons": [
    {{
      "title": "string",
      "goal": "string",
      "key_points": ["string", "string", "string"]
    }}
  ]
}}

Rules:
- Create {min_lessons} to {max_lessons} lessons.
- Each lesson should have 3 to 5 key points.
- Create 3 to 8 glossary items.
- Do not invent topics that are not supported by the documents.
- Be specific.
- Keep target_audience short.

DOCUMENT CONTENT:
{preview_text}
"""
    data = retry_llm_json(llm, prompt)
    return validate_outline(data)


def review_outline(llm: OllamaLLM, outline: Dict[str, Any], language: str, min_lessons: int, max_lessons: int) -> Dict[str, Any]:
    prompt = f"""
You are reviewing a generated training course outline.

{get_language_instruction(language)}

Return ONLY valid JSON with exactly the same schema as the input.

Your job:
- improve clarity
- remove redundancy
- make lesson progression more logical
- keep only information grounded in the source-derived outline
- do not add unrelated topics
- preserve {min_lessons} to {max_lessons} lessons

INPUT OUTLINE:
{json.dumps(outline, ensure_ascii=False, indent=2)}
"""
    reviewed = retry_llm_json(llm, prompt)
    return validate_outline(reviewed)


def retrieve_lesson_context(vectorstore: FAISS, lesson_title: str, key_points: List[str], top_k: int, retrieval_type: str) -> List[Any]:
    query = lesson_title
    if key_points:
        query += "\n" + "\n".join(key_points)

    if retrieval_type == "mmr":
        return vectorstore.max_marginal_relevance_search(query, k=top_k, fetch_k=max(top_k * 2, 8))
    return vectorstore.similarity_search(query, k=top_k)


def generate_lesson_html_section(
    llm: OllamaLLM,
    lesson_title: str,
    lesson_goal: str,
    key_points: List[str],
    retrieved_docs: List[Any],
    language: str,
    include_source_excerpts: bool,
) -> Dict[str, Any]:
    context_blocks = []
    sources = []
    source_excerpts = []

    for doc in retrieved_docs:
        page = doc.metadata.get("page")
        page_num = page + 1 if isinstance(page, int) else "N/A"
        doc_name = doc.metadata.get("document_name", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "N/A")
        excerpt = clean_text(doc.page_content)[:300]
        sources.append({"document_name": doc_name, "page": page_num, "chunk_id": chunk_id})
        source_excerpts.append({"document_name": doc_name, "page": page_num, "chunk_id": chunk_id, "excerpt": excerpt})
        context_blocks.append(f"[Document: {doc_name} | Page: {page_num} | Chunk: {chunk_id}]\n{doc.page_content}")

    context_text = "\n\n".join(context_blocks)
    prompt = f"""
You are creating one lesson section for an HTML training course.

{get_language_instruction(language)}

Write the lesson using ONLY the provided context.

Lesson title: {lesson_title}
Lesson goal: {lesson_goal}
Key points:
{json.dumps(key_points, ensure_ascii=False, indent=2)}

Return ONLY valid JSON with this schema:
{{
  "lesson_html": "<section>...</section>",
  "summary": "string",
  "key_takeaways": ["string", "string", "string"]
}}

Requirements for lesson_html:
- Use semantic HTML.
- Start with <section class="lesson-section"> and end with </section>.
- Include:
  - <h2> lesson title
  - <p> lesson goal/introduction
  - 2 to 4 short paragraphs
  - <ul> with key takeaways
  - one subsection titled "In practice"
- Do not include full HTML page, only section HTML.
- Do not use markdown.
- Keep it readable and concise.
- Do not invent facts beyond the context.

CONTEXT:
{context_text}
"""
    try:
        data = retry_llm_json(llm, prompt)
        if not isinstance(data, dict):
            raise ValueError("Lesson JSON is not an object.")
        lesson_html = sanitize_lesson_html(str(data.get("lesson_html", "")).strip())
        summary = str(data.get("summary", "")).strip()
        key_takeaways = data.get("key_takeaways", [])
        if not lesson_html:
            raise ValueError("Empty lesson HTML.")
        if not isinstance(key_takeaways, list):
            key_takeaways = []
        return {
            "lesson_html": lesson_html,
            "summary": summary or lesson_goal,
            "key_takeaways": [str(x).strip() for x in key_takeaways if str(x).strip()][:5],
            "sources": deduplicate_sources(sources),
            "source_excerpts": source_excerpts[: min(3, len(source_excerpts))] if include_source_excerpts else [],
        }
    except Exception:
        fallback = fallback_lesson_html(lesson_title, lesson_goal, key_points, retrieved_docs)
        fallback["sources"] = deduplicate_sources(sources)
        fallback["source_excerpts"] = source_excerpts[: min(3, len(source_excerpts))] if include_source_excerpts else []
        return fallback


def generate_pretest(llm: OllamaLLM, outline: Dict[str, Any], pretest_questions: int, difficulty: str, language: str) -> List[Dict[str, Any]]:
    prompt = f"""
You are generating a short diagnostic pre-test for a course.

{get_language_instruction(language)}

Difficulty: {difficulty}
Generate exactly {pretest_questions} questions.

Return ONLY valid JSON as an array.
Each item must follow this schema:

[
  {{
    "question": "string",
    "type": "single_choice",
    "options": ["A", "B", "C", "D"],
    "correct_answer": "one of the options exactly",
    "explanation": "string"
  }}
]

Rules:
- Use ONLY topics that appear in the course outline below.
- Keep questions broad and diagnostic.
- Each question must have exactly 4 options.
- Only 1 option must be correct.

COURSE OUTLINE:
{json.dumps(outline, ensure_ascii=False, indent=2)}
"""
    data = retry_llm_json(llm, prompt)
    if not isinstance(data, list):
        raise ValueError("Pre-test output is not a JSON array.")
    cleaned = []
    for item in data:
        validated = validate_question_item(item, allow_true_false=False)
        if validated:
            cleaned.append(validated)
    cleaned = deduplicate_questions(cleaned)
    if not cleaned:
        raise ValueError("No valid pre-test questions were generated.")
    return cleaned[:pretest_questions]


def generate_quiz(
    llm: OllamaLLM,
    outline: Dict[str, Any],
    lesson_payloads: List[Dict[str, Any]],
    difficulty: str,
    quiz_questions: int,
    language: str,
) -> List[Dict[str, Any]]:
    lesson_summaries = []
    lesson_titles = []

    for idx, lesson in enumerate(outline.get("lessons", []), start=1):
        lesson_titles.append(lesson.get("title", ""))
        summary = lesson_payloads[idx - 1].get("summary", "")
        lesson_summaries.append(
            {
                "lesson_number": idx,
                "title": lesson.get("title", ""),
                "goal": lesson.get("goal", ""),
                "key_points": lesson.get("key_points", []),
                "summary": summary,
                "key_takeaways": lesson_payloads[idx - 1].get("key_takeaways", []),
            }
        )

    prompt = f"""
You are generating a final quiz for a training course.

{get_language_instruction(language)}

Difficulty: {difficulty}
Generate exactly {quiz_questions} questions.

Return ONLY valid JSON as an array.
Each item must follow this schema:

[
  {{
    "question": "string",
    "type": "single_choice" or "true_false",
    "options": ["A", "B", "C", "D"] OR ["True", "False"],
    "correct_answer": "one of the options exactly",
    "explanation": "string",
    "lesson_title": "string"
  }}
]

Rules:
- Use ONLY the information from the lesson summaries below.
- At least 70% of questions should be single_choice.
- true_false questions must use exactly ["True", "False"].
- single_choice questions must have exactly 4 options.
- Only 1 option must be correct.
- Options must be plausible.
- Explanations should be concise.
- Avoid duplicate questions.
- Cover multiple lessons, not only one lesson.
- lesson_title must match one of the provided lesson titles exactly.

LESSON TITLES:
{json.dumps(lesson_titles, ensure_ascii=False, indent=2)}

LESSON SUMMARIES:
{json.dumps(lesson_summaries, ensure_ascii=False, indent=2)}
"""
    data = retry_llm_json(llm, prompt)
    if not isinstance(data, list):
        raise ValueError("Quiz output is not a JSON array.")

    cleaned_questions = []
    for item in data:
        validated = validate_question_item(item, allow_true_false=True)
        if validated:
            lesson_title = str(item.get("lesson_title", "")).strip()
            validated["lesson_title"] = lesson_title if lesson_title in lesson_titles else ""
            cleaned_questions.append(validated)

    cleaned_questions = deduplicate_questions(cleaned_questions)
    if not cleaned_questions:
        raise ValueError("No valid quiz questions were generated.")
    return cleaned_questions[:quiz_questions]


def review_quiz(llm: OllamaLLM, quiz_data: List[Dict[str, Any]], lesson_titles: List[str], language: str) -> List[Dict[str, Any]]:
    prompt = f"""
You are reviewing a generated training quiz.

{get_language_instruction(language)}

Return ONLY valid JSON as an array.

Your job:
- remove duplicates
- improve wording
- make distractors more plausible
- keep the number of questions the same or slightly smaller
- preserve schema:
[
  {{
    "question": "string",
    "type": "single_choice" or "true_false",
    "options": [...],
    "correct_answer": "string",
    "explanation": "string",
    "lesson_title": "string"
  }}
]

Allowed lesson titles:
{json.dumps(lesson_titles, ensure_ascii=False, indent=2)}

QUIZ:
{json.dumps(quiz_data, ensure_ascii=False, indent=2)}
"""
    reviewed = retry_llm_json(llm, prompt)
    if not isinstance(reviewed, list):
        return quiz_data

    cleaned_questions = []
    for item in reviewed:
        validated = validate_question_item(item, allow_true_false=True)
        if validated:
            lesson_title = str(item.get("lesson_title", "")).strip()
            validated["lesson_title"] = lesson_title if lesson_title in lesson_titles else ""
            cleaned_questions.append(validated)

    cleaned_questions = deduplicate_questions(cleaned_questions)
    return cleaned_questions if cleaned_questions else quiz_data


def ensure_minimum_quiz_coverage(quiz_data: List[Dict[str, Any]], lesson_titles: List[str]) -> Dict[str, Any]:
    covered = set()
    for q in quiz_data:
        title = str(q.get("lesson_title", "")).strip()
        if title:
            covered.add(title)

    missing = [title for title in lesson_titles if title not in covered]
    return {
        "covered_lessons": sorted(list(covered)),
        "missing_lessons": missing,
        "coverage_ratio": round(len(covered) / len(lesson_titles), 2) if lesson_titles else 1.0,
    }


# =========================
# HTML builder
# =========================
def build_question_card_html(question: Dict[str, Any], idx: int, prefix: str, language: str) -> str:
    qid = f"{prefix}-q-{idx}"
    question_text = html.escape(str(question.get("question", "")))
    explanation = html.escape(str(question.get("explanation", "")))
    options = question.get("options", [])
    correct_answer = str(question.get("correct_answer", ""))
    explanation_label = "Показать объяснение" if language == "ru" else "Show explanation"

    options_html = []
    for opt_idx, option in enumerate(options):
        option_escaped = html.escape(str(option))
        input_id = f"{qid}-opt-{opt_idx}"
        options_html.append(
            f"""
<label class="quiz-option" for="{input_id}">
  <input type="radio" name="{qid}" id="{input_id}" value="{option_escaped}" data-correct="{html.escape(correct_answer)}" />
  <span>{option_escaped}</span>
</label>
"""
        )

    lesson_badge = ""
    if question.get("lesson_title"):
        lesson_badge = f'<div class="question-lesson">{html.escape(str(question["lesson_title"]))}</div>'

    return f"""
<div class="quiz-card" data-question="{qid}">
  {lesson_badge}
  <div class="quiz-question">{idx}. {question_text}</div>
  <div class="quiz-options">
    {''.join(options_html)}
  </div>
  <details class="quiz-explanation">
    <summary>{explanation_label}</summary>
    <p>{explanation}</p>
  </details>
</div>
"""


def build_glossary_html(glossary: List[Dict[str, Any]]) -> str:
    if not glossary:
        return "<p class='muted'>No glossary items generated.</p>"

    items = []
    for entry in glossary:
        term = html.escape(str(entry.get("term", "")).strip())
        definition = html.escape(str(entry.get("definition", "")).strip())
        if not term or not definition:
            continue
        items.append(
            f"""
<div class="glossary-item">
  <h3>{term}</h3>
  <p>{definition}</p>
</div>
"""
        )

    return "\n".join(items) if items else "<p class='muted'>No glossary items generated.</p>"


def build_markdown_summary(outline: Dict[str, Any], docs_info: List[Dict[str, Any]], lesson_payloads: List[Dict[str, Any]], language: str) -> str:
    is_ru = language == "ru"
    title = outline.get("course_title", "Generated Course")
    description = outline.get("course_description", "")
    target = outline.get("target_audience", "")
    prerequisites = outline.get("prerequisites", [])
    learning_outcomes = outline.get("learning_outcomes", [])
    lessons = outline.get("lessons", [])

    lines = [f"# {title}", "", description, "", f"**{'Целевая аудитория' if is_ru else 'Target audience'}:** {target}", ""]

    if prerequisites:
        lines.append(f"## {'Требования' if is_ru else 'Prerequisites'}")
        lines.extend([f"- {item}" for item in prerequisites])
        lines.append("")

    if learning_outcomes:
        lines.append(f"## {'Результаты обучения' if is_ru else 'Learning outcomes'}")
        lines.extend([f"- {item}" for item in learning_outcomes])
        lines.append("")

    lines.append(f"## {'Документы' if is_ru else 'Documents'}")
    lines.extend([f"- {doc['name']} ({doc['pages']} sections, {doc.get('type', 'unknown')})" for doc in docs_info])
    lines.append("")

    lines.append(f"## {'Уроки' if is_ru else 'Lessons'}")
    for idx, lesson in enumerate(lessons, start=1):
        lines.append(f"### {idx}. {lesson.get('title', '')}")
        lines.append(lesson.get("goal", ""))
        key_points = lesson.get("key_points", [])
        if key_points:
            lines.extend([f"- {item}" for item in key_points])
        summary = lesson_payloads[idx - 1].get("summary", "") if idx - 1 < len(lesson_payloads) else ""
        if summary:
            lines.append("")
            lines.append(summary)
        lines.append("")

    return "\n".join(lines)


def build_course_html(
    outline: Dict[str, Any],
    lesson_payloads: List[Dict[str, Any]],
    docs_info: List[Dict[str, Any]],
    pretest_data: List[Dict[str, Any]],
    final_quiz_data: List[Dict[str, Any]],
    language: str,
    include_source_excerpts: bool,
) -> str:
    is_ru = language == "ru"

    labels = {
        "overview": "Обзор" if is_ru else "Overview",
        "prerequisites": "Требования" if is_ru else "Prerequisites",
        "pretest": "Входной тест" if is_ru else "Pre-test",
        "glossary": "Глоссарий" if is_ru else "Glossary",
        "final_quiz": "Итоговый тест" if is_ru else "Final quiz",
        "learning_outcomes": "Результаты обучения" if is_ru else "Learning outcomes",
        "documents_used": "Использованные документы" if is_ru else "Documents used",
        "target_audience": "Целевая аудитория" if is_ru else "Target audience",
        "estimated_duration": "Оценочная длительность" if is_ru else "Estimated duration",
        "lessons": "Уроки" if is_ru else "Lessons",
        "documents_count": "Документы" if is_ru else "Documents used",
        "final_questions": "Вопросы финального теста" if is_ru else "Final quiz questions",
        "pretest_intro": "Пройдите короткий диагностический тест перед изучением уроков." if is_ru else "Use this short diagnostic quiz before starting the lessons.",
        "final_intro": "Пройдите итоговый тест, чтобы проверить усвоение материала." if is_ru else "Complete the final quiz to check knowledge gained through the course.",
        "check_pretest": "Проверить входной тест" if is_ru else "Check pre-test",
        "check_final": "Проверить итоговый тест" if is_ru else "Check final quiz",
        "reset": "Сбросить" if is_ru else "Reset",
        "sources_used": "Использованные источники" if is_ru else "Sources used",
        "source_excerpts": "Фрагменты источников" if is_ru else "Source excerpts",
        "generated_on": "Сгенерировано" if is_ru else "Generated on",
        "minutes": "минут" if is_ru else "minutes",
        "ai_generated_course": "Курс, сгенерированный ИИ" if is_ru else "AI-generated course",
        "from_source_docs": "Курс создан на основе исходных документов" if is_ru else "AI-generated course from source documents",
        "score": "Результат" if is_ru else "Score",
    }

    course_title = html.escape(outline.get("course_title", "Generated Course"))
    course_description = html.escape(outline.get("course_description", ""))
    target_audience = html.escape(outline.get("target_audience", ""))
    learning_outcomes = outline.get("learning_outcomes", [])
    prerequisites = outline.get("prerequisites", [])
    glossary = outline.get("glossary", [])

    overview_items = "\n".join(f"<li>{html.escape(str(item))}</li>" for item in learning_outcomes) or "<li>No learning outcomes generated.</li>"
    prerequisites_items = "\n".join(f"<li>{html.escape(str(item))}</li>" for item in prerequisites) or "<li>No prerequisites specified.</li>"
    docs_items = "\n".join(
        f"<li><strong>{html.escape(doc['name'])}</strong> — {doc['pages']} sections — {html.escape(doc.get('type', 'unknown'))}</li>"
        for doc in docs_info
    ) or "<li>No documents listed.</li>"

    toc_items = [
        f'<li><a href="#overview">{labels["overview"]}</a></li>',
        f'<li><a href="#prerequisites">{labels["prerequisites"]}</a></li>',
    ]
    if pretest_data:
        toc_items.append(f'<li><a href="#pretest">{labels["pretest"]}</a></li>')
    for i, lesson in enumerate(outline.get("lessons", []), start=1):
        title = html.escape(lesson.get("title", f"Lesson {i}"))
        toc_items.append(f'<li><a href="#lesson-{i}">{title}</a></li>')
    toc_items.append(f'<li><a href="#glossary">{labels["glossary"]}</a></li>')
    if final_quiz_data:
        toc_items.append(f'<li><a href="#final-quiz">{labels["final_quiz"]}</a></li>')
    toc_html = "\n".join(toc_items)

    lesson_sections = []
    for i, payload in enumerate(lesson_payloads, start=1):
        raw_section = payload["lesson_html"]
        section_with_anchor = raw_section.replace('<section class="lesson-section">', f'<section class="lesson-section" id="lesson-{i}">', 1)

        sources = payload.get("sources", [])
        source_items = "\n".join(
            f"<li>{html.escape(str(src['document_name']))} — page {html.escape(str(src['page']))}, chunk {html.escape(str(src['chunk_id']))}</li>"
            for src in sources
        )
        if source_items:
            section_with_anchor += f"""
<div class="lesson-sources">
  <h3>{labels['sources_used']}</h3>
  <ul>{source_items}</ul>
</div>
"""

        excerpts = payload.get("source_excerpts", []) if include_source_excerpts else []
        if excerpts:
            excerpt_items = "".join(
                f"<div class=\"source-excerpt\"><strong>{html.escape(str(e['document_name']))}</strong> — page {html.escape(str(e['page']))}, chunk {html.escape(str(e['chunk_id']))}<p>{html.escape(str(e['excerpt']))}</p></div>"
                for e in excerpts
            )
            section_with_anchor += f"""
<div class="lesson-excerpts">
  <h3>{labels['source_excerpts']}</h3>
  {excerpt_items}
</div>
"""
        lesson_sections.append(section_with_anchor)

    lesson_sections_html = "\n\n".join(lesson_sections)
    pretest_html = "".join(build_question_card_html(q, idx + 1, "pretest", language) for idx, q in enumerate(pretest_data))
    final_quiz_html = "".join(build_question_card_html(q, idx + 1, "final", language) for idx, q in enumerate(final_quiz_data))
    glossary_html = build_glossary_html(glossary)

    lesson_count = len(outline.get("lessons", []))
    estimated_minutes = estimate_duration_minutes(outline)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    pretest_section = f"""
      <section class="card" id="pretest">
        <h2>{labels['pretest']}</h2>
        <p class="muted">{labels['pretest_intro']}</p>
        <div class="quiz-actions">
          <button type="button" onclick="gradeQuiz('pretest')">{labels['check_pretest']}</button>
          <button type="button" class="secondary" onclick="resetQuiz('pretest')">{labels['reset']}</button>
        </div>
        <div class="progress-wrap"><div class="progress-bar" id="pretest-progress"></div></div>
        <div class="quiz-results" id="pretest-results"></div>
        {pretest_html}
      </section>
""" if pretest_data else ""

    final_quiz_section = f"""
      <section class="card" id="final-quiz">
        <h2>{labels['final_quiz']}</h2>
        <p class="muted">{labels['final_intro']}</p>
        <div class="quiz-actions">
          <button type="button" onclick="gradeQuiz('final')">{labels['check_final']}</button>
          <button type="button" class="secondary" onclick="resetQuiz('final')">{labels['reset']}</button>
        </div>
        <div class="progress-wrap"><div class="progress-bar" id="final-progress"></div></div>
        <div class="quiz-results" id="final-results"></div>
        {final_quiz_html}
      </section>
""" if final_quiz_data else ""

    return f"""<!DOCTYPE html>
<html lang="{language}">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{course_title}</title>
  <style>
    :root {{ --bg:#0f172a; --panel:#ffffff; --text:#1f2937; --muted:#64748b; --line:#e2e8f0; --primary:#2563eb; --primary-soft:#dbeafe; --shadow:0 10px 30px rgba(15,23,42,0.08); --radius:18px; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Arial, Helvetica, sans-serif; background:#f8fafc; color:var(--text); line-height:1.65; }}
    .layout {{ display:grid; grid-template-columns:280px 1fr; min-height:100vh; }}
    .sidebar {{ position:sticky; top:0; align-self:start; height:100vh; overflow:auto; background:var(--bg); color:#fff; padding:24px 18px; }}
    .sidebar h2 {{ margin:0 0 8px; font-size:1.25rem; }}
    .sidebar p {{ color:#cbd5e1; margin-top:0; font-size:0.95rem; }}
    .sidebar ol {{ padding-left:18px; margin:18px 0 0; }}
    .sidebar li {{ margin-bottom:10px; }}
    .sidebar a {{ color:#bfdbfe; text-decoration:none; }}
    .sidebar a:hover {{ text-decoration:underline; }}
    .content {{ padding:28px; max-width:1100px; width:100%; margin:0 auto; }}
    .card {{ background:var(--panel); border-radius:var(--radius); padding:24px; box-shadow:var(--shadow); margin-bottom:20px; }}
    .hero {{ padding:28px; }}
    .hero h1 {{ margin:8px 0 12px; font-size:2.2rem; line-height:1.2; color:#0f172a; }}
    .badge {{ display:inline-block; background:var(--primary-soft); color:#1d4ed8; padding:6px 10px; border-radius:999px; font-size:0.9rem; margin-right:8px; margin-bottom:8px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:14px; margin-top:16px; }}
    .stat {{ border:1px solid var(--line); border-radius:14px; padding:14px; background:#fff; }}
    .stat-label {{ color:var(--muted); font-size:0.9rem; }}
    .stat-value {{ margin-top:6px; font-size:1.2rem; font-weight:700; color:#0f172a; }}
    h2, h3 {{ color:#0f172a; }}
    ul, ol {{ padding-left:20px; }}
    .muted {{ color:var(--muted); }}
    .lesson-section {{ background:#fff; border-radius:var(--radius); padding:24px; box-shadow:var(--shadow); margin-bottom:12px; }}
    .lesson-sources, .lesson-excerpts {{ margin-top:18px; padding-top:16px; border-top:1px solid var(--line); }}
    .source-excerpt {{ border:1px solid var(--line); padding:12px; border-radius:12px; margin-bottom:10px; background:#f8fafc; }}
    .glossary-grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(240px, 1fr)); gap:14px; }}
    .glossary-item {{ border:1px solid var(--line); border-radius:14px; padding:16px; background:#fff; }}
    .quiz-actions {{ display:flex; flex-wrap:wrap; gap:10px; margin-bottom:18px; }}
    button {{ border:0; background:var(--primary); color:#fff; padding:10px 14px; border-radius:12px; cursor:pointer; font-size:0.95rem; }}
    button.secondary {{ background:#334155; }}
    .quiz-card {{ border:1px solid var(--line); border-radius:16px; padding:18px; margin-bottom:14px; background:#fff; }}
    .question-lesson {{ display:inline-block; margin-bottom:10px; font-size:0.8rem; background:#eef2ff; color:#4338ca; padding:4px 10px; border-radius:999px; }}
    .quiz-question {{ font-weight:700; margin-bottom:10px; }}
    .quiz-options {{ display:grid; gap:10px; }}
    .quiz-option {{ display:flex; align-items:center; gap:10px; border:1px solid var(--line); border-radius:12px; padding:10px 12px; cursor:pointer; transition:0.2s ease; }}
    .quiz-option:hover {{ background:#f8fafc; }}
    .quiz-option.correct {{ border-color:#86efac; background:#f0fdf4; }}
    .quiz-option.incorrect {{ border-color:#fca5a5; background:#fef2f2; }}
    .quiz-explanation {{ margin-top:12px; color:var(--muted); }}
    .quiz-results {{ margin-top:14px; padding:14px; border-radius:14px; background:#eff6ff; color:#1e3a8a; display:none; }}
    .progress-wrap {{ height:10px; border-radius:999px; background:#e2e8f0; overflow:hidden; margin-top:12px; }}
    .progress-bar {{ height:100%; width:0%; background:linear-gradient(90deg, #2563eb, #38bdf8); transition:width 0.3s ease; }}
    .footer {{ margin-top:28px; color:var(--muted); font-size:0.95rem; text-align:center; }}
    @media (max-width:980px) {{ .layout {{ grid-template-columns:1fr; }} .sidebar {{ position:relative; height:auto; }} .content {{ padding:18px; }} }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <h2>{course_title}</h2>
      <p>{labels['from_source_docs']}</p>
      <ol>{toc_html}</ol>
    </aside>
    <main class="content">
      <section class="card hero" id="overview">
        <span class="badge">{labels['ai_generated_course']}</span>
        <span class="badge">HTML</span>
        <span class="badge">{lesson_count} {labels['lessons'].lower()}</span>
        <h1>{course_title}</h1>
        <p>{course_description}</p>
        <p><strong>{labels['target_audience']}:</strong> {target_audience}</p>
        <div class="grid">
          <div class="stat"><div class="stat-label">{labels['estimated_duration']}</div><div class="stat-value">{estimated_minutes} {labels['minutes']}</div></div>
          <div class="stat"><div class="stat-label">{labels['lessons']}</div><div class="stat-value">{lesson_count}</div></div>
          <div class="stat"><div class="stat-label">{labels['documents_count']}</div><div class="stat-value">{len(docs_info)}</div></div>
          <div class="stat"><div class="stat-label">{labels['final_questions']}</div><div class="stat-value">{len(final_quiz_data)}</div></div>
        </div>
      </section>
      <section class="card" id="prerequisites"><h2>{labels['prerequisites']}</h2><ul>{prerequisites_items}</ul></section>
      <section class="card"><h2>{labels['learning_outcomes']}</h2><ul>{overview_items}</ul></section>
      <section class="card"><h2>{labels['documents_used']}</h2><ul>{docs_items}</ul></section>
      {pretest_section}
      {lesson_sections_html}
      <section class="card" id="glossary"><h2>{labels['glossary']}</h2><div class="glossary-grid">{glossary_html}</div></section>
      {final_quiz_section}
      <div class="footer">{labels['generated_on']} {generated_at}</div>
    </main>
  </div>
  <script>
    const SCORE_LABEL = {json.dumps(labels['score'])};
    function getQuizCards(prefix) {{ return Array.from(document.querySelectorAll(`[data-question^="${{prefix}}-q-"]`)); }}
    function updateProgress(prefix) {{ const cards = getQuizCards(prefix); const answered = cards.filter(card => card.querySelector('input[type="radio"]:checked')).length; const total = cards.length || 1; const percent = Math.round((answered / total) * 100); const bar = document.getElementById(`${{prefix}}-progress`); if (bar) bar.style.width = `${{percent}}%`; }}
    function gradeQuiz(prefix) {{ const cards = getQuizCards(prefix); let correct = 0; cards.forEach(card => {{ const selected = card.querySelector('input[type="radio"]:checked'); const options = card.querySelectorAll('.quiz-option'); options.forEach(opt => opt.classList.remove('correct','incorrect')); if (!selected) return; const correctAnswer = selected.getAttribute('data-correct'); const optionLabels = card.querySelectorAll('.quiz-option'); optionLabels.forEach(label => {{ const input = label.querySelector('input'); if (!input) return; if (input.value === correctAnswer) label.classList.add('correct'); if (input.checked && input.value !== correctAnswer) label.classList.add('incorrect'); }}); if (selected.value === correctAnswer) correct += 1; }}); const results = document.getElementById(`${{prefix}}-results`); if (results) {{ const total = cards.length; const percent = total ? Math.round((correct / total) * 100) : 0; results.style.display = 'block'; results.innerHTML = `<strong>${{SCORE_LABEL}}:</strong> ${{correct}} / ${{total}} (${{percent}}%)`; }} updateProgress(prefix); }}
    function resetQuiz(prefix) {{ const cards = getQuizCards(prefix); cards.forEach(card => {{ card.querySelectorAll('input[type="radio"]').forEach(input => input.checked = false); card.querySelectorAll('.quiz-option').forEach(opt => opt.classList.remove('correct','incorrect')); }}); const results = document.getElementById(`${{prefix}}-results`); if (results) {{ results.style.display = 'none'; results.innerHTML = ''; }} updateProgress(prefix); }}
    document.addEventListener('change', (event) => {{ const target = event.target; if (target.matches('input[type="radio"]')) {{ const name = target.name || ''; if (name.startsWith('pretest-')) updateProgress('pretest'); if (name.startsWith('final-')) updateProgress('final'); }} }});
    updateProgress('pretest'); updateProgress('final');
  </script>
</body>
</html>
"""


# =========================
# Save outputs
# =========================
def save_json(path: Path, data: Any) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return str(path)


def save_text(path: Path, text: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return str(path)


def save_course_html(output_dir: str, html_text: str, output_prefix: str = "") -> str:
    return save_text(build_output_path(output_dir, "course.html", output_prefix), html_text)


def save_quiz_json(output_dir: str, quiz_data: List[Dict[str, Any]], output_prefix: str = "") -> str:
    return save_json(build_output_path(output_dir, "quiz.json", output_prefix), quiz_data)


def save_pretest_json(output_dir: str, pretest_data: List[Dict[str, Any]], output_prefix: str = "") -> str:
    return save_json(build_output_path(output_dir, "pretest.json", output_prefix), pretest_data)


def save_outline_json(output_dir: str, outline: Dict[str, Any], output_prefix: str = "") -> str:
    return save_json(build_output_path(output_dir, "course_outline.json", output_prefix), outline)


def save_markdown_summary(output_dir: str, markdown_text: str, output_prefix: str = "") -> str:
    return save_text(build_output_path(output_dir, "course_summary.md", output_prefix), markdown_text)


def save_lesson_summaries(output_dir: str, lesson_payloads: List[Dict[str, Any]], outline: Dict[str, Any], output_prefix: str = "") -> str:
    path = build_output_path(output_dir, "lesson_summaries.json", output_prefix)
    lessons = outline.get("lessons", [])
    payload = []
    for idx, lesson in enumerate(lessons):
        lesson_payload = lesson_payloads[idx] if idx < len(lesson_payloads) else {}
        payload.append({
            "lesson_number": idx + 1,
            "title": lesson.get("title", ""),
            "goal": lesson.get("goal", ""),
            "key_points": lesson.get("key_points", []),
            "summary": lesson_payload.get("summary", ""),
            "key_takeaways": lesson_payload.get("key_takeaways", []),
            "sources": lesson_payload.get("sources", []),
            "source_excerpts": lesson_payload.get("source_excerpts", []),
        })
    return save_json(path, payload)


def save_generation_report(output_dir: str, outline: Dict[str, Any], docs_info: List[Dict[str, Any]], pretest_data: List[Dict[str, Any]], quiz_data: List[Dict[str, Any]], args: argparse.Namespace, elapsed_seconds: float) -> str:
    path = build_output_path(output_dir, "generation_report.json", args.output_prefix)
    report = {
        "generated_at": datetime.now().isoformat(),
        "model": args.model,
        "embedding_model": args.embedding_model,
        "difficulty": args.difficulty,
        "retrieval_type": args.retrieval_type,
        "language": args.language,
        "output_prefix": args.output_prefix,
        "top_k": args.top_k,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "documents_count": len(docs_info),
        "documents": docs_info,
        "lessons_count": len(outline.get("lessons", [])),
        "pretest_questions_count": len(pretest_data),
        "final_quiz_questions_count": len(quiz_data),
        "estimated_duration_minutes": estimate_duration_minutes(outline),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "quiz_coverage": ensure_minimum_quiz_coverage(quiz_data, [lesson.get("title", "") for lesson in outline.get("lessons", [])]) if quiz_data else {"covered_lessons": [], "missing_lessons": [], "coverage_ratio": 0},
        "source_extensions": sorted(list({doc.get("type", "unknown") for doc in docs_info})),
        "skip_pretest": args.skip_pretest,
        "skip_final_quiz": args.skip_final_quiz,
        "include_source_excerpts": args.include_source_excerpts,
    }
    return save_json(path, report)


def save_course_metadata(output_dir: str, outline: Dict[str, Any], docs_info: List[Dict[str, Any]], args: argparse.Namespace) -> str:
    path = build_output_path(output_dir, "course_metadata.json", args.output_prefix)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "model": args.model,
        "embedding_model": args.embedding_model,
        "difficulty": args.difficulty,
        "retrieval_type": args.retrieval_type,
        "language": args.language,
        "output_prefix": args.output_prefix,
        "top_k": args.top_k,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "documents": docs_info,
        "outline": outline,
    }
    return save_json(path, payload)


def save_course_bundle(output_dir: str, outline: Dict[str, Any], docs_info: List[Dict[str, Any]], lesson_payloads: List[Dict[str, Any]], pretest_data: List[Dict[str, Any]], quiz_data: List[Dict[str, Any]], args: argparse.Namespace) -> str:
    path = build_output_path(output_dir, "course_bundle.json", args.output_prefix)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "model": args.model,
            "embedding_model": args.embedding_model,
            "difficulty": args.difficulty,
            "retrieval_type": args.retrieval_type,
            "language": args.language,
            "output_prefix": args.output_prefix,
        },
        "documents": docs_info,
        "outline": outline,
        "lessons": lesson_payloads,
        "pretest": pretest_data,
        "final_quiz": quiz_data,
    }
    return save_json(path, payload)


# =========================
# Main
# =========================
def main() -> None:
    args = parse_args()
    ensure_directories(args.docs_path, args.db, args.manifest_file, args.output_dir, args.log_dir)

    if args.min_lessons < 1 or args.max_lessons < args.min_lessons:
        print("(X) Invalid lesson range. Check --min-lessons and --max-lessons.")
        sys.exit(1)

    started = time.time()
    log_message(args.log_dir, "Starting course and quiz generation pipeline")

    source_files = collect_source_files(args.docs_path)
    print(f"Found {len(source_files)} source file(s).")

    try:
        vectorstore, docs_info = load_or_create_vectorstore(args, source_files)
        log_message(args.log_dir, f"Vector store ready. Documents loaded: {len(docs_info)}")
    except Exception as exc:
        print(f"(X) Failed to prepare vector store: {exc}")
        log_message(args.log_dir, f"Vector store error: {exc}")
        sys.exit(1)

    try:
        llm = OllamaLLM(model=args.model)
        log_message(args.log_dir, f"Initialized Ollama model: {args.model}")
    except Exception as exc:
        print(f"(X) Failed to initialize LLM: {exc}")
        log_message(args.log_dir, f"LLM init error: {exc}")
        sys.exit(1)

    try:
        print("--- Generating course outline... ---")
        preview_text = get_combined_preview_text(source_files, max_chars_per_file=args.max_preview_chars_per_file)
        outline = generate_course_outline(llm, preview_text, args.difficulty, args.language, args.min_lessons, args.max_lessons)
        if not args.disable_review_pass:
            outline = review_outline(llm, outline, args.language, args.min_lessons, args.max_lessons)
        log_message(args.log_dir, "Course outline generated successfully")
    except Exception as exc:
        print(f"(X) Failed to generate course outline: {exc}")
        log_message(args.log_dir, f"Course outline error: {exc}")
        sys.exit(1)

    lesson_payloads = []
    lessons = outline.get("lessons", [])
    try:
        for idx, lesson in enumerate(lessons, start=1):
            lesson_title = str(lesson.get("title", f"Lesson {idx}")).strip()
            lesson_goal = str(lesson.get("goal", "")).strip()
            key_points = lesson.get("key_points", []) if isinstance(lesson.get("key_points", []), list) else []
            print(f"--- Generating lesson {idx}/{len(lessons)}: {lesson_title} ---")
            log_message(args.log_dir, f"Generating lesson: {lesson_title}")
            retrieved_docs = retrieve_lesson_context(vectorstore, lesson_title, key_points, args.top_k, args.retrieval_type)
            lesson_payloads.append(generate_lesson_html_section(llm, lesson_title, lesson_goal, key_points, retrieved_docs, args.language, args.include_source_excerpts))
        log_message(args.log_dir, "All lesson sections generated successfully")
    except Exception as exc:
        print(f"(X) Failed during lesson generation: {exc}")
        log_message(args.log_dir, f"Lesson generation error: {exc}")
        sys.exit(1)

    pretest_data: List[Dict[str, Any]] = []
    if not args.skip_pretest:
        try:
            print("--- Generating pre-test... ---")
            pretest_data = generate_pretest(llm, outline, args.pretest_questions, args.difficulty, args.language)
            log_message(args.log_dir, f"Pre-test generated successfully with {len(pretest_data)} questions")
        except Exception as exc:
            print(f"(X) Failed to generate pre-test: {exc}")
            log_message(args.log_dir, f"Pre-test generation error: {exc}")
            sys.exit(1)

    quiz_data: List[Dict[str, Any]] = []
    if not args.skip_final_quiz:
        try:
            print("--- Generating final quiz... ---")
            quiz_data = generate_quiz(llm, outline, lesson_payloads, args.difficulty, args.quiz_questions, args.language)
            if not args.disable_review_pass:
                quiz_data = review_quiz(llm, quiz_data, [lesson.get("title", "") for lesson in outline.get("lessons", [])], args.language)
            log_message(args.log_dir, f"Quiz generated successfully with {len(quiz_data)} questions")
        except Exception as exc:
            print(f"(X) Failed to generate quiz: {exc}")
            log_message(args.log_dir, f"Quiz generation error: {exc}")
            sys.exit(1)

    try:
        print("--- Building final HTML course... ---")
        course_html = build_course_html(outline, lesson_payloads, docs_info, pretest_data, quiz_data, args.language, args.include_source_excerpts)
        markdown_summary = build_markdown_summary(outline, docs_info, lesson_payloads, args.language)

        course_path = save_course_html(args.output_dir, course_html, args.output_prefix)
        outline_path = save_outline_json(args.output_dir, outline, args.output_prefix)
        quiz_path = save_quiz_json(args.output_dir, quiz_data, args.output_prefix)
        pretest_path = save_pretest_json(args.output_dir, pretest_data, args.output_prefix)
        metadata_path = save_course_metadata(args.output_dir, outline, docs_info, args)
        summaries_path = save_lesson_summaries(args.output_dir, lesson_payloads, outline, args.output_prefix)
        markdown_path = save_markdown_summary(args.output_dir, markdown_summary, args.output_prefix)
        bundle_path = save_course_bundle(args.output_dir, outline, docs_info, lesson_payloads, pretest_data, quiz_data, args)

        elapsed = time.time() - started
        report_path = save_generation_report(args.output_dir, outline, docs_info, pretest_data, quiz_data, args, elapsed)

        print("\n[SUCCESS] Generation complete!")
        print(f"Course HTML:        {course_path}")
        print(f"Course outline:     {outline_path}")
        print(f"Pre-test JSON:      {pretest_path}")
        print(f"Final quiz JSON:    {quiz_path}")
        print(f"Lesson summaries:   {summaries_path}")
        print(f"Markdown summary:   {markdown_path}")
        print(f"Metadata:           {metadata_path}")
        print(f"Bundle:             {bundle_path}")
        print(f"Report:             {report_path}")
        print(f"Time:               {elapsed:.2f}s")

        log_message(args.log_dir, f"Generation complete. course={course_path}, outline={outline_path}, pretest={pretest_path}, quiz={quiz_path}, summaries={summaries_path}, markdown={markdown_path}, metadata={metadata_path}, bundle={bundle_path}, report={report_path}, elapsed={elapsed:.2f}s")
    except Exception as exc:
        print(f"(X) Failed to save outputs: {exc}")
        log_message(args.log_dir, f"Save outputs error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
