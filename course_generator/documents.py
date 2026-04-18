import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from langchain_community.document_loaders import PyPDFLoader, TextLoader

from course_generator.constants import SUPPORTED_EXTENSIONS
from course_generator.utils import clean_text


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
