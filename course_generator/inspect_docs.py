from typing import Any, Dict, List

from course_generator.documents import DocCollection, collect_source_files, document_display_name


def _human_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def build_documents_report(
    docs_path: str,
    *,
    recursive: bool = False,
    max_files: int | None = None,
) -> Dict[str, Any]:
    """Inspect source documents without building an index or calling the LLM."""
    dc = collect_source_files(docs_path, recursive=recursive, max_files=max_files)
    details: List[Dict[str, Any]] = []
    ext_counts: Dict[str, int] = {}

    for path in dc.files:
        ext = path.suffix.lower().lstrip(".")
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
        size = path.stat().st_size
        details.append(
            {
                "name": document_display_name(path, dc.root),
                "path": str(path.resolve()),
                "type": ext,
                "size_bytes": size,
                "size_human": _human_size(size),
            }
        )

    total_bytes = sum(d["size_bytes"] for d in details)
    return {
        "docs_path": docs_path,
        "docs_root": str(dc.root),
        "recursive": recursive,
        "document_count": len(details),
        "documents_total_found": dc.total_found or len(details),
        "truncated": dc.truncated,
        "max_files": max_files,
        "total_size_bytes": total_bytes,
        "total_size_human": _human_size(total_bytes),
        "extensions": ext_counts,
        "documents": details,
    }
