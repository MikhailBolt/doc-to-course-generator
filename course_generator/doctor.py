from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List

from course_generator.bundle_io import find_latest_bundle_path
from course_generator.constants import SUPPORTED_EXTENSIONS
from course_generator.documents import collect_source_files
from course_generator.errors import DocumentSourceError
from course_generator.health import check_embeddings, check_ollama
from course_generator.history import list_recent_bundles, list_recent_reports


def run_doctor(args: Namespace, *, check_embeddings_model: bool = False) -> Dict[str, Any]:
    """Run lightweight environment and project readiness checks."""
    from course_generator import __version__

    checks: List[Dict[str, Any]] = []

    def add(check_id: str, label: str, ok: bool, detail: str = "", **extra: Any) -> None:
        row: Dict[str, Any] = {"id": check_id, "label": label, "ok": ok, "detail": detail}
        row.update(extra)
        checks.append(row)

    ollama = check_ollama(args.model, timeout=float(getattr(args, "ollama_timeout", 120) or 5))
    add(
        "ollama",
        "Ollama",
        bool(ollama.get("ok") and ollama.get("model_available")),
        detail=f"host={ollama.get('host')} model={args.model}",
        models_count=ollama.get("models_count"),
    )

    docs_path = Path(args.docs_path)
    docs_ok = False
    docs_detail = ""
    docs_count = 0
    try:
        if docs_path.is_file():
            docs_ok = docs_path.suffix.lower() in SUPPORTED_EXTENSIONS
            docs_count = 1 if docs_ok else 0
            docs_detail = docs_path.name
        elif docs_path.is_dir():
            dc = collect_source_files(
                args.docs_path,
                recursive=getattr(args, "recursive_docs", False),
                max_files=None,
            )
            docs_count = len(dc.files)
            docs_ok = docs_count > 0
            docs_detail = f"{docs_count} file(s) under {args.docs_path}"
        else:
            docs_detail = f"path not found: {args.docs_path}"
    except DocumentSourceError as exc:
        docs_detail = str(exc)
    add("docs", "Source documents", docs_ok, detail=docs_detail, document_count=docs_count)

    output_dir = Path(args.output_dir)
    output_ok = False
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_ok = True
            output_detail = f"created {args.output_dir}"
        except Exception as exc:
            output_detail = str(exc)
    else:
        output_ok = output_dir.is_dir()
        output_detail = f"{len(list(output_dir.glob('*')))} item(s)"
    add("output_dir", "Output directory", output_ok, detail=output_detail)

    faiss_path = Path(args.db)
    faiss_ok = faiss_path.is_dir() and any(faiss_path.iterdir()) if faiss_path.exists() else False
    add("faiss", "FAISS index", faiss_ok, detail=str(args.db))

    if check_embeddings_model:
        emb = check_embeddings(args.embedding_model)
        add(
            "embeddings",
            "Embedding model",
            bool(emb.get("ok")),
            detail=emb.get("error", f"dims={emb.get('dimensions', 0)}"),
            model=args.embedding_model,
        )

    bundle_path = ""
    try:
        bundle_path = find_latest_bundle_path(args.output_dir)
        add("latest_bundle", "Latest bundle", True, detail=bundle_path)
    except FileNotFoundError:
        add("latest_bundle", "Latest bundle", False, detail="no course_bundle.json found")

    recent_runs = list_recent_reports(args.output_dir, limit=3)
    add(
        "recent_runs",
        "Recent generation reports",
        bool(recent_runs),
        detail=f"{len(recent_runs)} report(s)" if recent_runs else "none",
    )

    required_ok = all(c["ok"] for c in checks if c["id"] in {"ollama", "docs", "output_dir"})
    optional_ids = {"faiss", "latest_bundle", "recent_runs", "embeddings"}
    optional_ok = all(c["ok"] for c in checks if c["id"] in optional_ids and c["id"] != "embeddings")

    return {
        "ok": required_ok,
        "generator_version": __version__,
        "ready_for_generation": required_ok,
        "optional_ready": optional_ok,
        "checks": checks,
        "latest_bundle": bundle_path or None,
        "recent_bundles": list_recent_bundles(args.output_dir, limit=3),
        "recent_runs": recent_runs,
    }
