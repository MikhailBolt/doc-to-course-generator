from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from course_generator.constants import DEFAULT_OUTLINE_RAG_MAX_CHARS, DEFAULT_OUTLINE_RAG_MAX_CHUNKS, DEFAULT_OUTLINE_RAG_TOP_K_PER_QUERY
from course_generator.documents import (
    build_manifest_data,
    is_index_stale,
    load_file_documents,
    save_manifest,
)
from course_generator.utils import clean_text


def _chunk_dedup_key(doc: Any) -> Tuple[Any, str]:
    return (doc.metadata.get("chunk_id"), str(doc.metadata.get("document_name", "")))


def retrieve_outline_context(
    vectorstore: FAISS,
    source_files: List[Path],
    *,
    retrieval_type: str,
    top_k_per_query: int = DEFAULT_OUTLINE_RAG_TOP_K_PER_QUERY,
    max_chunks: int = DEFAULT_OUTLINE_RAG_MAX_CHUNKS,
    max_chars: int = DEFAULT_OUTLINE_RAG_MAX_CHARS,
) -> str:
    """Collect diverse retrieved chunks to ground outline generation on RAG, not only file prefixes."""
    queries = [
        "main topics structure overview learning objectives course outline summary",
        "key concepts definitions terminology important facts",
        "procedures steps requirements examples use cases",
    ]
    collected: List[Any] = []
    seen: set = set()

    def add_docs(docs: List[Any]) -> None:
        nonlocal collected
        for doc in docs:
            key = _chunk_dedup_key(doc)
            if key in seen:
                continue
            seen.add(key)
            collected.append(doc)
            if len(collected) >= max_chunks:
                return

    fetch_k = max(top_k_per_query * 2, 8)

    for q in queries:
        if len(collected) >= max_chunks:
            break
        if retrieval_type == "mmr":
            docs = vectorstore.max_marginal_relevance_search(q, k=top_k_per_query, fetch_k=fetch_k)
        else:
            docs = vectorstore.similarity_search(q, k=top_k_per_query)
        add_docs(docs)

    for path in source_files:
        if len(collected) >= max_chunks:
            break
        q = f"important content concepts from document {path.name}"
        if retrieval_type == "mmr":
            docs = vectorstore.max_marginal_relevance_search(q, k=4, fetch_k=12)
        else:
            docs = vectorstore.similarity_search(q, k=4)
        add_docs(docs)

    blocks = []
    for doc in collected:
        doc_name = doc.metadata.get("document_name", "unknown")
        page = doc.metadata.get("page")
        page_num = page + 1 if isinstance(page, int) else "N/A"
        cid = doc.metadata.get("chunk_id", "N/A")
        text = clean_text(doc.page_content)
        blocks.append(f"[Document: {doc_name} | Page: {page_num} | Chunk: {cid}]\n{text}")

    joined = "\n\n".join(blocks)
    if len(joined) > max_chars:
        joined = joined[:max_chars] + "\n\n[... truncated ...]"
    return joined


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


def load_or_create_vectorstore(args: Namespace, source_files: List[Path]) -> Tuple[FAISS, List[Dict[str, Any]]]:
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


def retrieve_lesson_context(vectorstore: FAISS, lesson_title: str, key_points: List[str], top_k: int, retrieval_type: str) -> List[Any]:
    query = lesson_title
    if key_points:
        query += "\n" + "\n".join(key_points)

    if retrieval_type == "mmr":
        return vectorstore.max_marginal_relevance_search(query, k=top_k, fetch_k=max(top_k * 2, 8))
    return vectorstore.similarity_search(query, k=top_k)
