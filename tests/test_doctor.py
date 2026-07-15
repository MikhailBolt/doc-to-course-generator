import json
from argparse import Namespace
from pathlib import Path

from course_generator.doctor import run_doctor


def _args(tmp_path, **overrides):
    base = Namespace(
        docs_path=str(tmp_path / "docs"),
        db=str(tmp_path / "vectorstore"),
        manifest_file=str(tmp_path / "vectorstore" / "manifest.json"),
        output_dir=str(tmp_path / "output"),
        log_dir=str(tmp_path / "logs"),
        model="llama3",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        recursive_docs=False,
        ollama_timeout=5,
        check_embeddings=False,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_run_doctor_missing_docs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "course_generator.doctor.check_ollama",
        lambda model, timeout=5: {"ok": True, "model_available": True, "host": "http://127.0.0.1:11434"},
    )
    (tmp_path / "output").mkdir()
    report = run_doctor(_args(tmp_path))
    docs_check = next(c for c in report["checks"] if c["id"] == "docs")
    assert docs_check["ok"] is False
    assert report["ready_for_generation"] is False


def test_run_doctor_with_docs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "course_generator.doctor.check_ollama",
        lambda model, timeout=5: {"ok": True, "model_available": True, "host": "http://127.0.0.1:11434"},
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "sample.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "output").mkdir()
    report = run_doctor(_args(tmp_path, docs_path=str(docs)))
    assert report["ready_for_generation"] is True
    docs_check = next(c for c in report["checks"] if c["id"] == "docs")
    assert docs_check["ok"] is True
    assert docs_check["document_count"] == 1


def test_run_doctor_finds_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "course_generator.doctor.check_ollama",
        lambda model, timeout=5: {"ok": True, "model_available": True, "host": "http://127.0.0.1:11434"},
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "sample.txt").write_text("hello", encoding="utf-8")
    out = tmp_path / "output"
    out.mkdir()
    bundle = {
        "outline": {
            "course_title": "T",
            "course_description": "d",
            "target_audience": "a",
            "prerequisites": [],
            "learning_outcomes": ["x"],
            "glossary": [],
            "lessons": [{"title": "L", "goal": "g", "key_points": ["a"]}],
        },
        "lessons": [],
    }
    (out / "course_bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
    report = run_doctor(_args(tmp_path, docs_path=str(docs), output_dir=str(out)))
    bundle_check = next(c for c in report["checks"] if c["id"] == "latest_bundle")
    assert bundle_check["ok"] is True
