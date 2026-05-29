from argparse import Namespace
from pathlib import Path

from course_generator.documents import DocCollection
from course_generator.planning import build_run_plan, estimate_llm_calls


def test_estimate_llm_calls_full_course():
    args = Namespace(
        disable_review_pass=False,
        skip_pretest=False,
        skip_final_quiz=False,
        quality_llm_review=False,
    )
    assert estimate_llm_calls(args, 5) == 1 + 1 + 5 + 1 + 1 + 1  # outline, review, lessons, pretest, quiz, quiz review


def test_build_run_plan_lists_documents(tmp_path):
    doc = tmp_path / "a.txt"
    doc.write_text("hello", encoding="utf-8")
    dc = DocCollection(files=[doc], root=tmp_path)
    args = Namespace(
        min_lessons=2,
        max_lessons=4,
        recursive_docs=False,
        from_outline=None,
        outline_only=False,
        skip_outline_rag=True,
        disable_review_pass=True,
        skip_pretest=True,
        skip_final_quiz=True,
        export_docx=False,
        no_delivery_zip=True,
        model="llama3",
        embedding_model="mini",
    )
    plan = build_run_plan(dc, args)
    assert plan["document_count"] == 1
    assert plan["documents"] == ["a.txt"]
    assert "Build or load FAISS index" in plan["pipeline_steps"][0]
