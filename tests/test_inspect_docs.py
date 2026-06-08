from course_generator.inspect_docs import build_documents_report


def test_build_documents_report(tmp_path):
    (tmp_path / "a.txt").write_text("hello world", encoding="utf-8")
    (tmp_path / "b.md").write_text("# Title", encoding="utf-8")
    report = build_documents_report(str(tmp_path), recursive=False)
    assert report["document_count"] == 2
    assert "txt" in report["extensions"]
    assert report["total_size_bytes"] > 0
