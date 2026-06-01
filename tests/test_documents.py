import pytest

from course_generator.documents import collect_source_files, document_display_name
from course_generator.errors import DocumentSourceError


def test_collect_recursive_includes_nested(tmp_path):
    (tmp_path / "root.txt").write_text("a", encoding="utf-8")
    sub = tmp_path / "inner"
    sub.mkdir()
    (sub / "leaf.md").write_text("b", encoding="utf-8")

    flat = collect_source_files(str(tmp_path), recursive=False)
    assert len(flat.files) == 1
    assert flat.files[0].name == "root.txt"

    rec = collect_source_files(str(tmp_path), recursive=True)
    assert len(rec.files) == 2
    names = sorted(f.name for f in rec.files)
    assert names == ["leaf.md", "root.txt"]


def test_collect_missing_path_raises(tmp_path):
    with pytest.raises(DocumentSourceError, match="does not exist"):
        collect_source_files(str(tmp_path / "missing"), recursive=False)


def test_document_display_name_under_root(tmp_path):
    root = tmp_path.resolve()
    sub = tmp_path / "ch"
    sub.mkdir()
    fp = sub / "note.txt"
    fp.write_text("x", encoding="utf-8")

    lab = document_display_name(fp.resolve(), root)
    norm = lab.replace("\\", "/")
    assert norm.endswith("ch/note.txt")
