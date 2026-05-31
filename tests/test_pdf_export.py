from pathlib import Path

import pytest

from course_generator.pdf_export import export_markdown_to_pdf


def test_export_markdown_to_pdf(tmp_path):
    pytest.importorskip("reportlab")
    out = tmp_path / "summary.pdf"
    path = export_markdown_to_pdf("# Title\n\n- bullet one\n\nBody text.", out)
    assert Path(path).is_file()
    assert Path(path).stat().st_size > 100
