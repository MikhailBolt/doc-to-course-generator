from course_generator.audit import audit_output_paths


def test_audit_output_paths_missing(tmp_path):
    issues = audit_output_paths({"course_html": str(tmp_path / "missing.html")})
    assert any("not found" in i.lower() for i in issues)
