import json

from course_generator.report_diff import diff_generation_reports


def test_diff_generation_reports(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    base = {
        "generated_at": "2026-01-01",
        "model": "llama3",
        "lessons_count": 5,
        "quality": {"overall_score": 80, "grade": "B"},
        "elapsed_seconds": 100,
    }
    a.write_text(json.dumps(base), encoding="utf-8")
    b.write_text(json.dumps({**base, "lessons_count": 6, "quality": {"overall_score": 90, "grade": "A"}}), encoding="utf-8")
    diff = diff_generation_reports(str(a), str(b))
    assert "lessons_count" in diff["deltas"]
    assert diff["deltas"]["lessons_count"]["a"] == 5
