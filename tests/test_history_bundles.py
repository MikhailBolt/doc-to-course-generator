import json

from course_generator.history import list_recent_bundles


def test_list_recent_bundles(tmp_path):
    outline = {
        "course_title": "Test Course",
        "lessons": [{"title": "L1"}, {"title": "L2"}],
    }
    bundle = {
        "outline": outline,
        "lessons": [{"lesson_html": "<section></section>"}],
        "pretest": [{"q": 1}],
        "final_quiz": [{"q": 1}, {"q": 2}],
    }
    path = tmp_path / "course_bundle.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")

    rows = list_recent_bundles(str(tmp_path), limit=5)
    assert len(rows) == 1
    assert rows[0]["course_title"] == "Test Course"
    assert rows[0]["lessons_count"] == 2
    assert rows[0]["lesson_payloads_count"] == 1
    assert rows[0]["pretest_count"] == 1
    assert rows[0]["quiz_count"] == 2


def test_list_recent_bundles_empty_dir(tmp_path):
    assert list_recent_bundles(str(tmp_path)) == []
