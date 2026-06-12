import json

import pytest

from course_generator.bundle_io import load_bundle_json, parse_lesson_indices, validate_bundle_file


def _minimal_outline():
    return {
        "course_title": "T",
        "course_description": "D",
        "target_audience": "A",
        "prerequisites": [],
        "learning_outcomes": ["x"],
        "glossary": [],
        "lessons": [{"title": "L1", "goal": "G", "key_points": ["a", "b"]}],
    }


def test_parse_lesson_indices():
    assert parse_lesson_indices("1, 3,3") == [1, 3]
    assert parse_lesson_indices("") == []


def test_load_bundle_json(tmp_path):
    path = tmp_path / "bundle.json"
    path.write_text(
        json.dumps(
            {
                "outline": _minimal_outline(),
                "lessons": [{"lesson_html": "<section></section>", "summary": "s"}],
                "pretest": [],
                "final_quiz": [],
            }
        ),
        encoding="utf-8",
    )
    bundle = load_bundle_json(str(path))
    assert bundle["outline"]["course_title"] == "T"
    assert len(bundle["lessons"]) == 1


def test_validate_bundle_file(tmp_path):
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps({"outline": _minimal_outline(), "lessons": []}), encoding="utf-8")
    summary = validate_bundle_file(str(path))
    assert summary["ok"] is True


def test_load_bundle_missing_outline(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        load_bundle_json(str(path))
