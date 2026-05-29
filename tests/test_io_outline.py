import json

import pytest

from course_generator.io import load_outline_json


def test_load_outline_json_valid(tmp_path):
    path = tmp_path / "outline.json"
    path.write_text(
        json.dumps(
            {
                "course_title": "T",
                "course_description": "D",
                "target_audience": "Beginners",
                "prerequisites": [],
                "learning_outcomes": ["Learn X"],
                "glossary": [],
                "lessons": [
                    {
                        "title": "L1",
                        "goal": "G",
                        "key_points": ["a", "b"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    data = load_outline_json(str(path))
    assert data["course_title"] == "T"
    assert len(data["lessons"]) == 1


def test_load_outline_json_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_outline_json(str(tmp_path / "missing.json"))
