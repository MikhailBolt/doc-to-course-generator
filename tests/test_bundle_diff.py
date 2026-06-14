import json

from course_generator.bundle_diff import diff_course_bundles


def _minimal_bundle(title: str, lessons):
    return {
        "generated_at": "2026-01-01T00:00:00",
        "outline": {
            "course_title": title,
            "course_description": "D",
            "target_audience": "A",
            "prerequisites": [],
            "learning_outcomes": ["x"],
            "glossary": [],
            "lessons": [{"title": t, "goal": "G", "key_points": ["a"]} for t in lessons],
        },
        "lessons": [{"lesson_html": "<section></section>", "summary": "s"}] * len(lessons),
        "pretest": [],
        "final_quiz": [],
    }


def test_diff_course_bundles(tmp_path):
    path_a = tmp_path / "a_bundle.json"
    path_b = tmp_path / "b_bundle.json"
    path_a.write_text(json.dumps(_minimal_bundle("Course A", ["Intro", "Advanced"])), encoding="utf-8")
    path_b.write_text(json.dumps(_minimal_bundle("Course B", ["Intro", "Basics"])), encoding="utf-8")

    diff = diff_course_bundles(str(path_a), str(path_b))
    assert diff["deltas"]["course_title"]["a"] == "Course A"
    assert diff["deltas"]["course_title"]["b"] == "Course B"
    assert len(diff["lesson_title_changes"]) == 1
    assert diff["lesson_title_changes"][0]["lesson"] == 2
