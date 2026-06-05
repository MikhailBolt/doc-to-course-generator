import json

from course_generator.checkpoints import find_recent_checkpoints


def test_find_recent_checkpoints(tmp_path):
    cp_dir = tmp_path / ".checkpoints" / "run1"
    cp_dir.mkdir(parents=True)
    cp = cp_dir / "checkpoint.json"
    cp.write_text(
        json.dumps({"stage": "lessons", "lessons_completed": 2, "model": "llama3", "outline": {}}),
        encoding="utf-8",
    )
    rows = find_recent_checkpoints(str(tmp_path), limit=3)
    assert len(rows) == 1
    assert rows[0]["lessons_completed"] == 2
