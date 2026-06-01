from argparse import Namespace

from course_generator.checkpoints import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip(tmp_path):
    args = Namespace(
        output_dir=str(tmp_path),
        output_prefix="run-a",
        docs_path="docs",
        model="llama3",
        language="en",
    )
    outline = {
        "course_title": "T",
        "course_description": "D",
        "target_audience": "A",
        "prerequisites": [],
        "learning_outcomes": [],
        "glossary": [],
        "lessons": [{"title": "L1", "goal": "G", "key_points": ["x"]}],
    }
    lessons = [{"summary": "s", "lesson_html": "<section></section>"}]
    path = save_checkpoint(
        tmp_path / "cp.json",
        outline=outline,
        lesson_payloads=lessons,
        outline_rag_used=True,
        stage="lessons",
        args=args,
    )
    loaded = load_checkpoint(path)
    assert loaded["outline"]["course_title"] == "T"
    assert len(loaded["lesson_payloads"]) == 1
    assert loaded["outline_rag_used"] is True
