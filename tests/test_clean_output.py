from course_generator.clean_output import clean_output_dir, plan_clean_output


def test_plan_clean_keeps_newest(tmp_path):
    older = tmp_path / "old_course_bundle.json"
    newer = tmp_path / "course_bundle.json"
    older.write_text("{}", encoding="utf-8")
    import time

    time.sleep(0.02)
    newer.write_text("{}", encoding="utf-8")
    plan = plan_clean_output(str(tmp_path), keep_last=1)
    assert plan["delete_count"] == 1
    assert plan["keep_count"] == 1
    assert any(p.endswith("course_bundle.json") and "old_" not in p for p in plan["keep"])


def test_clean_output_dry_run(tmp_path):
    path = tmp_path / "generation_report.json"
    path.write_text("{}", encoding="utf-8")
    plan = clean_output_dir(str(tmp_path), keep_last=0, dry_run=True)
    assert plan["dry_run"] is True
    assert path.exists()
    assert plan["delete_count"] == 1


def test_clean_output_deletes(tmp_path):
    path = tmp_path / "course.html"
    path.write_text("<html></html>", encoding="utf-8")
    keep = tmp_path / "notes.txt"
    keep.write_text("keep", encoding="utf-8")
    plan = clean_output_dir(str(tmp_path), keep_last=0, dry_run=False)
    assert not path.exists()
    assert keep.exists()
    assert plan["ok"] is True
