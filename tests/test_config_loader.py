import json
from argparse import Namespace
from pathlib import Path

from course_generator.config_loader import apply_config_file, load_config_file
from course_generator.cli import default_args


def test_load_config_file(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"language": "ru", "min_lessons": 3}), encoding="utf-8")
    data = load_config_file(str(path))
    assert data["language"] == "ru"


def test_apply_config_only_default_fields(tmp_path):
    defaults = default_args()
    args = Namespace(**vars(defaults))
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"language": "ru", "min_lessons": 5}), encoding="utf-8")
    apply_config_file(args, str(cfg), defaults)
    assert args.language == "ru"
    assert args.min_lessons == 5

    args2 = Namespace(**vars(defaults))
    args2.min_lessons = 99
    apply_config_file(args2, str(cfg), defaults)
    assert args2.min_lessons == 99
    assert args2.language == "ru"
