import json
from argparse import Namespace

import pytest

from course_generator.batch import list_batch_configs


def test_list_batch_configs(tmp_path):
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    (tmp_path / "b.json").write_text("{}", encoding="utf-8")
    configs = list_batch_configs(str(tmp_path))
    assert len(configs) == 2


def test_list_batch_configs_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_batch_configs(str(tmp_path))
