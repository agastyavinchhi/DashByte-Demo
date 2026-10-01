from __future__ import annotations

import pytest

from pipeline import paths
from pipeline.config import load_config

pytestmark = pytest.mark.unit

HELPERS = [
    (paths.raw_dir, "raw"),
    (paths.features_dir, "features"),
    (paths.models_dir, "models"),
    (paths.predictions_dir, "predictions"),
    (paths.quality_dir, "quality"),
]


@pytest.mark.parametrize("helper,name", HELPERS)
def test_helper_returns_path_and_creates_nothing(tmp_data_dir, helper, name):
    cfg = load_config()
    assert helper(cfg) == tmp_data_dir / name
    assert not tmp_data_dir.exists()


def test_ensure_data_dirs_creates_then_reports_exists(tmp_data_dir):
    cfg = load_config()

    first = paths.ensure_data_dirs(cfg)
    assert set(first) == set(paths.DATA_DIR_NAMES)
    for name, (path, created) in first.items():
        assert path == tmp_data_dir / name
        assert path.is_dir()
        assert created is True

    second = paths.ensure_data_dirs(cfg)
    assert all(created is False for _, created in second.values())


def test_ensure_data_dirs_only_creates_missing(tmp_data_dir):
    cfg = load_config()
    paths.models_dir(cfg).mkdir(parents=True)

    result = paths.ensure_data_dirs(cfg)
    assert result["models"][1] is False
    assert all(result[n][1] for n in paths.DATA_DIR_NAMES if n != "models")


def test_ensure_data_dirs_fails_loudly_when_name_is_a_file(tmp_data_dir):
    cfg = load_config()
    tmp_data_dir.mkdir()
    paths.raw_dir(cfg).write_text("not a dir")

    with pytest.raises(FileExistsError):
        paths.ensure_data_dirs(cfg)


def test_ensure_data_dirs_matches_helpers(tmp_data_dir):
    cfg = load_config()
    result = paths.ensure_data_dirs(cfg)
    assert {name: path for name, (path, _) in result.items()} == {
        name: helper(cfg) for helper, name in HELPERS
    }
