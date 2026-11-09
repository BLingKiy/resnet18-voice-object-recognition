"""命令行验收工具的基础测试（不加载真实模型）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from helpers import BLUE, DIM, GREEN, RED, color_encoder, one_hot, solid

from core.library import FeatureLibrary, load_library
from tools.demo_cli import main


def _saved_library(tmp_path: Path) -> Path:
    library = FeatureLibrary(path=tmp_path / "features.json", dim=DIM)
    encoder = color_encoder({RED: one_hot(0), GREEN: one_hot(1), BLUE: one_hot(2)})
    library.add("苹果", [solid(RED), solid(GREEN)], encoder=encoder)
    library.add("香蕉", [solid(BLUE)], encoder=encoder)
    return library.save()


def test_stats_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _saved_library(tmp_path)

    code = main(["stats", "--library", str(path), "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["object_count"] == 2
    assert payload["total_vectors"] == 3
    assert payload["dim"] == DIM
    assert len(payload["objects"]) == 2


def test_threshold_set_and_read(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _saved_library(tmp_path)

    assert main(["threshold", "--library", str(path), "--set", "0.63", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["threshold"] == pytest.approx(0.63)
    assert payload["saved"] is True
    assert load_library(path).threshold == pytest.approx(0.63)

    assert main(["threshold", "--library", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["effective_threshold"] == pytest.approx(0.63)
    assert payload["saved"] is False


def test_missing_library_reports_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "none.json"

    code = main(["stats", "--library", str(missing)])

    assert code == 2
    captured = capsys.readouterr()
    assert "特征库不存在" in captured.err


def test_add_empty_directory_reports_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    code = main(
        ["add", "--name", "苹果", "--dir", str(empty), "--library", str(tmp_path / "features.json")]
    )

    assert code == 2
    assert "没有图片" in capsys.readouterr().err


def test_recognize_missing_image_reports_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "recognize",
            "--image",
            str(tmp_path / "missing.jpg"),
            "--library",
            str(tmp_path / "features.json"),
        ]
    )

    assert code == 2
    assert "图片不存在" in capsys.readouterr().err


def test_eval_missing_directory_reports_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        ["eval", "--dir", str(tmp_path / "missing"), "--library", str(tmp_path / "features.json")]
    )

    assert code == 2
    assert "测试目录不存在" in capsys.readouterr().err


def test_threshold_rejects_out_of_range(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _saved_library(tmp_path)

    code = main(["threshold", "--library", str(path), "--set", "1.5"])

    assert code == 2
    assert "阈值" in capsys.readouterr().err
    assert load_library(path).threshold is None
