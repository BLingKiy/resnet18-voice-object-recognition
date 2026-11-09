"""``tools/realtime_check.py`` 的离线测试（不依赖摄像头 / opencv）。

用假编码器造出"背景 / 已知物体 / 未知物体"三组可分离的分布，
验证报告能给出 T_bg/T_rt 建议，并且写回只动 realtime.json。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from helpers import color_encoder, one_hot, solid

from core.library import FeatureLibrary
from core.realtime_config import RealtimeConfig
from tools.realtime_check import (
    discover_frames,
    evaluate_frames,
    format_report,
    recommend_thresholds,
)

GRAY = (128, 128, 128)
RED = (220, 40, 40)
BLUE = (40, 60, 220)

ENCODER = color_encoder(
    {
        GRAY: one_hot(0),
        RED: one_hot(1),
        BLUE: one_hot(2),
    }
)


def _write_images(directory: Path, color: tuple[int, int, int], count: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        solid(color, (48, 48)).save(directory / f"{index:02d}.png")


def test_discover_frames_splits_background_known_and_unknown(tmp_path: Path) -> None:
    _write_images(tmp_path / "background", GRAY, 3)
    _write_images(tmp_path / "mouse", RED, 2)
    _write_images(tmp_path / "unknown" / "剪刀", BLUE, 2)

    frame_set = discover_frames(tmp_path)

    assert len(frame_set.background) == 3
    assert list(frame_set.known) == ["mouse"] and len(frame_set.known["mouse"]) == 2
    assert list(frame_set.unknown) == ["剪刀"] and len(frame_set.unknown["剪刀"]) == 2


def test_discover_frames_flat_root_is_background(tmp_path: Path) -> None:
    _write_images(tmp_path, GRAY, 4)

    frame_set = discover_frames(tmp_path)

    assert len(frame_set.background) == 4
    assert frame_set.warnings  # 明确提示"把散图当背景"


def test_evaluate_frames_recommends_separable_thresholds(tmp_path: Path) -> None:
    _write_images(tmp_path / "background", GRAY, 6)
    _write_images(tmp_path / "mouse", RED, 2)
    _write_images(tmp_path / "unknown" / "剪刀", BLUE, 2)

    library = FeatureLibrary(dim=8)
    library.add("mouse", [solid(RED)], encoder=ENCODER)
    config = RealtimeConfig(
        threshold_static=0.6,
        threshold_realtime=0.6,
        threshold_bg=0.9,
        gate_enabled=True,
        crop_margin=0.05,
    )
    frame_set = discover_frames(tmp_path)

    report = evaluate_frames(
        frame_set, library, config, ENCODER, take=2, drop_first=1
    )

    assert report["background"]["frames"] == 6
    assert report["background"]["learned"] == 2
    assert report["background"]["bg_sim"]["max"] == pytest.approx(1.0, abs=1e-6)
    assert report["known"]["mouse"]["effective"]["min"] == pytest.approx(1.0, abs=1e-6)
    assert report["unknown"]["剪刀"]["effective"]["max"] == pytest.approx(0.0, abs=1e-6)
    assert report["known"]["mouse"]["correct"] == 2

    thresholds = report["thresholds"]
    assert thresholds["static"] == pytest.approx(0.6)
    assert thresholds["bg_recommended"] == pytest.approx(0.5, abs=1e-4)
    assert thresholds["realtime_recommended"] == pytest.approx(0.5, abs=1e-4)
    assert thresholds["reasonable"]["overall"] is True

    text = format_report(report)
    assert "背景相似度基线" in text
    assert "空场景最高分" in text
    assert "T_bg" in text and "T_rt" in text


def test_recommend_thresholds_without_unknown_does_not_claim_verified() -> None:
    background = {
        "bg_sim": {"n": 3, "min": 0.9, "mean": 0.95, "max": 1.0},
        "bg_sim_holdout": {"n": 0, "min": None, "mean": None, "max": None},
        "false_positive": {"threshold": 0.6, "max": 0.2, "count": 0},
    }
    known = {
        "mouse": {
            "bg_sim": {"n": 2, "min": 0.1, "mean": 0.1, "max": 0.1},
            "effective": {"n": 2, "min": 0.9, "mean": 0.92, "max": 0.94},
        }
    }
    library = FeatureLibrary(dim=8)
    library.set_threshold(0.6)
    config = RealtimeConfig(0.6, 0.6, 0.9, True, 0.05)

    thresholds = recommend_thresholds(
        background=background, known=known, unknown={}, config=config, library=library
    )

    assert thresholds["bg_recommended"] == pytest.approx(0.5, abs=1e-4)
    assert thresholds["realtime_recommended"] == pytest.approx(0.87, abs=1e-4)
    assert thresholds["reasonable"]["unknown_checked"] is False
    assert thresholds["reasonable"]["overall"] is False


def test_cli_frames_writes_only_realtime_config(tmp_path: Path, capsys) -> None:
    from tools.realtime_check import main

    _write_images(tmp_path / "frames" / "background", GRAY, 6)
    _write_images(tmp_path / "frames" / "mouse", RED, 2)
    _write_images(tmp_path / "frames" / "unknown" / "剪刀", BLUE, 2)

    library = FeatureLibrary(path=tmp_path / "features.json", dim=8)
    library.add("mouse", [solid(RED)], encoder=ENCODER)
    library.save()
    library_digest = library.path.read_bytes()

    config_path = tmp_path / "realtime.json"
    from core.realtime_config import save_realtime_config

    save_realtime_config(RealtimeConfig(0.6, 0.6, 0.9, True, 0.05), config_path)

    # 没有真实 ResNet 权重时 main 会尝试加载默认编码器；这条用例只验证
    # 参数/写盘路径，所以 monkeypatch 掉编码器。
    import tools.realtime_check as tool

    original = tool.get_default_encoder
    tool.get_default_encoder = lambda: ENCODER
    try:
        code = main(
            [
                "--frames",
                str(tmp_path / "frames"),
                "--library",
                str(library.path),
                "--config",
                str(config_path),
                "--take",
                "2",
                "--drop-first",
                "1",
                "--write",
                "--json",
            ]
        )
    finally:
        tool.get_default_encoder = original

    assert code == 0
    assert library.path.read_bytes() == library_digest
    written = json.loads(config_path.read_text(encoding="utf-8"))
    assert written["schema"] == "rn18-realtime-config"
    assert written["threshold_static"] == pytest.approx(0.6)
    assert written["threshold_bg"] == pytest.approx(0.5)
    assert written["threshold_realtime"] == pytest.approx(0.5)
    assert "已写入" in capsys.readouterr().err
