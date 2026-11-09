"""``app.py`` 界面事件测试：录入持久化、已知/未知识别、标注图与空输入。

这些测试用 ``tests/helpers.py`` 里的假编码器，不下载 ResNet18 权重，
因此可以和核心层单测一起秒级跑完；真实模型的端到端验证在
``test_integration_resnet.py`` 与 ``RUN_RESNET_TESTS=1`` 下进行。
"""

from __future__ import annotations

from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")

from helpers import BLUE, DIM, GREEN, RED, YELLOW, color_encoder, one_hot, solid
from PIL import Image

import app
from core.library import FeatureLibrary, load_library
from core.recognize import Match, Recognition

# 三种注册颜色都映射到同一类，用来模拟"同一物体多张图"。
MULTI_VIEW_ENCODER = color_encoder(
    {
        RED: one_hot(0),
        GREEN: one_hot(0),
        BLUE: one_hot(0),
        YELLOW: one_hot(3),
    }
)


def _save_solid(
    path: Path, color: tuple[int, int, int], size: tuple[int, int] = (64, 64)
) -> str:
    """把纯色图写到磁盘并返回路径字符串，模拟 Gradio 上传后的临时文件。"""
    solid(color, size).save(path)
    return str(path)


def test_enroll_three_images_increases_count_and_persists(tmp_path: Path) -> None:
    library = FeatureLibrary(path=tmp_path / "features.json", dim=DIM)
    files = [
        _save_solid(tmp_path / "apple_1.png", RED),
        _save_solid(tmp_path / "apple_2.png", GREEN),
        _save_solid(tmp_path / "apple_3.png", BLUE),
    ]

    status, rows, summary = app.enroll_images(
        "苹果", files, library=library, encoder=MULTI_VIEW_ENCODER
    )

    assert "已录入" in status and "3" in status
    assert rows[0][0] == "苹果"
    assert rows[0][1] == "3"
    assert rows[0][2] == "apple_1.png"
    assert "1 个物体 · 3 条特征" in summary

    # 关键：不是只看内存计数，而是重新从磁盘加载，模拟重启。
    reloaded = load_library(library.path)
    assert reloaded.total_vectors == 3
    record = reloaded.get("苹果")
    assert record is not None and len(record.entries) == 3


def test_enroll_duplicate_images_are_skipped(tmp_path: Path) -> None:
    library = FeatureLibrary(path=tmp_path / "features.json", dim=DIM)
    first = _save_solid(tmp_path / "same_1.png", RED)
    second = _save_solid(tmp_path / "same_2.png", RED)  # 像素完全相同，只是文件名不同

    app.enroll_images(
        "苹果", [first, second], library=library, encoder=MULTI_VIEW_ENCODER
    )
    status, rows, _ = app.enroll_images(
        "苹果", [first, second], library=library, encoder=MULTI_VIEW_ENCODER
    )

    assert "没有新增特征" in status
    # 两张图像素完全一致，第一次上传时后一张就被去重，所以该物体只有 1 条。
    assert rows[0][1] == "1"


def test_known_image_returns_name_similarity_and_annotated_image(
    tmp_path: Path,
) -> None:
    library = FeatureLibrary(path=tmp_path / "features.json", dim=DIM)
    app.enroll_images(
        "苹果",
        [_save_solid(tmp_path / "apple.png", RED)],
        library=library,
        encoder=MULTI_VIEW_ENCODER,
    )
    app.enroll_images(
        "香蕉",
        [_save_solid(tmp_path / "banana.png", YELLOW)],
        library=library,
        encoder=color_encoder({RED: one_hot(0), YELLOW: one_hot(1)}),
    )

    query = solid(RED, (48, 48))  # 与注册图尺寸不同，确保不是同一张文件
    headline, annotated, rows = app.recognize_image(
        query, library=library, encoder=MULTI_VIEW_ENCODER
    )

    assert "识别结果：苹果" in headline
    assert "100.00%" in headline
    assert "占位值" in headline
    assert isinstance(annotated, Image.Image)
    assert annotated.size == (48, 48)
    assert rows[0][1] == "苹果" and rows[0][3] == "100.00%"


def test_unenrolled_image_is_unknown(tmp_path: Path) -> None:
    library = FeatureLibrary(path=tmp_path / "features.json", dim=DIM)
    app.enroll_images(
        "苹果",
        [_save_solid(tmp_path / "apple.png", RED)],
        library=library,
        encoder=MULTI_VIEW_ENCODER,
    )

    headline, annotated, rows = app.recognize_image(
        solid(YELLOW, (48, 48)), library=library, encoder=MULTI_VIEW_ENCODER
    )

    assert "识别结果：未知物体" in headline
    assert "低于判定阈值" in headline
    assert isinstance(annotated, Image.Image)
    # 近邻表仍保留最高分候选，便于现场解释"为什么判未知"。
    assert rows and rows[0][1] == "苹果"


def test_annotate_marks_known_and_unknown_without_mutating_input() -> None:
    known = Recognition(
        is_known=True,
        name="苹果",
        similarity=0.95,
        threshold=0.60,
        matches=(Match(name="苹果", image="apple.png", similarity=0.95, rank=1),),
    )
    unknown = Recognition(
        is_known=False,
        name="未知物体",
        similarity=0.21,
        threshold=0.60,
    )

    source = solid(RED, (120, 90))
    known_image = app.annotate_image(source, known)
    unknown_image = app.annotate_image(source, unknown)

    assert source.getpixel((0, 0)) == RED  # 原图未被修改
    assert known_image.getpixel((0, 0)) != unknown_image.getpixel((0, 0))
    assert known_image.getpixel((0, 0))[1] > known_image.getpixel((0, 0))[0]  # 绿色
    assert unknown_image.getpixel((0, 0))[0] > unknown_image.getpixel((0, 0))[1]  # 红色


def test_blank_inputs_are_rejected() -> None:
    library = FeatureLibrary(path=Path("unused.json"), dim=DIM)

    status, _, _ = app.enroll_images(
        "  ", [], library=library, encoder=MULTI_VIEW_ENCODER
    )
    assert "物体名称" in status

    status, _, _ = app.enroll_images(
        "苹果", [], library=library, encoder=MULTI_VIEW_ENCODER
    )
    assert "至少上传一张" in status

    status, image, rows = app.recognize_image(None, library=library)
    assert "上传一张测试图片" in status and image is None and rows == []

    status, image, rows = app.recognize_image(solid(RED), library=library)
    assert "特征库为空" in status and image is None and rows == []


def test_build_demo_constructs_ui(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = FeatureLibrary(path=tmp_path / "features.json", dim=DIM)
    monkeypatch.setattr(app, "get_library", lambda: library)

    demo = app.build_demo()

    assert isinstance(demo, gr.Blocks)
    assert len(demo.blocks) >= 15
