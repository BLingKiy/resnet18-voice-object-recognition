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
from voice import AsrAudioError, AsrDependencyError, AsrNotConfigured, BaiduAsrError

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


# --------------------------------------------------------------------------- #
# P3 语音录名：界面的降级、回填与键盘路径回归（asr 全部 mock，不联网）
# --------------------------------------------------------------------------- #
def _find_button(demo: gr.Blocks, label: str) -> gr.Button:
    """按显示文案在 Blocks 里找到按钮组件。"""
    for block in demo.blocks.values():
        if isinstance(block, gr.Button) and block.value == label:
            return block
    raise AssertionError(f"界面里找不到按钮：{label}")


def test_asr_button_is_disabled_and_hint_shown_without_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = FeatureLibrary(path=tmp_path / "features.json", dim=DIM)
    monkeypatch.setattr(app, "get_library", lambda: library)
    monkeypatch.setattr(app, "is_asr_configured", lambda: False)

    demo = app.build_demo()

    assert _find_button(demo, "语音识别名称").interactive is False
    audios = [b for b in demo.blocks.values() if isinstance(b, gr.Audio)]
    # Tab1 的麦克风录音是唯一一个带 microphone 来源的音频组件；
    # Tab2 的播报播放器只用于放音（见 test_app_tts.py）。
    assert any("microphone" in audio.sources for audio in audios)
    hints = [
        str(b.value)
        for b in demo.blocks.values()
        if isinstance(b, gr.Markdown) and b.value
    ]
    assert any("未配置语音识别密钥" in hint and "键盘" in hint for hint in hints)


def test_asr_button_is_enabled_when_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = FeatureLibrary(path=tmp_path / "features.json", dim=DIM)
    monkeypatch.setattr(app, "get_library", lambda: library)
    monkeypatch.setattr(app, "is_asr_configured", lambda: True)

    demo = app.build_demo()

    assert _find_button(demo, "语音识别名称").interactive is True
    hints = [
        str(b.value)
        for b in demo.blocks.values()
        if isinstance(b, gr.Markdown) and b.value
    ]
    assert any("语音识别名称" in hint for hint in hints)


def test_voice_click_without_keys_keeps_name_and_never_calls_asr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    def fake_asr(path: object) -> str:
        calls.append(path)
        return "鼠标。"

    monkeypatch.setattr(app, "is_asr_configured", lambda: False)
    monkeypatch.setattr(app, "asr", fake_asr)

    name, status, button = app.transcribe_name("record.wav", "苹果")

    # 降级分支：不覆盖已输入名称、不调用 asr、按钮更新为置灰。
    assert name == "苹果"
    assert "未配置语音识别密钥" in status and "键盘" in status
    assert button["interactive"] is False
    assert calls == []


def test_voice_click_fills_recognized_text_into_name_box(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_asr(path: object) -> str:
        seen["path"] = path
        return "  鼠标。 "

    monkeypatch.setattr(app, "is_asr_configured", lambda: True)
    monkeypatch.setattr(app, "asr", fake_asr)

    name, status, button = app.transcribe_name("record.wav", "苹果")

    assert name == "鼠标。"  # 首尾空白被去掉，用户可以继续编辑
    assert "已识别为「鼠标。」" in status
    assert button["interactive"] is True
    assert seen["path"] == "record.wav"


def test_voice_click_without_recording_does_not_call_asr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app, "is_asr_configured", lambda: True)
    monkeypatch.setattr(app, "asr", lambda path: pytest.fail("没有录音时不应调用 asr"))

    name, status, _ = app.transcribe_name(None, "苹果")

    assert name == "苹果"
    assert "请先" in status


def test_late_asr_not_configured_falls_back_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """启动后 Key 被移走：is_asr_configured 仍为 True，但 asr 抛未配置异常。"""

    def boom(path: object) -> str:
        raise AsrNotConfigured("Key 在运行期被移除")

    monkeypatch.setattr(app, "is_asr_configured", lambda: True)
    monkeypatch.setattr(app, "asr", boom)

    name, status, button = app.transcribe_name("record.wav", "苹果")

    assert name == "苹果"
    assert "未配置语音识别密钥" in status
    assert button["interactive"] is False


@pytest.mark.parametrize("error_cls", [AsrDependencyError, AsrAudioError, BaiduAsrError])
def test_other_asr_errors_become_ui_messages(
    error_cls: type[Exception], monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(path: object) -> str:
        raise error_cls("模拟错误")

    monkeypatch.setattr(app, "is_asr_configured", lambda: True)
    monkeypatch.setattr(app, "asr", boom)

    name, status, button = app.transcribe_name("record.wav", "苹果")

    assert name == "苹果"  # 失败不清空键盘输入
    assert "模拟错误" in status
    assert button["interactive"] is True


def test_silent_recognition_keeps_typed_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app, "is_asr_configured", lambda: True)
    monkeypatch.setattr(app, "asr", lambda path: "   ")

    name, status, _ = app.transcribe_name("record.wav", "苹果")

    assert name == "苹果"
    assert "没有听清" in status


def test_recognized_name_can_be_enrolled_through_keyboard_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """语音回填名称 -> 用户确认 -> 原有 enroll_images 入库，条数按张数增加。"""
    monkeypatch.setattr(app, "is_asr_configured", lambda: True)
    monkeypatch.setattr(app, "asr", lambda path: "鼠标。")

    name, _, _ = app.transcribe_name("record.wav", "")
    library = FeatureLibrary(path=tmp_path / "features.json", dim=DIM)
    files = [
        _save_solid(tmp_path / "mouse_1.png", RED),
        _save_solid(tmp_path / "mouse_2.png", GREEN),
    ]

    status, rows, _ = app.enroll_images(
        name, files, library=library, encoder=MULTI_VIEW_ENCODER
    )

    assert "已录入" in status and "2" in status
    assert rows[0][0] == "鼠标。"
    assert rows[0][1] == "2"
    assert load_library(library.path).get("鼠标。") is not None
