"""Tab2「识别 + 播报」的界面行为测试。

关注三件事：

1. 播报文本来自**当次**识别结果（已知念名称、未知念「未知物体」）；
2. 每次点击都重新合成一次，静态链路不做去重；
3. 合成失败只显示统一文案，识别结果与标注图照常返回。

edge-tts 全部被替换成假实现，测试不联网。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

gr = pytest.importorskip("gradio")

from helpers import RED, YELLOW, DIM, color_encoder, one_hot, solid
from PIL import Image

import app
from core.library import FeatureLibrary
from core.recognize import Match, Recognition
from voice.tts import (
    TTS_UNAVAILABLE_MESSAGE,
    TtsResult,
    TtsSynthesisError,
)

ENCODER = color_encoder({RED: one_hot(0), YELLOW: one_hot(1)})


def _library_with_apple(tmp_path: Path) -> FeatureLibrary:
    """一个只录了红色「苹果」的特征库。"""
    library = FeatureLibrary(path=tmp_path / "features.json", dim=DIM)
    apple = tmp_path / "apple.png"
    solid(RED).save(apple)
    app.enroll_images("苹果", [str(apple)], library=library, encoder=ENCODER)
    return library


def _fake_result(text: str, tmp_path: Path) -> TtsResult:
    return TtsResult(
        text=text,
        audio_path=tmp_path / "tts-fake.mp3",
        voice="zh-CN-XiaoxiaoNeural",
        duration_ms=1234,
        elapsed_ms=456,
        bytes_written=7404,
    )


def test_announcement_text_uses_current_result() -> None:
    known = Recognition(
        is_known=True,
        name="苹果",
        similarity=0.93,
        threshold=0.60,
        matches=(Match(name="苹果", image="a.png", similarity=0.93, rank=1),),
    )
    unknown = Recognition(
        is_known=False,
        name="未知物体",
        similarity=0.31,
        threshold=0.60,
    )

    assert app.announcement_text(known) == "识别结果：苹果"
    assert app.announcement_text(unknown) == "未知物体"


def test_recognize_and_speak_announces_known_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = _library_with_apple(tmp_path)
    spoken: list[str] = []

    def fake_synthesize(text: str, **kwargs: Any) -> TtsResult:
        spoken.append(text)
        return _fake_result(text, tmp_path)

    monkeypatch.setattr(app, "synthesize", fake_synthesize)

    headline, annotated, rows, status, audio = app.recognize_and_speak(
        solid(RED, (48, 48)), library=library, encoder=ENCODER
    )

    assert "识别结果：苹果" in headline
    assert isinstance(annotated, Image.Image) and rows
    assert spoken == ["识别结果：苹果"]
    assert "已播报：识别结果：苹果" in status
    assert "456 ms" in status and "1.23 s" in status
    assert audio == str(tmp_path / "tts-fake.mp3")


def test_recognize_and_speak_announces_unknown_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = _library_with_apple(tmp_path)
    spoken: list[str] = []

    def fake_synthesize(text: str, **kwargs: Any) -> TtsResult:
        spoken.append(text)
        return _fake_result(text, tmp_path)

    monkeypatch.setattr(app, "synthesize", fake_synthesize)

    headline, _, _, status, _ = app.recognize_and_speak(
        solid(YELLOW, (48, 48)), library=library, encoder=ENCODER
    )

    assert "识别结果：未知物体" in headline
    assert spoken == ["未知物体"]
    assert "已播报：未知物体" in status


def test_static_link_synthesizes_on_every_click(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = _library_with_apple(tmp_path)
    calls: list[str] = []

    def fake_synthesize(text: str, **kwargs: Any) -> TtsResult:
        calls.append(text)
        return _fake_result(text, tmp_path)

    monkeypatch.setattr(app, "synthesize", fake_synthesize)

    image = solid(RED, (48, 48))
    app.recognize_and_speak(image, library=library, encoder=ENCODER)
    app.recognize_and_speak(image, library=library, encoder=ENCODER)

    # 同一个物体连点两次也要念两次：去重只属于实时链路。
    assert calls == ["识别结果：苹果", "识别结果：苹果"]


@pytest.mark.parametrize("error_cls", [TtsSynthesisError, RuntimeError])
def test_tts_failure_shows_copy_and_keeps_recognition_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_cls: type[Exception],
) -> None:
    library = _library_with_apple(tmp_path)

    def boom(text: str, **kwargs: Any) -> TtsResult:
        raise error_cls("模拟合成失败")

    monkeypatch.setattr(app, "synthesize", boom)

    headline, annotated, rows, status, audio = app.recognize_and_speak(
        solid(RED, (48, 48)), library=library, encoder=ENCODER
    )

    # 识别结果完整返回，播报失败只体现在状态行与播放器上。
    assert "识别结果：苹果" in headline
    assert isinstance(annotated, Image.Image)
    assert rows and rows[0][1] == "苹果"
    assert status == f"⚠️ {TTS_UNAVAILABLE_MESSAGE}"
    assert "语音播报不可用，请检查网络" in status
    assert audio is None


def test_no_image_does_not_speak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = _library_with_apple(tmp_path)
    monkeypatch.setattr(
        app, "synthesize", lambda *a, **k: pytest.fail("没有图片时不应播报")
    )

    headline, annotated, rows, status, audio = app.recognize_and_speak(
        None, library=library, encoder=ENCODER
    )

    assert "请先上传一张测试图片" in headline
    assert annotated is None and rows == []
    assert status == "" and audio is None


def test_empty_library_does_not_speak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = FeatureLibrary(path=tmp_path / "features.json", dim=DIM)
    monkeypatch.setattr(
        app, "synthesize", lambda *a, **k: pytest.fail("空库时不应播报")
    )

    headline, _, _, status, audio = app.recognize_and_speak(
        solid(RED), library=library, encoder=ENCODER
    )

    assert "特征库为空" in headline
    assert status == "" and audio is None


def test_tab2_has_autoplay_audio_player(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = FeatureLibrary(path=tmp_path / "features.json", dim=DIM)
    monkeypatch.setattr(app, "get_library", lambda: library)

    demo = app.build_demo()

    players = [
        block
        for block in demo.blocks.values()
        if isinstance(block, gr.Audio) and getattr(block, "autoplay", False)
    ]
    assert len(players) == 1
    hints = [
        str(block.value)
        for block in demo.blocks.values()
        if isinstance(block, gr.Markdown) and block.value
    ]
    assert any("语音播报" in hint for hint in hints)


def test_speak_result_reports_synthesis_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = Recognition(
        is_known=True,
        name="鼠标",
        similarity=0.88,
        threshold=0.60,
    )
    monkeypatch.setattr(
        app, "synthesize", lambda text, **kwargs: _fake_result(text, tmp_path)
    )

    status, audio = app.speak_result(result)

    assert "已播报：识别结果：鼠标" in status
    assert "456 ms" in status and "音色" in status
    assert audio == str(tmp_path / "tts-fake.mp3")
