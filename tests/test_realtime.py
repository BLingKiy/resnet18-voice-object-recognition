"""Tab3 实时识别 + 教学式录入的状态机测试（不连摄像头、不联网）。

覆盖任务五条验收判据中可离线验证的部分：

1. 持续识别：帧 -> 名称/相似度/标注图，TTS 由 ``SpeechGate`` 节流；
2. 教学闭环：抓 5 帧 -> ASR 回填 -> 确认 -> 库 +5 -> 同物体再识别并播报；
3. 防回声：教学态 ``teaching``/``recognition_paused``/``tts_paused`` 同时为真，
   退出后 ``SpeechGate.reset()`` 让冷却清零；
4. 键盘兜底：不调用 ASR、直接提交键盘名称同样入库；
5. 隔离：坏帧只影响本帧；取消教学不写库；写盘失败回滚内存。

假编码器按左上角像素颜色查表，与 ``tests/helpers.py`` 一致，不加载 ResNet18。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

gr = pytest.importorskip("gradio")

from helpers import DIM, color_encoder, one_hot, solid

import app
from core.library import FeatureLibrary, load_library
from core.pipeline import BACKGROUND_LABEL
from core.realtime_config import RealtimeConfig
from voice import SpeechGate, TtsSynthesisError

APPLE = (220, 40, 40)
BANANA = (240, 210, 40)
CUP = (40, 60, 220)
MOUSE = (128, 0, 128)
WHITE = (250, 250, 250)
BLACK = (5, 5, 5)
CYAN = (0, 255, 255)
ORANGE = (255, 140, 0)
PINK = (255, 105, 180)
GRAY = (128, 128, 128)
TEAL = (0, 128, 128)

# 教学帧：不同颜色 -> 同一个新物体向量，模拟同一物体的多个视角。
# 颜色必须互不相同，否则 FeatureLibrary 会按像素摘要去重，凑不满 5 条。
TEACH_ENCODER = color_encoder(
    {
        APPLE: one_hot(0),
        BANANA: one_hot(1),
        CUP: one_hot(2),
        MOUSE: one_hot(4),
        WHITE: one_hot(4),
        CYAN: one_hot(4),
        ORANGE: one_hot(4),
        BLACK: one_hot(4),
        PINK: one_hot(4),
        GRAY: one_hot(4),
        TEAL: one_hot(4),
    }
)


class FakeClock:
    """可手动推进的单调时钟，用来精确验证 0.2 s 抓帧间隔与 4 s 冷却。"""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def library(tmp_path: Path) -> FeatureLibrary:
    """已有「苹果」的库；教学新增物体从 1 个物体 / 1 条特征开始。"""
    lib = FeatureLibrary(path=tmp_path / "features.json", dim=DIM)
    lib.add("苹果", [solid(APPLE)], encoder=TEACH_ENCODER)
    lib.save()
    return lib


def make_controller(
    library: FeatureLibrary,
    *,
    spoken: list[str] | None = None,
    clock: FakeClock | None = None,
) -> app.RealtimeController:
    """构造控制器并注入假时钟 / 同步假播报（不联网）。"""
    fake_clock = clock or FakeClock()
    gate = SpeechGate(4.0, clock=fake_clock)
    return app.RealtimeController(
        library,
        encoder=TEACH_ENCODER,
        gate=gate,
        clock=fake_clock,
    )


def make_background_controller(
    library: FeatureLibrary,
    tmp_path: Path,
    *,
    clock: FakeClock | None = None,
    learn_frames: int = 12,
    drop_first: int = 2,
    gate_enabled: bool = True,
) -> app.RealtimeController:
    """构造带背景学习的控制器：背景文件放 tmp，阈值显式注入。"""
    fake_clock = clock or FakeClock()
    return app.RealtimeController(
        library,
        encoder=TEACH_ENCODER,
        gate=SpeechGate(4.0, clock=fake_clock),
        clock=fake_clock,
        background_path=tmp_path / "background.json",
        config=RealtimeConfig(
            threshold_static=0.6,
            threshold_realtime=0.6,
            threshold_bg=0.9,
            bg_crop_margin=0.05,
            bg_gate_enabled=gate_enabled,
        ),
        learn_frames=learn_frames,
        drop_first=drop_first,
    )


def _learn_background(
    controller: app.RealtimeController,
    color: tuple[int, int, int] = APPLE,
) -> None:
    """喂够 learn_frames 帧，让控制器学完背景（前 drop_first 帧只做预热）。"""
    needed = controller.learn_frames
    for _ in range(needed):
        controller.process(solid(color))
    assert controller.background_ready is True


def _fake_speaker(spoken: list[str], tmp_path: Path):
    """替换 ``app.speak_in_background``：同步记录文本并回调成功。"""

    def fake(text: str, *, on_success: Any = None, on_error: Any = None, **_: Any):
        spoken.append(text)
        path = tmp_path / f"tts_{len(spoken)}.mp3"
        path.write_bytes(b"ID3fake")
        if on_success is not None:
            on_success(path)
        return threading.Thread()

    return fake


def _capture_five(controller: app.RealtimeController, clock: FakeClock) -> None:
    """先丢 1 个预热帧，再按 0.25 s 间隔抓满 5 帧。

    触发教学的当前帧已经算第 1 个预热帧，所以这里再丢 1 帧即可开始计数。
    颜色必须互不相同且亮度可用，否则会被去重或画质检查跳过。
    """
    for color in (WHITE, CYAN, ORANGE, PINK, GRAY, TEAL):
        if controller.captured_count >= controller.count:
            break
        clock.advance(0.25)
        controller.offer_frame(solid(color))
    assert controller.captured_count == controller.count


def test_continuous_recognition_overlays_and_throttles_tts(
    library: FeatureLibrary, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spoken: list[str] = []
    monkeypatch.setattr(app, "speak_in_background", _fake_speaker(spoken, tmp_path))
    clock = FakeClock()
    # 持续识别需要多个已知物体才能验证"结果变化"。
    library.add("香蕉", [solid(BANANA)], encoder=TEACH_ENCODER)
    library.add("杯子", [solid(CUP)], encoder=TEACH_ENCODER)
    controller = make_controller(library, clock=clock)

    first = controller.process(solid(APPLE, (64, 48)))
    assert first.frame is not None and first.frame.shape == (48, 64, 3)
    assert "识别结果：苹果" in first.headline
    assert first.gate_spoken is True and spoken == ["识别结果：苹果"]

    # 帧流节流：距上一帧不足 0.3 s 的帧不进入 recognize_stream。
    clock.advance(0.1)
    skipped = controller.process(solid(BANANA))
    assert skipped.frame is None and skipped.gate_spoken is False

    # 同一结果：不重复播报。
    clock.advance(0.4)
    same = controller.process(solid(APPLE))
    assert same.gate_spoken is False and spoken == ["识别结果：苹果"]

    # 结果变了，但冷却未到：压住不播。
    clock.advance(0.5)
    changed = controller.process(solid(BANANA))
    assert "香蕉" in changed.headline
    assert changed.gate_spoken is False and spoken == ["识别结果：苹果"]

    # 冷却到：换结果才播。
    clock.advance(3.5)
    later = controller.process(solid(BANANA))
    assert later.gate_spoken is True and spoken[-1] == "识别结果：香蕉"

    # 冷却早过了，但结果没变：仍然不播。
    clock.advance(10.0)
    assert controller.process(solid(BANANA)).gate_spoken is False
    assert spoken == ["识别结果：苹果", "识别结果：香蕉"]


def test_teaching_pauses_recognition_and_captures_spaced_frames(
    library: FeatureLibrary, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spoken: list[str] = []
    monkeypatch.setattr(app, "speak_in_background", _fake_speaker(spoken, tmp_path))
    clock = FakeClock()
    controller = make_controller(library, clock=clock)

    controller.process(solid(APPLE))  # 先正常播一次
    assert spoken

    ok, message = controller.begin_teaching(solid(APPLE))
    assert ok and "教学态" in message
    assert controller.teaching is True
    # 防回声的三个状态必须同时成立（任务第三节）。
    assert controller.recognition_paused is True
    assert controller.tts_paused is True
    assert controller.audio_path is None  # 待播音频被清空，旧声音不会再响

    # 当前帧只做预热，不计入 5 张注册帧。
    assert controller.captured_count == 0
    assert controller._warmup_remaining == 1  # 明确验证预热状态

    # 教学态：帧只进抓帧缓冲，不识别、不播报、不覆盖画面。
    clock.advance(0.1)
    paused = controller.process(solid(BANANA))
    assert paused.frame is None and paused.gate_spoken is False
    assert controller.captured_count == 0  # 这帧仍是第 2 个预热帧，被丢弃
    assert spoken == ["识别结果：苹果"]

    for color in (WHITE, CYAN, ORANGE, PINK, GRAY):
        clock.advance(0.25)
        controller.process(solid(color))
    assert controller.captured_count == 5
    # 任务要求 0.2–0.3 s；这里用 0.25 s，且不是 5 个相邻帧。
    assert all(0.2 <= gap <= 0.3 for gap in controller.capture_intervals)
    assert controller.teaching is True and spoken == ["识别结果：苹果"]


def test_commit_adds_five_vectors_and_immediate_recognition(
    library: FeatureLibrary, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spoken: list[str] = []
    monkeypatch.setattr(app, "speak_in_background", _fake_speaker(spoken, tmp_path))
    clock = FakeClock()
    controller = make_controller(library, clock=clock)

    before_objects, before_vectors = library.object_count, library.total_vectors
    assert (before_objects, before_vectors) == (1, 1)

    controller.begin_teaching(solid(APPLE))
    _capture_five(controller, clock)
    status, summary = controller.commit_teaching("鼠标")

    assert "新增 **5** 条特征" in status
    assert f"{before_vectors} → **{before_vectors + 5}** 条" in status
    assert "1 → 2 个物体" in status
    assert library.object_count == 2 and library.total_vectors == 6
    assert controller.teaching is False
    assert controller.recognition_paused is False and controller.tts_paused is False

    # 重启后仍在：重新从磁盘加载。
    reloaded = load_library(library.path)
    record = reloaded.get("鼠标")
    assert record is not None and len(record.entries) == 5
    assert all(entry.image.startswith("teach-鼠标-") for entry in record.entries)

    # 同一物体立刻被识别，并且冷却已清零 -> 马上播报。
    after = controller.process(solid(BLACK))
    assert after.frame is not None
    assert "识别结果：鼠标" in after.headline
    assert after.gate_spoken is True
    assert spoken[-1] == "识别结果：鼠标"


def test_keyboard_fallback_commits_without_asr(
    library: FeatureLibrary, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app, "speak_in_background", _fake_speaker([], tmp_path))
    monkeypatch.setattr(
        app, "asr", lambda path: pytest.fail("键盘兜底不应调用 ASR")
    )
    clock = FakeClock()
    controller = make_controller(library, clock=clock)

    controller.begin_teaching(solid(APPLE))
    _capture_five(controller, clock)
    status, _ = controller.commit_teaching("键盘鼠标")

    assert "已入库「键盘鼠标」" in status
    assert library.get("键盘鼠标") is not None
    assert len(library.get("键盘鼠标").entries) == 5


def test_asr_only_runs_inside_teaching_and_fills_name(
    library: FeatureLibrary, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []

    def fake_asr(path: object) -> str:
        calls.append(path)
        return "  鼠标。 "

    monkeypatch.setattr(app, "is_asr_configured", lambda: True)
    monkeypatch.setattr(app, "asr", fake_asr)
    monkeypatch.setattr(app, "speak_in_background", _fake_speaker([], tmp_path))
    controller = make_controller(library)

    controller.process(solid(APPLE))
    assert calls == []  # 持续识别不调用 ASR

    controller.begin_teaching(solid(APPLE))
    name, status, _ = controller.transcribe("teach.wav", "")
    assert name == "鼠标。"
    assert "已回填「鼠标。」" in status
    assert calls == ["teach.wav"]


def test_cancel_discards_frames_and_never_writes(
    library: FeatureLibrary, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app, "speak_in_background", _fake_speaker([], tmp_path))
    clock = FakeClock()
    controller = make_controller(library, clock=clock)
    before_bytes = library.path.read_bytes()
    before_vectors = library.total_vectors

    controller.begin_teaching(solid(APPLE))
    clock.advance(0.3)
    controller.offer_frame(solid(WHITE))  # 第 2 个预热帧，丢弃
    clock.advance(0.3)
    controller.offer_frame(solid(CYAN))  # 正式第 1 帧
    clock.advance(0.3)
    controller.offer_frame(solid(ORANGE))  # 正式第 2 帧
    assert controller.captured_count == 2

    status, _, _ = controller.cancel_teaching()

    assert "已取消教学" in status and "丢弃 2 帧" in status
    assert controller.teaching is False
    assert library.total_vectors == before_vectors
    assert library.path.read_bytes() == before_bytes
    assert controller.captured_count == 0


def test_teaching_discards_warmup_frames_before_capture(
    library: FeatureLibrary, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """首帧常黑/过曝：当前帧 + 下一帧必须丢弃，不能算进 5 张注册图。"""
    monkeypatch.setattr(app, "speak_in_background", _fake_speaker([], tmp_path))
    clock = FakeClock()
    controller = make_controller(library, clock=clock)

    ok, _ = controller.begin_teaching(solid(APPLE))
    assert ok is True
    assert controller.captured_count == 0  # 当前帧只做预热

    clock.advance(0.1)
    assert controller.offer_frame(solid(WHITE)) is False  # 第 2 个预热帧
    assert controller.captured_count == 0

    clock.advance(0.25)
    assert controller.offer_frame(solid(CYAN)) is True  # 第 1 张正式注册帧
    assert controller.captured_count == 1


def test_inflight_tts_completion_is_dropped_during_teaching(
    library: FeatureLibrary, monkeypatch: pytest.MonkeyPatch
) -> None:
    """教学开始前已在跑的 TTS 不能在教学录音期间把音频写回来（防回声）。"""
    pending: list[Any] = []

    def deferred_speak(
        text: str, *, on_success: Any = None, on_error: Any = None, **_: Any
    ) -> None:
        pending.append(on_success)

    monkeypatch.setattr(app, "speak_in_background", deferred_speak)
    controller = make_controller(library)
    controller.process(solid(APPLE))
    assert pending and controller.audio_path is None

    controller.begin_teaching(solid(APPLE))
    assert controller.audio_path is None
    pending[0](Path("C:/old.mp3"))  # 旧合成此刻才完成
    assert controller.audio_path is None
    assert controller.process(solid(WHITE)).audio is None


def test_short_capture_commit_keeps_teaching_and_ui_truthful(
    library: FeatureLibrary, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """抓帧不足时 commit 必须拒绝，界面不能谎报「已恢复」。"""
    monkeypatch.setattr(app, "speak_in_background", _fake_speaker([], tmp_path))
    clock = FakeClock()
    controller = make_controller(library, clock=clock)
    monkeypatch.setattr(app, "get_realtime", lambda library=None: controller)
    before = library.total_vectors

    controller.begin_teaching(solid(APPLE))
    clock.advance(0.3)
    controller.offer_frame(solid(WHITE))  # 预热丢弃
    clock.advance(0.3)
    controller.offer_frame(solid(CYAN))  # 只有 1 张正式帧

    out = app.teaching_confirm("鼠标")

    assert "只抓到 1/5 帧" in out[0]
    assert controller.teaching is True
    assert library.total_vectors == before
    # 名称框/录音保留（gr.update），实时结论不写「已恢复」。
    assert not isinstance(out[3], str)
    assert not isinstance(out[5], str)


def test_encoder_failure_leaves_library_untouched(
    library: FeatureLibrary, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """5 帧必须在 extract_features 里全部算完再 append；中途失败不留半条脏数据。"""
    monkeypatch.setattr(app, "speak_in_background", _fake_speaker([], tmp_path))

    def broken_encoder(images: Any) -> Any:
        raise RuntimeError("编码器故障")

    clock = FakeClock()
    controller = app.RealtimeController(
        library,
        encoder=broken_encoder,
        gate=SpeechGate(4.0, clock=clock),
        clock=clock,
    )
    before_vectors = library.total_vectors
    before_bytes = library.path.read_bytes()

    controller.begin_teaching(solid(APPLE))
    _capture_five(controller, clock)
    status, _ = controller.commit_teaching("鼠标")

    assert "入库失败" in status and "编码器故障" in status
    assert library.total_vectors == before_vectors
    assert library.path.read_bytes() == before_bytes
    assert controller.teaching is False


def test_sequence_teaching_uses_warmup_then_captures_five(
    library: FeatureLibrary, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """无摄像头兜底：7 张序列 = 1 张当前帧 + 1 张预热 + 5 张注册帧。"""
    colors = (WHITE, CYAN, ORANGE, PINK, GRAY, TEAL, APPLE)
    paths: list[str] = []
    for index, color in enumerate(colors):
        path = tmp_path / f"seq_{index}.png"
        solid(color, (64, 64)).save(path)
        paths.append(str(path))

    clock = FakeClock()
    controller = app.RealtimeController(
        library,
        encoder=TEACH_ENCODER,
        gate=SpeechGate(4.0, clock=clock),
        clock=clock,
    )
    monkeypatch.setattr(app, "get_realtime", lambda library=None: controller)
    # teach_from_sequence 用 time.sleep 控制 0.3 s 节奏；测试里同步推进假时钟。
    monkeypatch.setattr(app.time, "sleep", clock.advance)

    outputs = list(app.teach_from_sequence(paths))

    assert outputs and "已进入教学态" in outputs[0][0]
    assert controller.teaching is True
    assert controller.captured_count == 5


def test_save_failure_rolls_back_in_memory_library(
    library: FeatureLibrary, monkeypatch: pytest.MonkeyPatch
) -> None:
    """写盘失败时不能留下"内存 +5、磁盘 +0"的半条记录。"""
    clock = FakeClock()
    controller = app.RealtimeController(
        library,
        encoder=TEACH_ENCODER,
        gate=SpeechGate(4.0, clock=clock),
        clock=clock,
        reloader=lambda: load_library(library.path),
    )
    monkeypatch.setattr(
        app, "save_library", lambda *a, **k: (_ for _ in ()).throw(OSError("磁盘满"))
    )

    controller.begin_teaching(solid(APPLE))
    _capture_five(controller, clock)
    status, _ = controller.commit_teaching("鼠标")

    assert "入库失败" in status and "磁盘满" in status
    assert controller.library.total_vectors == 1
    assert load_library(library.path).total_vectors == 1
    assert controller.teaching is False


def test_gate_reset_after_teaching_allows_same_result(
    library: FeatureLibrary, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spoken: list[str] = []
    monkeypatch.setattr(app, "speak_in_background", _fake_speaker(spoken, tmp_path))
    clock = FakeClock()
    controller = make_controller(library, clock=clock)

    assert controller.process(solid(APPLE)).gate_spoken is True
    controller.begin_teaching(solid(APPLE))
    clock.advance(5.0)
    controller.cancel_teaching()
    # 教学结束后同一结果应立刻能播：reset 清空了"上次播报"与冷却计时。
    assert controller.process(solid(APPLE)).gate_spoken is True
    assert spoken == ["识别结果：苹果", "识别结果：苹果"]


def test_bad_frame_does_not_break_other_tabs(
    library: FeatureLibrary, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app, "speak_in_background", _fake_speaker([], tmp_path))
    library.add("杯子", [solid(CUP)], encoder=TEACH_ENCODER)
    controller = make_controller(library)

    bad = controller.process("does-not-exist.jpg")
    assert bad.frame is None and "无法解码" in bad.headline

    # Tab1 仍能录入，Tab2 仍能识别：Tab3 的异常没有污染共享状态。
    apple_path = tmp_path / "apple.png"
    solid(APPLE).save(apple_path)
    status, rows, _ = app.enroll_images(
        "苹果2", [str(apple_path)], library=library, encoder=TEACH_ENCODER
    )
    assert "已录入" in status

    headline, annotated, _ = app.recognize_image(
        solid(CUP), library=library, encoder=TEACH_ENCODER
    )
    assert "识别结果：杯子" in headline and annotated is not None


def test_teaching_transcribe_outside_teaching_is_rejected(
    library: FeatureLibrary, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app, "is_asr_configured", lambda: True)
    monkeypatch.setattr(app, "asr", lambda path: pytest.fail("非教学态不应调用 ASR"))
    controller = make_controller(library)

    name, status, _ = controller.transcribe("teach.wav", "苹果")

    assert name == "苹果" and "请先点「教它」" in status


def test_tts_failure_keeps_recognition(
    library: FeatureLibrary, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(text: str, *, on_error=None, **_: Any):
        assert on_error is not None
        on_error(TtsSynthesisError("语音合成失败（ClientConnectorError）"))
        return threading.Thread()

    monkeypatch.setattr(app, "speak_in_background", failing)
    controller = make_controller(library)

    outcome = controller.process(solid(APPLE))

    assert outcome.frame is not None and "识别结果：苹果" in outcome.headline
    assert "语音播报不可用" in outcome.tts_status
    assert "ClientConnectorError" in outcome.tts_status


def test_ui_event_functions_wire_controller(
    library: FeatureLibrary, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """界面回调本身也要覆盖：输出个数、暂停帧、确认返回值的形状。"""
    spoken: list[str] = []
    monkeypatch.setattr(app, "speak_in_background", _fake_speaker(spoken, tmp_path))
    clock = FakeClock()
    controller = make_controller(library, clock=clock)
    monkeypatch.setattr(app, "get_realtime", lambda library=None: controller)

    frame_out = app.realtime_frame(solid(APPLE, (48, 40)))
    assert len(frame_out) == 6
    assert frame_out[0] is not None and frame_out[0].shape == (40, 48, 3)

    started = app.start_teaching(solid(APPLE))
    assert len(started) == 6
    assert controller.teaching is True
    assert started[3] is None  # 清空播报，防回声

    # 教学态下帧回调不刷新画面；这一帧是第 2 个预热帧，仍不计数。
    clock.advance(0.3)
    paused = app.realtime_frame(solid(WHITE))
    assert len(paused) == 6
    assert not isinstance(paused[0], np.ndarray)
    assert controller.captured_count == 0

    _capture_five(controller, clock)
    confirmed = app.teaching_confirm("鼠标")
    assert len(confirmed) == 8
    assert "已入库" in confirmed[0]
    assert confirmed[5].startswith("▶ 已恢复")

    # 再开一次教学后取消，验证界面回调的取消分支。
    app.start_teaching(solid(APPLE))
    clock.advance(0.3)
    app.realtime_frame(solid(WHITE))
    cancelled = app.teaching_cancel()
    assert len(cancelled) == 8
    assert "已取消教学" in cancelled[0]


def test_play_frame_sequence_yields_one_frame_per_image(
    library: FeatureLibrary, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """无摄像头兜底：预置帧序列生成器每张图 yield 一次，输出形状与流事件一致。"""
    first = tmp_path / "frame_1.png"
    second = tmp_path / "frame_2.png"
    solid(APPLE).save(first)
    solid(BANANA).save(second)
    library.add("香蕉", [solid(BANANA)], encoder=TEACH_ENCODER)
    monkeypatch.setattr(app, "speak_in_background", _fake_speaker([], tmp_path))
    controller = app.RealtimeController(
        library,
        encoder=TEACH_ENCODER,
        gate=SpeechGate(4.0, clock=FakeClock()),
        clock=FakeClock(),
        min_interval=0.0,
    )
    monkeypatch.setattr(app, "get_realtime", lambda library=None: controller)
    monkeypatch.setattr(app.time, "sleep", lambda _seconds: None)

    outputs = list(app.play_frame_sequence([str(first), str(second)]))

    assert len(outputs) == 2
    assert all(len(item) == 6 for item in outputs)
    assert outputs[0][0] is not None and outputs[1][0] is not None
    assert "识别结果：苹果" in outputs[0][1]
    assert "识别结果：香蕉" in outputs[1][1]


def test_build_demo_has_realtime_tab_and_teaching_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = FeatureLibrary(path=tmp_path / "features.json", dim=DIM)
    monkeypatch.setattr(app, "get_library", lambda: library)
    monkeypatch.setattr(app, "is_asr_configured", lambda: True)

    demo = app.build_demo()
    buttons = {
        block.value
        for block in demo.blocks.values()
        if isinstance(block, gr.Button)
    }
    assert "教它（抓 5 帧，暂停识别与播报）" in buttons
    assert "确认入库" in buttons and "取消教学" in buttons
    assert "用帧序列抓 5 帧" in buttons

    cameras = [
        block
        for block in demo.blocks.values()
        if isinstance(block, gr.Image) and getattr(block, "streaming", False)
    ]
    assert len(cameras) == 1
    assert "webcam" in (cameras[0].sources or [])
    # Gradio 4.44 没有 stream_every 参数：服务端在 RealtimeController 里
    # 按 REALTIME_STREAM_EVERY 节流，保证实际识别频率 0.3–0.5 s。
    assert app.REALTIME_STREAM_EVERY == pytest.approx(0.3)

    # 两个播报播放器（Tab2 静态 + Tab3 实时）都自动播放；两个麦克风输入。
    audios = [b for b in demo.blocks.values() if isinstance(b, gr.Audio)]
    playback = [b for b in audios if getattr(b, "autoplay", False)]
    microphones = [b for b in audios if not getattr(b, "autoplay", False)]
    assert len(playback) == 2
    assert len(microphones) == 2


# --------------------------------------------------------------------------- #
# P5.1 背景自适应
# --------------------------------------------------------------------------- #
def test_background_learning_collects_frames_and_persists(
    library: FeatureLibrary, tmp_path: Path
) -> None:
    controller = make_background_controller(library, tmp_path)

    first = controller.process(solid(APPLE))
    assert controller.background_ready is False
    assert "正在学习背景" in first.headline
    assert first.frame is not None  # 学习期间仍显示原始画面
    assert "背景学习中" in first.stats

    _learn_background(controller, APPLE)

    assert controller.background is not None
    assert (
        controller.background.sample_count
        == controller.learn_frames - controller.drop_first
    )
    assert "背景已建立" in controller.stats_text()

    saved_raw = json.loads(
        (tmp_path / "background.json").read_text(encoding="utf-8")
    )
    assert saved_raw["dim"] == DIM  # 假编码器是 8 维，生产链路是 512
    assert saved_raw["sample_count"] == (
        controller.learn_frames - controller.drop_first
    )
    assert "objects" not in saved_raw
    # 背景文件绝不能混进物体库。
    assert library.object_count == 1 and library.total_vectors == 1


def test_background_gate_hit_does_not_retrieve_or_speak(
    library: FeatureLibrary, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spoken: list[str] = []
    monkeypatch.setattr(app, "speak_in_background", _fake_speaker(spoken, tmp_path))
    controller = make_background_controller(library, tmp_path)
    _learn_background(controller, APPLE)

    outcome = controller.process(solid(APPLE))

    assert outcome.is_background is True
    assert outcome.frame is not None
    assert BACKGROUND_LABEL in outcome.headline
    assert "不进入物体检索" in outcome.headline
    assert outcome.gate_spoken is False and spoken == []
    assert controller.last_result is not None
    assert controller.last_result.name == BACKGROUND_LABEL
    assert controller.last_stream is not None
    assert controller.last_stream.speech == ""
    assert controller.last_stream.bg_similarity == pytest.approx(1.0, abs=1e-6)
    assert "T_rt" in outcome.stats and "T_bg" in outcome.stats


def test_gate_switch_off_returns_to_rc1_behavior(
    library: FeatureLibrary, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spoken: list[str] = []
    monkeypatch.setattr(app, "speak_in_background", _fake_speaker(spoken, tmp_path))
    controller = make_background_controller(library, tmp_path)
    _learn_background(controller, APPLE)

    headline, stats = controller.set_gate_enabled(False)
    assert "背景门已关闭" in headline
    assert "背景门 **关**" in stats

    outcome = controller.process(solid(APPLE))

    assert outcome.is_background is False
    assert "识别结果：苹果" in outcome.headline
    assert outcome.gate_spoken is True and spoken == ["识别结果：苹果"]
    assert controller.last_stream is not None
    assert controller.last_stream.bg_similarity is None


def test_relearn_background_resets_and_rebuilds(
    library: FeatureLibrary, tmp_path: Path
) -> None:
    controller = make_background_controller(library, tmp_path)
    _learn_background(controller, APPLE)
    assert controller.background_ready is True

    headline, stats = controller.relearn_background()

    assert "重新学习背景" in headline
    assert controller.background is None
    assert controller.background_learning is True
    assert "背景学习中" in stats

    _learn_background(controller, APPLE)
    assert controller.background_ready is True


def test_teaching_still_works_after_background_is_learned(
    library: FeatureLibrary, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app, "speak_in_background", _fake_speaker([], tmp_path))
    clock = FakeClock()
    controller = make_background_controller(library, tmp_path, clock=clock)
    _learn_background(controller, APPLE)

    ok, _ = controller.begin_teaching(solid(APPLE))
    assert ok is True
    _capture_five(controller, clock)
    status, _ = controller.commit_teaching("鼠标")

    assert "已入库「鼠标」" in status
    assert controller.teaching is False
    # 已经学好的背景不被教学流程清掉。
    assert controller.background is not None
    assert controller.background_learning is False


def test_build_demo_has_relearn_button_and_gate_checkbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = FeatureLibrary(path=tmp_path / "features.json", dim=DIM)
    controller = make_background_controller(library, tmp_path)
    monkeypatch.setattr(app, "get_library", lambda: library)
    monkeypatch.setattr(app, "is_asr_configured", lambda: True)
    monkeypatch.setattr(app, "get_realtime", lambda library=None: controller)

    demo = app.build_demo()
    buttons = {
        block.value
        for block in demo.blocks.values()
        if isinstance(block, gr.Button)
    }
    assert "重新学背景（对准空场景）" in buttons

    checkboxes = [
        block for block in demo.blocks.values() if isinstance(block, gr.Checkbox)
    ]
    assert any("背景门" in (box.label or "") for box in checkboxes)
