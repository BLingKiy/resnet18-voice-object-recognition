"""``voice/tts.py`` 单测：合成结果、失败语义与实时链路节流。

全部用假的 edge-tts 与假时钟，不联网、不睡眠（除"不阻塞"那一条用 0.2 秒
真实等待验证调用方立即返回）；真实 Edge TTS 的联网验证放在
``tools/verify_tts.py``，由人工执行并把输出贴进验收记录。
"""

from __future__ import annotations

import importlib
import sys
import threading
import time
from pathlib import Path

import pytest

# 注意：``voice.__init__`` 把 ``voice.tts`` 这个包属性覆盖成了同名函数，
# 所以 ``import voice.tts as mod`` 拿到的是函数；必须走 importlib。
tts_module = importlib.import_module("voice.tts")

from voice.tts import (  # noqa: E402
    TTS_COOLDOWN_SECONDS,
    TTS_UNAVAILABLE_MESSAGE,
    TtsAudioError,
    TtsDependencyError,
    TtsSynthesisError,
    TtsThrottler,
    normalize_text,
    resolve_cooldown,
    synthesize,
    tts,
)

# --------------------------------------------------------------------------- #
# 假 edge-tts：按 100ns 单位给句边界元数据，音频块给定字节数
# --------------------------------------------------------------------------- #
_AUDIO_CHUNK = b"\xff\xfb\x90\x00" * 300  # 1200 字节，足够当"有音频"


class _FakeCommunicate:
    """只实现本项目用到的两个入参与 ``stream()``。"""

    def __init__(self, text: str, voice: str, **kwargs: object) -> None:
        self.text = text
        self.voice = voice
        self.kwargs = kwargs

    async def stream(self):
        yield {"type": "audio", "data": _AUDIO_CHUNK}
        # 1200 ms = 12_000_000 * 100ns
        yield {
            "type": "SentenceBoundary",
            "offset": 0.0,
            "duration": 12_000_000.0,
            "text": self.text,
        }


class _FakeEdgeTts:
    Communicate = _FakeCommunicate


class _FailingCommunicate(_FakeCommunicate):
    async def stream(self):
        raise ConnectionResetError("模拟断网")
        yield  # pragma: no cover - 让函数成为 async generator


class _FailingEdgeTts:
    Communicate = _FailingCommunicate


class _SilentCommunicate(_FakeCommunicate):
    async def stream(self):
        # 只有元数据、没有音频块：等价于"服务回了空音频"
        yield {"type": "SentenceBoundary", "offset": 0.0, "duration": 1_000_000.0}


class _SilentEdgeTts:
    Communicate = _SilentCommunicate


@pytest.fixture
def fake_edge_tts(monkeypatch: pytest.MonkeyPatch) -> type[_FakeEdgeTts]:
    monkeypatch.setattr(tts_module, "_require_edge_tts", lambda: _FakeEdgeTts)
    return _FakeEdgeTts


def test_synthesize_writes_file_and_reports_duration_and_elapsed(
    tmp_path: Path, fake_edge_tts: type[_FakeEdgeTts]
) -> None:
    result = synthesize("识别结果：苹果", output_dir=tmp_path)

    assert result.text == "识别结果：苹果"
    assert result.audio_path.parent == tmp_path
    assert result.audio_path.is_file()
    assert result.audio_path.suffix == ".mp3"
    assert result.bytes_written == len(_AUDIO_CHUNK)
    # 1200 字节 × 8 / 48 kbps = 200 ms
    assert result.duration_ms == 200
    assert result.elapsed_ms >= 0
    assert result.voice == tts_module.DEFAULT_VOICE


def test_duration_is_derived_from_cbr_bytes_not_sentence_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """句边界元数据会漏掉尾部静音，时长必须以实际字节数为准。"""

    class _LongerThanMetadata(_FakeCommunicate):
        async def stream(self):
            # 元数据只报 100 ms，但实际写了 12000 字节（= 2000 ms）
            yield {"type": "SentenceBoundary", "offset": 0.0, "duration": 1_000_000.0}
            yield {"type": "audio", "data": _AUDIO_CHUNK * 10}

    class _LongerThanMetadataTts:
        Communicate = _LongerThanMetadata

    monkeypatch.setattr(
        tts_module, "_require_edge_tts", lambda: _LongerThanMetadataTts
    )

    result = synthesize("未知物体", output_dir=tmp_path)

    assert result.duration_ms == 2000


def test_tts_returns_path_string(tmp_path: Path, fake_edge_tts: type[_FakeEdgeTts]) -> None:
    path = tts("识别结果：鼠标", output_dir=tmp_path)

    assert isinstance(path, str)
    assert Path(path).is_file()


def test_empty_text_is_rejected_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tts_module,
        "_require_edge_tts",
        lambda: pytest.fail("空文本不应触发 edge-tts 导入"),
    )

    with pytest.raises(TtsAudioError):
        synthesize("   ", output_dir=tmp_path)


def test_network_failure_becomes_synthesis_error_and_leaves_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tts_module, "_require_edge_tts", lambda: _FailingEdgeTts)

    with pytest.raises(TtsSynthesisError) as excinfo:
        synthesize("识别结果：苹果", output_dir=tmp_path)

    assert "ConnectionResetError" in str(excinfo.value)
    assert list(tmp_path.glob("tts-*.mp3")) == []


def test_empty_audio_is_treated_as_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tts_module, "_require_edge_tts", lambda: _SilentEdgeTts)

    with pytest.raises(TtsSynthesisError):
        synthesize("识别结果：苹果", output_dir=tmp_path)

    assert list(tmp_path.glob("tts-*.mp3")) == []


def test_missing_dependency_is_reported_with_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # sys.modules[name] = None 会让后续 import 直接抛 ImportError。
    monkeypatch.setitem(sys.modules, "edge_tts", None)

    with pytest.raises(TtsDependencyError) as excinfo:
        synthesize("识别结果：苹果", output_dir=tmp_path)

    assert "pip install edge-tts" in str(excinfo.value)


def test_write_failure_is_wrapped_as_audio_error(
    tmp_path: Path, fake_edge_tts: type[_FakeEdgeTts]
) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("我是文件，不是目录", encoding="utf-8")

    with pytest.raises(TtsAudioError):
        synthesize("识别结果：苹果", output_dir=blocked / "sub")


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text("识别结果：\n苹果 ") == "识别结果： 苹果"
    with pytest.raises(TtsAudioError):
        normalize_text(None)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, TTS_COOLDOWN_SECONDS), (2.0, 3.0), (9.0, 5.0), (4.2, 4.2), ("abc", 4.0)],
)
def test_cooldown_is_clamped_into_3_to_5_seconds(
    raw: object, expected: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 显式参数优先于环境变量；这里先保证环境变量不会影响断言。
    monkeypatch.delenv(tts_module.ENV_COOLDOWN, raising=False)
    assert resolve_cooldown(raw) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# 节流器：同结果不重复、变化才播、冷却生效、不阻塞
# --------------------------------------------------------------------------- #
class _Clock:
    """可手动推进的假时钟。"""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _recording_throttler(
    clock: _Clock, spoken: list[str], *, cooldown: float = 4.0
) -> TtsThrottler:
    def speak(text: str) -> None:
        spoken.append(text)

    return TtsThrottler(cooldown_seconds=cooldown, speak=speak, clock=clock)


def test_same_result_is_not_repeated_within_cooldown() -> None:
    clock, spoken = _Clock(), []
    throttler = _recording_throttler(clock, spoken)

    assert throttler.announce("识别结果：苹果") is True
    assert throttler.wait_idle(timeout=2.0)

    clock.advance(1.0)  # 冷却窗口内
    assert throttler.announce("识别结果：苹果") is False

    clock.advance(10.0)  # 冷却早就过了，但文本没变
    assert throttler.announce("识别结果：苹果") is False

    stats = throttler.stats()
    assert spoken == ["识别结果：苹果"]
    assert stats["spoken"] == 1
    assert stats["skipped_same"] == 2


def test_changed_result_plays_immediately_once_cooldown_elapsed() -> None:
    clock, spoken = _Clock(), []
    throttler = _recording_throttler(clock, spoken)

    assert throttler.announce("识别结果：苹果") is True
    assert throttler.wait_idle(timeout=2.0)

    clock.advance(4.0)  # 恰好到冷却边界
    assert throttler.announce("识别结果：鼠标") is True
    assert throttler.wait_idle(timeout=2.0)

    assert spoken == ["识别结果：苹果", "识别结果：鼠标"]


def test_change_inside_cooldown_is_dropped_not_queued() -> None:
    clock, spoken = _Clock(), []
    throttler = _recording_throttler(clock, spoken)

    throttler.announce("识别结果：苹果")
    assert throttler.wait_idle(timeout=2.0)

    clock.advance(1.0)
    assert throttler.announce("识别结果：鼠标") is False  # 冷却窗口内变化也不播

    clock.advance(3.5)  # 窗口打开，新结果仍在，这时才播
    assert throttler.announce("识别结果：鼠标") is True
    assert throttler.wait_idle(timeout=2.0)

    assert spoken == ["识别结果：苹果", "识别结果：鼠标"]
    assert throttler.stats()["skipped_cooldown"] == 1


def test_reset_clears_history_so_same_text_can_play_again() -> None:
    clock, spoken = _Clock(), []
    throttler = _recording_throttler(clock, spoken)

    throttler.announce("识别结果：苹果")
    assert throttler.wait_idle(timeout=2.0)

    throttler.reset()
    assert throttler.announce("识别结果：苹果") is True
    assert throttler.wait_idle(timeout=2.0)

    assert spoken == ["识别结果：苹果", "识别结果：苹果"]


def test_announce_does_not_block_the_caller() -> None:
    started = threading.Event()
    finished = threading.Event()

    def slow_speak(text: str) -> None:
        started.set()
        time.sleep(0.3)
        finished.set()

    throttler = TtsThrottler(cooldown_seconds=4.0, speak=slow_speak)

    begin = time.perf_counter()
    assert throttler.announce("识别结果：键盘") is True
    elapsed = time.perf_counter() - begin

    assert elapsed < 0.1, f"announce 阻塞了 {elapsed * 1000:.1f} ms"
    assert started.wait(timeout=2.0)
    assert not finished.is_set()  # 合成还在后台跑，说明确实没被同步等待
    assert throttler.wait_idle(timeout=5.0)


def test_busy_worker_skips_new_result() -> None:
    clock = _Clock()
    release = threading.Event()

    def blocking_speak(text: str) -> None:
        release.wait(timeout=5.0)

    throttler = TtsThrottler(cooldown_seconds=4.0, speak=blocking_speak, clock=clock)
    assert throttler.announce("识别结果：苹果") is True

    clock.advance(10.0)  # 冷却已过，但上一段还在合成
    assert throttler.announce("识别结果：鼠标") is False
    assert throttler.stats()["skipped_busy"] == 1

    release.set()
    assert throttler.wait_idle(timeout=5.0)


def test_failure_is_recorded_and_message_is_stable() -> None:
    def boom(text: str) -> None:
        raise TtsSynthesisError("模拟断网")

    clock = _Clock()
    throttler = TtsThrottler(cooldown_seconds=4.0, speak=boom, clock=clock)

    assert throttler.announce("识别结果：苹果") is True
    assert throttler.wait_idle(timeout=2.0)

    stats = throttler.stats()
    assert stats["failed"] == 1
    assert stats["spoken"] == 0
    assert "TtsSynthesisError" in str(stats["last_error"])
    assert TTS_UNAVAILABLE_MESSAGE == "语音播报不可用，请检查网络"
    # 失败后允许下一次变化重试（冷却仍生效）
    clock.advance(5.0)
    assert throttler.announce("识别结果：苹果") is True
    assert throttler.wait_idle(timeout=2.0)
    assert throttler.stats()["failed"] == 2
