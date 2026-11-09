"""``voice/tts.py`` 单元测试：合成、失败路径、后台线程与节流。

全部用假 ``edge_tts`` 模块，不联网——现场演示机器的网络是最不可控的
依赖，单测必须能离线跑完。真实联网合成单独放在 ``test_live_synthesis``，
只有显式设置 ``RN18_TTS_LIVE=1`` 时才执行。
"""

from __future__ import annotations

import importlib
import os
import sys
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from voice import (
    SpeechGate,
    TtsConfig,
    TtsDependencyError,
    TtsError,
    TtsSynthesisError,
    speak_in_background,
    synthesize,
    tts,
)

# voice/__init__.py 把包属性 ``tts`` 绑成了函数，子模块必须用 importlib 取。
tts_module = importlib.import_module("voice.tts")

FAKE_PAYLOAD = b"ID3\x03\x00\x00\x00fake-mp3-payload"


def _fake_edge_tts(
    calls: list[tuple[str, str, dict[str, Any]]],
    *,
    error: BaseException | None = None,
    payload: bytes = FAKE_PAYLOAD,
) -> Any:
    """构造一个只做"记录参数 + 写文件"的假 edge_tts 模块。

    Args:
        calls: 每次 ``Communicate(...)`` 都会把 (text, voice, kwargs) 追加进来。
        error: 非 None 时 ``save()`` 直接抛这个异常，用来模拟断网/引擎报错。
        payload: 成功时写入文件的字节内容。

    Returns:
        可直接替换 :func:`tts_module._load_edge_tts` 返回值的假模块。
    """

    class FakeCommunicate:
        def __init__(self, text: str, voice: str, **kwargs: Any) -> None:
            calls.append((text, voice, kwargs))

        async def save(self, path: str) -> None:
            if error is not None:
                raise error
            Path(path).write_bytes(payload)

    return SimpleNamespace(Communicate=FakeCommunicate)


def test_synthesize_writes_a_new_mp3_for_each_call(tmp_path: Path) -> None:
    """核心判据：每次调用都按**当次文本**生成一个新文件。"""
    calls: list[tuple[str, str, dict[str, Any]]] = []

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tts_module, "_load_edge_tts", lambda: _fake_edge_tts(calls))
        first = tts("识别结果：苹果", out_dir=tmp_path)
        second = tts("识别结果：香蕉", out_dir=tmp_path)

    assert Path(first).is_file() and Path(second).is_file()
    assert first != second, "同一路径不会让 gr.Audio 再次 autoplay，必须每次新文件"
    assert Path(first).read_bytes() == FAKE_PAYLOAD
    assert [call[0] for call in calls] == ["识别结果：苹果", "识别结果：香蕉"]
    assert all(call[1] == tts_module.DEFAULT_VOICE for call in calls)


def test_synthesize_uses_configured_voice_and_rate(tmp_path: Path) -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tts_module, "_load_edge_tts", lambda: _fake_edge_tts(calls))
        synthesize(
            "未知物体",
            config=TtsConfig(voice="zh-CN-YunxiNeural", rate="+10%", volume="+0%"),
            out_dir=tmp_path,
        )

    text, voice, kwargs = calls[0]
    assert text == "未知物体"
    assert voice == "zh-CN-YunxiNeural"
    assert kwargs["rate"] == "+10%"


def test_engine_failure_raises_catchable_error_and_leaves_no_file(
    tmp_path: Path,
) -> None:
    """断网/引擎报错 -> TtsSynthesisError（可捕获），且不留半成品文件。"""

    def boom() -> Any:
        return _fake_edge_tts([], error=RuntimeError("network unreachable"))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tts_module, "_load_edge_tts", boom)
        with pytest.raises(TtsSynthesisError) as excinfo:
            synthesize("识别结果：苹果", out_dir=tmp_path)

    assert "请检查网络" in str(excinfo.value)
    assert "RuntimeError" in str(excinfo.value)
    assert not list(tmp_path.glob("rn18-*.mp3"))


def test_empty_or_blank_audio_raises_instead_of_reporting_success(
    tmp_path: Path,
) -> None:
    """引擎"成功"但没产出音频时不能当成功，否则界面会显示已播报却没声音。"""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            tts_module, "_load_edge_tts", lambda: _fake_edge_tts([], payload=b"")
        )
        with pytest.raises(TtsSynthesisError) as excinfo:
            synthesize("识别结果：苹果", out_dir=tmp_path)

    assert "没有返回音频数据" in str(excinfo.value)
    assert not list(tmp_path.glob("rn18-*.mp3"))


def test_blank_text_is_rejected_before_touching_the_network(tmp_path: Path) -> None:
    def must_not_load() -> Any:  # pragma: no cover - 被调用即失败
        raise AssertionError("空文本不应加载 edge_tts")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tts_module, "_load_edge_tts", must_not_load)
        with pytest.raises(TtsSynthesisError):
            synthesize("   ", out_dir=tmp_path)


def test_missing_edge_tts_becomes_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未安装 edge-tts 时抛 TtsDependencyError，而不是裸 ImportError。"""
    monkeypatch.setitem(sys.modules, "edge_tts", None)  # None 让 import 直接失败

    with pytest.raises(TtsDependencyError) as excinfo:
        tts_module._load_edge_tts()

    assert "pip install edge-tts" in str(excinfo.value)


def test_tts_entry_point_returns_string_path(tmp_path: Path) -> None:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tts_module, "_load_edge_tts", lambda: _fake_edge_tts([]))
        path = tts("识别结果：水杯", out_dir=tmp_path)

    assert isinstance(path, str)
    assert Path(path).suffix == ".mp3"


def test_prune_keeps_only_recent_files(tmp_path: Path) -> None:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tts_module, "_load_edge_tts", lambda: _fake_edge_tts([]))
        mp.setattr(tts_module, "_KEEP_FILES", 3)
        for index in range(5):
            synthesize(f"识别结果：物体{index}", out_dir=tmp_path)

    assert len(list(tmp_path.glob("rn18-*.mp3"))) == 3


def test_speak_in_background_reports_success_and_failure(tmp_path: Path) -> None:
    """后台线程：成功回调拿到路径，失败回调拿到可展示的 TtsError。"""
    ok_paths: list[Path] = []
    errors: list[TtsError] = []

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tts_module, "_load_edge_tts", lambda: _fake_edge_tts([]))
        thread = speak_in_background(
            "识别结果：耳机", on_success=ok_paths.append, out_dir=tmp_path
        )
        thread.join(timeout=10)

    assert not thread.is_alive()
    assert len(ok_paths) == 1 and ok_paths[0].is_file()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            tts_module,
            "_load_edge_tts",
            lambda: _fake_edge_tts([], error=RuntimeError("offline")),
        )
        thread = speak_in_background(
            "识别结果：耳机", on_error=errors.append, out_dir=tmp_path
        )
        thread.join(timeout=10)

    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], TtsSynthesisError)


def test_speech_gate_speaks_only_on_change_and_after_cooldown() -> None:
    """实时链路节流规则：结果变化 + 冷却已过，两个条件同时满足才播。"""
    now = [0.0]
    gate = SpeechGate(cooldown_seconds=4.0, clock=lambda: now[0])

    assert gate.should_speak("识别结果：苹果") is True
    # 同一物体连续帧：文本没变，不重复播报（10 秒后也一样）
    now[0] = 10.0
    assert gate.should_speak("识别结果：苹果") is False
    # 结果变了，但距上次播报只过了 1 秒：压住不播
    now[0] = 1.0
    assert gate.should_speak("识别结果：香蕉") is False
    # 冷却时间到：换物体就播
    now[0] = 4.5
    assert gate.should_speak("识别结果：香蕉") is True
    # 空文本永不播
    assert gate.should_speak("   ") is False

    gate.reset()
    now[0] = 4.6
    assert gate.should_speak("识别结果：香蕉") is True


def test_speech_gate_rejects_negative_cooldown() -> None:
    with pytest.raises(ValueError):
        SpeechGate(cooldown_seconds=-1.0)


@pytest.mark.skipif(
    os.environ.get("RN18_TTS_LIVE") != "1",
    reason="真实 Edge TTS 需要联网；设置 RN18_TTS_LIVE=1 才执行",
)
def test_live_synthesis_produces_playable_mp3(tmp_path: Path) -> None:
    """真实联网合成一条，验证 edge-tts 参数在当前版本仍然有效。"""
    path = synthesize("识别结果：苹果", out_dir=tmp_path, config=TtsConfig())

    assert path.is_file()
    assert path.stat().st_size > 1024
    assert path.read_bytes()[:3] == b"ID3" or path.read_bytes()[:2] == b"\xff\xfb"


def test_output_dir_can_be_overridden_by_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RN18_TTS_DIR", str(tmp_path))
    assert tts_module.default_output_dir() == tmp_path


def test_generated_filenames_are_unique() -> None:
    """文件名唯一性靠时间戳 + uuid 双保险（同一秒内连续调用也不能撞名）。"""
    names = {tts_module._build_filename("识别结果：苹果") for _ in range(50)}
    assert len(names) == 50
    assert all(name.endswith(".mp3") for name in names)


def test_synthesis_is_thread_safe_enough_for_parallel_calls(tmp_path: Path) -> None:
    """两个识别几乎同时完成时不能互相覆盖文件。"""
    results: list[str] = []
    lock = threading.Lock()

    def worker(text: str) -> None:
        path = synthesize(text, out_dir=tmp_path)
        with lock:
            results.append(path)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tts_module, "_load_edge_tts", lambda: _fake_edge_tts([]))
        threads = [
            threading.Thread(target=worker, args=(f"识别结果：物体{uuid.uuid4().hex[:4]}",))
            for _ in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

    assert len(results) == 4
    assert len(set(results)) == 4
    assert all(Path(item).is_file() for item in results)
