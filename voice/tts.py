"""Edge TTS 语音播报封装（P4：静态链路逐次播报 + 实时链路节流）。

对外契约（界面层只需依赖这些名字）::

    from voice import TtsError, TtsResult, TtsThrottler, announce_stream, tts

    path = tts("识别结果：苹果")        # 同步合成到本地 mp3，返回音频路径
    result = synthesize("未知物体")     # 同上，但带合成耗时与音频时长

    throttler = TtsThrottler()          # 实时链路：结果变化 + 冷却窗口才播
    throttler.announce("识别结果：鼠标")  # 立即返回，合成与播放交给后台线程

为什么只有 Edge TTS：选题书要求「语音播报识别结果」，课程环境允许联网，
Edge TTS 的中文音色（默认 ``zh-CN-XiaoxiaoNeural``）足够清楚。按 2026-11-09
人类确认，**不引入 pyttsx3 或任何本地引擎、不接第二家云厂商、不做离线兜底**：
失败就把 :class:`TtsSynthesisError` 抛给调用方，由界面统一显示
「语音播报不可用，请检查网络」，绝不静默失败、也不拿固定提示音冒充播报。

播放位置：本模块只负责「合成到文件」。界面把返回的音频路径塞进
``gr.Audio(autoplay=True)``，浏览器在用户点击「开始识别」之后播放，
因此不需要额外的播放器进程，也不会在无音频设备的机器上抛出新的异常。
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 与 ASR 层保持一致的位置约定：voice/ 在仓库根目录下。
_REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"
ENV_VOICE = "RN18_TTS_VOICE"
ENV_OUTPUT_DIR = "RN18_TTS_DIR"
ENV_COOLDOWN = "RN18_TTS_COOLDOWN"

# 冷却窗口：任务要求 3 至 5 秒。常量写成默认值，环境变量可改但会被夹在
# 这个区间内——现场调参不能把节流逻辑改坏（例如设成 0 会每帧都合成）。
TTS_COOLDOWN_MIN_SECONDS = 3.0
TTS_COOLDOWN_MAX_SECONDS = 5.0
TTS_COOLDOWN_SECONDS = 4.0

# 断网/合成失败时界面必须原样显示这句话（验收判据逐字比对）。
TTS_UNAVAILABLE_MESSAGE = "语音播报不可用，请检查网络"

_CONNECT_TIMEOUT_SECONDS = 10
_RECEIVE_TIMEOUT_SECONDS = 30

# edge-tts 固定输出 24 kHz / 48 kbps CBR mp3（见 edge_tts.communicate 的
# outputFormat），因此可以由字节数精确换算时长。
_MP3_BITRATE_BPS = 48_000

# 每次播报都会落一个 mp3；长期演示只保留最近若干条，避免目录无限膨胀。
_CACHE_KEEP_FILES = 40


class TtsError(Exception):
    """语音播报相关错误的基类。"""


class TtsDependencyError(TtsError):
    """缺少 edge-tts 依赖。"""


class TtsSynthesisError(TtsError):
    """合成失败：断网、Edge 服务不可达、鉴权被拒或没有返回音频。

    这是**可预期**的现场故障，调用方捕获后显示
    :data:`TTS_UNAVAILABLE_MESSAGE`，不要把它变成红色堆栈。
    """


class TtsAudioError(TtsError):
    """文本为空，或本地输出目录不可写。"""


@dataclass(frozen=True, slots=True)
class TtsResult:
    """一次合成结果。

    Attributes:
        text: 实际送去合成的文本（已规范化）。
        audio_path: 生成的 mp3 路径。
        voice: 使用的音色短名，例如 ``zh-CN-XiaoxiaoNeural``。
        duration_ms: 音频时长（毫秒），用于验收记录。
        elapsed_ms: 合成墙钟耗时（毫秒），用于验收记录。
        bytes_written: 音频字节数。
    """

    text: str
    audio_path: Path
    voice: str
    duration_ms: int
    elapsed_ms: int
    bytes_written: int


def normalize_text(text: Any) -> str:
    """把任意输入规范成一行可播报文本。

    Args:
        text: 待播报文本；``None`` 视为空。

    Returns:
        去掉首尾空白、把换行压成空格后的文本。

    Raises:
        TtsAudioError: 规范化后为空——空文本没有播放意义，直接报错，
            而不是合成一段静音让现场以为播报成功。
    """
    clean = " ".join(str(text or "").split())
    if not clean:
        raise TtsAudioError("播报文本为空")
    return clean


def resolve_voice(voice: str | None = None) -> str:
    """确定音色：显式参数 > ``RN18_TTS_VOICE`` > 默认中文女声。"""
    if voice and str(voice).strip():
        return str(voice).strip()
    env_voice = os.environ.get(ENV_VOICE, "").strip()
    return env_voice or DEFAULT_VOICE


def resolve_output_dir(output_dir: str | Path | None = None) -> Path:
    """确定音频输出目录：显式参数 > ``RN18_TTS_DIR`` > 仓库内 ``.tts/``。

    默认放在仓库里是为了现场演示可预期（拷贝整个目录就走），
    ``.tts/`` 已在 ``.gitignore`` 中，不会污染提交。
    """
    if output_dir is not None:
        return Path(output_dir).expanduser()
    env_dir = os.environ.get(ENV_OUTPUT_DIR, "").strip()
    if env_dir:
        return Path(env_dir).expanduser()
    return _REPO_ROOT / ".tts"


def resolve_cooldown(cooldown_seconds: float | None = None) -> float:
    """确定实时链路的冷却秒数，并夹到 ``[3, 5]`` 区间。

    显式参数 > ``RN18_TTS_COOLDOWN`` > :data:`TTS_COOLDOWN_SECONDS`。
    非法值（非数字、负数）回退到默认值，不抛异常——现场改环境变量写错
    不应该让界面启动失败。
    """
    raw: Any = cooldown_seconds
    if raw is None:
        raw = os.environ.get(ENV_COOLDOWN, "").strip() or TTS_COOLDOWN_SECONDS
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = TTS_COOLDOWN_SECONDS
    if value != value:  # NaN
        value = TTS_COOLDOWN_SECONDS
    return min(TTS_COOLDOWN_MAX_SECONDS, max(TTS_COOLDOWN_MIN_SECONDS, value))


def _require_edge_tts() -> Any:
    """延迟导入 edge-tts；缺失时给出可执行的安装提示。"""
    try:
        import edge_tts
    except ImportError as exc:
        raise TtsDependencyError(
            "缺少依赖 edge-tts。请运行 `pip install edge-tts`。"
        ) from exc
    return edge_tts


def _discard(path: Path) -> None:
    """删除半成品音频；删不掉也不影响主流程。"""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _prune_cache(directory: Path, keep: int = _CACHE_KEEP_FILES) -> None:
    """只保留最近 ``keep`` 个 mp3，清理失败静默忽略（磁盘忙、文件被占用）。"""
    try:
        files = sorted(
            directory.glob("tts-*.mp3"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return
    for stale in files[keep:]:
        _discard(stale)


async def _stream_to_file(
    edge_tts: Any,
    text: str,
    voice: str,
    path: Path,
) -> int:
    """把合成音频流写进文件，返回写入的字节数。

    只做"落盘"一件事：时长由 :func:`estimate_duration_ms` 按字节数换算。
    edge-tts 的音频块不带时长字段，句边界元数据（100 纳秒单位）只覆盖到
    最后一句语音，会漏掉尾部静音——实测比真实时长少 15~260 ms，所以不拿
    它当结论。
    """
    communicate = edge_tts.Communicate(
        text,
        voice,
        connect_timeout=_CONNECT_TIMEOUT_SECONDS,
        receive_timeout=_RECEIVE_TIMEOUT_SECONDS,
    )
    written = 0
    with path.open("wb") as handle:
        async for chunk in communicate.stream():
            if chunk.get("type") != "audio":
                continue
            data = chunk.get("data") or b""
            if data:
                handle.write(data)
                written += len(data)
    return written


def estimate_duration_ms(bytes_written: int) -> int:
    """按 48 kbps CBR 换算 mp3 时长（毫秒）。

    edge-tts 固定输出 ``audio-24khz-48kbitrate-mono-mp3``，码率恒定，因此
    ``字节数 × 8 ÷ 48000`` 就是时长。用真实音频交叉核对过：
    19440 字节 → 3240 ms、11232 字节 → 1872 ms，与 soundfile 读数完全一致。
    """
    return int(round(bytes_written * 8 * 1000 / _MP3_BITRATE_BPS))


def synthesize(
    text: str,
    *,
    voice: str | None = None,
    output_dir: str | Path | None = None,
) -> TtsResult:
    """同步合成一段语音到本地 mp3，返回路径与耗时。

    Args:
        text: 待播报文本，例如 ``"识别结果：苹果"``。
        voice: 音色短名；``None`` 表示按 ``RN18_TTS_VOICE`` / 默认值。
        output_dir: 输出目录；``None`` 表示按 ``RN18_TTS_DIR`` / ``.tts/``。

    Returns:
        :class:`TtsResult`，含音频路径、时长与合成耗时。

    Raises:
        TtsAudioError: 文本为空或输出目录不可写。
        TtsDependencyError: 未安装 edge-tts。
        TtsSynthesisError: 断网、服务拒绝或没有返回音频数据。
    """
    clean = normalize_text(text)
    voice_name = resolve_voice(voice)
    directory = resolve_output_dir(output_dir)

    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise TtsAudioError(
            f"无法创建播报音频目录 {directory}（{type(exc).__name__}）。"
        ) from exc

    edge_tts = _require_edge_tts()
    path = directory / (
        f"tts-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.mp3"
    )

    started = time.perf_counter()
    try:
        written = asyncio.run(_stream_to_file(edge_tts, clean, voice_name, path))
    except TtsError:
        _discard(path)
        raise
    except Exception as exc:  # noqa: BLE001 - 网络/协议异常统一转成现场可读文案
        _discard(path)
        # 只保留异常类型名：底层报错可能包含一长串 WebSocket 地址，
        # 对现场排障没用，还容易让界面显示成"程序坏了"。
        raise TtsSynthesisError(
            f"语音合成失败（{type(exc).__name__}），请检查网络后重试。"
        ) from exc

    if written <= 0:
        _discard(path)
        raise TtsSynthesisError("语音合成没有返回音频数据，请检查网络后重试。")

    elapsed_ms = int(round((time.perf_counter() - started) * 1000))
    _prune_cache(directory)
    return TtsResult(
        text=clean,
        audio_path=path,
        voice=voice_name,
        duration_ms=estimate_duration_ms(written),
        elapsed_ms=elapsed_ms,
        bytes_written=written,
    )


def tts(
    text: str,
    *,
    voice: str | None = None,
    output_dir: str | Path | None = None,
) -> str:
    """合成一段语音并返回音频路径字符串。

    与函数卡里 ``voice/tts.py  tts(text) -> str`` 的契约一致；需要时长、
    耗时等验收数据时用 :func:`synthesize`。

    Args:
        text: 待播报文本。
        voice: 音色短名，``None`` 用默认值。
        output_dir: 输出目录，``None`` 用默认目录。

    Returns:
        生成的 mp3 绝对路径。

    Raises:
        同 :func:`synthesize`。
    """
    return str(
        synthesize(text, voice=voice, output_dir=output_dir).audio_path
    )


class TtsThrottler:
    """实时链路的播报节流器：结果变化才播，且两次播报间隔不小于冷却窗口。

    三条规则：

    1. **同一文本不重复播**——连续帧上识别的永远是同一个物体，逐帧合成
       既费网络又吵。
    2. **冷却窗口**——距上次播报不足 ``cooldown_seconds`` 时不播。冷却期内
       结果变了也不播，等窗口打开后的下一帧再播最新结果（不排队、不补播），
       避免"画面在几个物体之间抖，喇叭把每个都念一遍"。
    3. **合成不阻塞**——播放逻辑在后台守护线程里跑，``announce()`` 只做判断
       与状态更新，帧循环不会被网络往返卡住。

    Args:
        cooldown_seconds: 冷却秒数；``None`` 表示按环境变量/默认值并夹到
            ``[3, 5]``。
        speak: 实际执行合成（与播放）的可调用对象，默认 :func:`tts`；
            测试注入假实现即可验证节流逻辑，不联网。
        clock: 单调时钟，默认 ``time.monotonic``；测试注入假时钟。
    """

    def __init__(
        self,
        cooldown_seconds: float | None = None,
        *,
        speak: Callable[[str], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cooldown_seconds = resolve_cooldown(cooldown_seconds)
        self._speak = speak if speak is not None else tts
        self._clock = clock
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._busy = False
        self._pending_text = ""
        # 计数只增不减，reset() 不清零：验收时要看"一共拦下多少次"。
        self._attempts = 0
        self._spoken = 0
        self._failed = 0
        self._skipped_same = 0
        self._skipped_cooldown = 0
        self._skipped_busy = 0
        self._last_error: str | None = None
        self._reset_state()

    # -- 状态 ------------------------------------------------------------- #
    def _reset_state(self) -> None:
        """把播报历史清空（停止态 -> 恢复识别时调用，避免补播旧结果）。"""
        self._last_text: str | None = None
        self._last_spoken_at: float | None = None

    def reset(self) -> None:
        """清空"上次播报了什么、什么时候播的"，但不打断正在合成的线程。

        教学态退出、实时页切回识别时需要调用：否则冷却计时还停在教学前，
        或者恢复瞬间把教学期间攒下的结果补播一遍。
        """
        with self._lock:
            self._reset_state()

    def wait_idle(self, timeout: float | None = None) -> bool:
        """等待后台合成线程结束。

        Args:
            timeout: 最长等待秒数；``None`` 表示一直等到结束。

        Returns:
            线程已结束返回 ``True``，超时返回 ``False``。
        """
        with self._lock:
            worker = self._worker
        if worker is None:
            return True
        worker.join(timeout)
        return not worker.is_alive()

    def stats(self) -> dict[str, Any]:
        """返回节流计数快照，供验收与调试打印。"""
        with self._lock:
            return {
                "cooldown_seconds": self.cooldown_seconds,
                "last_text": self._last_text,
                "last_spoken_at": self._last_spoken_at,
                "attempts": self._attempts,
                "spoken": self._spoken,
                "failed": self._failed,
                "skipped_same": self._skipped_same,
                "skipped_cooldown": self._skipped_cooldown,
                "skipped_busy": self._skipped_busy,
                "last_error": self._last_error,
            }

    @property
    def last_error(self) -> str | None:
        """最近一次合成失败的异常类型与信息；成功后清空。"""
        with self._lock:
            return self._last_error

    # -- 主入口 ----------------------------------------------------------- #
    def announce(self, text: str) -> bool:
        """按节流规则决定是否播报这段文本，并立即返回。

        Args:
            text: 播报文本，例如 ``"识别结果：鼠标"``；空文本直接忽略。

        Returns:
            ``True`` 表示已把合成任务交给后台线程；``False`` 表示被节流
            规则或空文本拦下。调用方**不能**用返回值判断"是否真的播放成功"，
            失败信息看 :attr:`last_error`。
        """
        clean = " ".join(str(text or "").split())
        if not clean:
            return False

        now = self._clock()
        with self._lock:
            if clean == self._last_text:
                self._skipped_same += 1
                return False
            if (
                self._last_spoken_at is not None
                and now - self._last_spoken_at < self.cooldown_seconds
            ):
                self._skipped_cooldown += 1
                return False
            if self._busy:
                self._skipped_busy += 1
                return False

            self._last_text = clean
            self._last_spoken_at = now
            self._pending_text = clean
            self._attempts += 1
            self._busy = True
            worker = threading.Thread(
                target=self._run_speak,
                name="rn18-tts",
                daemon=True,
            )
            self._worker = worker

        worker.start()
        return True

    def _run_speak(self) -> None:
        """后台线程主体：合成并播放，任何异常都收成 ``last_error``。"""
        with self._lock:
            text = self._pending_text

        try:
            self._speak(text)
        except Exception as exc:  # noqa: BLE001 - 后台线程不允许抛出
            with self._lock:
                self._failed += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
                # 失败后允许下一次变化重试；冷却时间仍然生效，
                # 因此断网时最多每 4 秒试一次，不会刷屏。
                self._last_text = None
        else:
            with self._lock:
                self._spoken += 1
                self._last_error = None
        finally:
            with self._lock:
                self._busy = False


_DEFAULT_THROTTLER: TtsThrottler | None = None
_DEFAULT_LOCK = threading.Lock()


def get_throttler() -> TtsThrottler:
    """返回进程级共享的节流器（实时链路用同一个冷却窗口）。"""
    global _DEFAULT_THROTTLER
    with _DEFAULT_LOCK:
        if _DEFAULT_THROTTLER is None:
            _DEFAULT_THROTTLER = TtsThrottler()
        return _DEFAULT_THROTTLER


def announce_stream(text: str) -> bool:
    """实时链路的一行式入口：把识别文本交给共享节流器。

    Args:
        text: 播报文本，已知物体传 ``"识别结果：<名称>"``，未知传 ``"未知物体"``。

    Returns:
        同 :meth:`TtsThrottler.announce`。
    """
    return get_throttler().announce(text)


__all__ = [
    "DEFAULT_VOICE",
    "TTS_COOLDOWN_MAX_SECONDS",
    "TTS_COOLDOWN_MIN_SECONDS",
    "TTS_COOLDOWN_SECONDS",
    "TTS_UNAVAILABLE_MESSAGE",
    "TtsAudioError",
    "TtsDependencyError",
    "TtsError",
    "TtsResult",
    "TtsSynthesisError",
    "TtsThrottler",
    "announce_stream",
    "get_throttler",
    "normalize_text",
    "resolve_cooldown",
    "resolve_output_dir",
    "resolve_voice",
    "synthesize",
    "tts",
]
