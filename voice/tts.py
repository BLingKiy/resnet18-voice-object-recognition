"""Edge TTS 语音播报封装（全项目唯一的播报引擎）。

对外契约::

    from voice import tts, synthesize, speak_in_background, SpeechGate

    path = tts("识别结果：苹果")        # -> "C:/.../rn18-tts-093015-ab12cd34.mp3"
    path = synthesize("未知物体")      # 同上，返回 Path

为什么只有 Edge TTS、没有本地降级（人类 2026-11-09 已确认）：

1. 选题书要求"语音播报识别结果"，播报内容必须随当次识别结果变化；
   固定提示音不算完成，所以失败时宁可明示"语音播报不可用"，
   也不能播一段和识别结果无关的声音。
2. 不引入 pyttsx3 或第二家云厂商：多一个引擎就多一套现场依赖与
   音色差异，演示机器上任何一处缺失都会变成新的翻车点。

因此本模块的错误处理原则是"**失败就抛，由界面说人话**"：
网络断了、edge-tts 没装、引擎返回空音频，一律抛 :class:`TtsError` 子类，
调用方捕获后显示「语音播报不可用，请检查网络」，识别结果本身照常展示。

另外两点实现约定：

- **每次合成写独立文件**：Gradio 的 ``gr.Audio(autoplay=True)`` 靠组件
  值变化触发播放，同一个路径重复赋值不会再次自动播放。文件名带
  时间戳 + 随机后缀，保证"每次识别都真的播一次当次文本"。
- **合成放后台线程**：:func:`speak_in_background` 给 P5 实时链路复用，
  避免 0.3–0.5 秒的帧流被 1–2 秒的合成阻塞；:class:`SpeechGate` 实现
  "结果变化 + 冷却秒数"的节流规则（P4 静态链路不节流，每次必播）。
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger("rn18.tts")

# 微软 Edge 的中文女声；音色/语速属于"需要人类拍板"的审美项，
# 这里先按文档默认值实现，改动只需调 TtsConfig，不影响调用方。
DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"
DEFAULT_RATE = "+0%"
DEFAULT_VOLUME = "+0%"

# 临时目录里最多保留多少个播报文件：够现场连着演示很多次，
# 又不会把磁盘塞满（一个 3 秒 mp3 约 20–40 KB）。
_KEEP_FILES = 24

# 产出目录可被环境变量覆盖，便于把证据 mp3 固定到某个目录再上传。
ENV_TTS_DIR = "RN18_TTS_DIR"


class TtsError(Exception):
    """语音播报相关错误的基类，界面只捕获这一层。

    捕获范围刻意收窄成一个基类：界面不需要知道是依赖缺失还是网络抖动，
    统一显示「语音播报不可用，请检查网络」，避免把技术细节甩给现场用户。
    """


class TtsDependencyError(TtsError):
    """未安装 edge-tts（或其依赖）。

    这是**可预期的部署问题**：调用方按普通播报失败处理即可，
    识别与特征库不受影响。
    """


class TtsSynthesisError(TtsError):
    """合成失败：文本为空、网络异常、引擎报错或返回空音频。

    异常信息里不含任何密钥（Edge TTS 走公开接口，本来也没有 Key），
    可以直接展示给用户。
    """


@dataclass(frozen=True, slots=True)
class TtsConfig:
    """一次合成使用的音色与语速配置。

    Args:
        voice: Edge TTS 音色名，默认 ``zh-CN-XiaoxiaoNeural``。
        rate: 语速，``"+0%"`` 表示原速；``"+20%"`` 更快。
        volume: 音量，``"+0%"`` 表示原音量。

    Attributes:
        voice: 见上。
        rate: 见上。
        volume: 见上。
    """

    voice: str = DEFAULT_VOICE
    rate: str = DEFAULT_RATE
    volume: str = DEFAULT_VOLUME


def default_output_dir() -> Path:
    """返回播报音频的产出目录。

    默认用系统临时目录而不是仓库目录：mp3 是演示产物、可随时重建，
    放进仓库既污染 ``git status``，又容易被误提交。

    Returns:
        目录路径；``RN18_TTS_DIR`` 有值时优先用它。
    """
    override = os.environ.get(ENV_TTS_DIR, "").strip()
    if override:
        return Path(override).expanduser()
    return Path(tempfile.gettempdir()) / "rn18-tts"


def _load_edge_tts() -> Any:
    """延迟导入 edge_tts，缺依赖时转成可读的 :class:`TtsDependencyError`。

    Returns:
        已导入的 ``edge_tts`` 模块。

    Raises:
        TtsDependencyError: 未安装 edge-tts。
    """
    try:
        import edge_tts
    except ImportError as exc:
        raise TtsDependencyError(
            "缺少依赖 edge-tts。请运行 `pip install edge-tts` 后重试。"
        ) from exc
    return edge_tts


def _run_coroutine(factory: Callable[[], Coroutine[Any, Any, None]]) -> None:
    """在"当前线程没有事件循环"的前提下同步执行一个协程。

    绝大多数情况走 ``asyncio.run``；只有调用方本身已经处在事件循环里
    （例如以后把界面嵌进 Jupyter/异步服务）时才退到独立线程，避免
    ``asyncio.run() cannot be called from a running event loop``。
    Gradio 的同步事件在无事件循环的工作线程里执行，正常情况下走第一条路径。

    Args:
        factory: 返回协程的零参函数；每次调用都必须新建协程对象，
            因为它可能被执行两次（当前线程 / 独立线程）。
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(factory())
        return

    error_box: dict[str, BaseException] = {}

    def worker() -> None:
        try:
            asyncio.run(factory())
        except BaseException as exc:  # noqa: BLE001 - 原样带回调用线程再抛
            error_box["error"] = exc

    thread = threading.Thread(target=worker, name="rn18-tts-synth", daemon=True)
    thread.start()
    thread.join()
    if "error" in error_box:
        raise error_box["error"]


def _build_filename(text: str) -> str:
    """生成"可读前缀 + 时间戳 + 随机串"的唯一文件名。

    唯一性是功能需求而不是洁癖：Gradio 只有在组件值变化时才自动播放，
    复用同一个文件名会导致"第二次识别听不到声音"。

    Args:
        text: 播报文本，仅用于截取一个便于人工识别的短前缀。

    Returns:
        形如 ``rn18-识别结果苹果-093015-ab12cd34.mp3`` 的文件名。
    """
    prefix = "".join(ch for ch in text if ch.isalnum())[:12] or "speech"
    stamp = time.strftime("%H%M%S")
    return f"rn18-{prefix}-{stamp}-{uuid.uuid4().hex[:8]}.mp3"


def _remove_quietly(path: Path) -> None:
    """尽力删除半成品文件；删不掉不是致命错误，不要盖住真正的合成异常。"""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        _LOGGER.debug("清理临时音频失败: %s", path)


def _prune(directory: Path) -> None:
    """只保留最新的若干个音频文件，避免长期演示后临时目录越堆越多。

    Args:
        directory: 播报音频目录。
    """
    try:
        files = sorted(
            (item for item in directory.glob("rn18-*.mp3") if item.is_file()),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return
    for stale in files[_KEEP_FILES:]:
        _remove_quietly(stale)


def synthesize(
    text: str,
    *,
    config: TtsConfig | None = None,
    out_dir: str | Path | None = None,
) -> Path:
    """把一段文本合成为本地 mp3 文件（同步阻塞，内部跑 asyncio）。

    Args:
        text: 要播报的文本；首尾空白会被去掉。
        config: 音色/语速配置，``None`` 表示使用 :class:`TtsConfig` 默认值。
        out_dir: 产出目录，``None`` 表示 :func:`default_output_dir`。

    Returns:
        新生成的 mp3 文件路径（每次调用都是新文件）。

    Raises:
        TtsDependencyError: 未安装 edge-tts。
        TtsSynthesisError: 文本为空、网络失败、引擎报错或没有产出音频。
    """
    clean = (text or "").strip()
    if not clean:
        # 空文本不该走到界面，但真走到了也不能"静默成功"返回一个空文件。
        raise TtsSynthesisError("播报文本为空，未生成音频。")

    cfg = config or TtsConfig()
    edge_tts = _load_edge_tts()

    target_dir = Path(out_dir).expanduser() if out_dir is not None else default_output_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / _build_filename(clean)

    async def _save() -> None:
        # 只传 edge-tts 多年稳定的三个参数（text/voice/rate/volume），
        # 避免绑定某个小版本才有的可选参数。
        communicate = edge_tts.Communicate(
            clean,
            cfg.voice,
            rate=cfg.rate,
            volume=cfg.volume,
        )
        await communicate.save(str(out_path))

    try:
        _run_coroutine(_save)
    except TtsError:
        raise
    except Exception as exc:  # noqa: BLE001 - 网络/协议/引擎异常统一转可读错误
        _remove_quietly(out_path)
        # 只保留异常类型名：aiohttp 的异常文本可能很长，对现场用户没有价值。
        raise TtsSynthesisError(
            f"语音合成失败（{type(exc).__name__}），请检查网络后重试。"
        ) from exc

    if not out_path.is_file() or out_path.stat().st_size == 0:
        # 引擎"成功"但没有内容：不当作成功，否则界面会显示已播报却没声音。
        _remove_quietly(out_path)
        raise TtsSynthesisError(
            "语音合成失败：引擎没有返回音频数据，请检查网络后重试。"
        )

    _prune(target_dir)
    return out_path


def tts(text: str, *, config: TtsConfig | None = None, out_dir: str | Path | None = None) -> str:
    """项目契约入口：文本 → 音频文件路径字符串。

    与空间文档里的 ``voice/tts.py  tts(text) -> str`` 保持一致，
    界面层可以直接把返回值塞给 ``gr.Audio``。

    Args:
        text: 要播报的文本。
        config: 音色/语速配置。
        out_dir: 产出目录。

    Returns:
        mp3 文件的绝对路径字符串。

    Raises:
        TtsError: 同 :func:`synthesize`。
    """
    return str(synthesize(text, config=config, out_dir=out_dir))


def speak_in_background(
    text: str,
    *,
    on_success: Callable[[Path], None] | None = None,
    on_error: Callable[[TtsError], None] | None = None,
    config: TtsConfig | None = None,
    out_dir: str | Path | None = None,
) -> threading.Thread:
    """在后台线程里合成语音，立即返回线程对象（P5 实时链路复用）。

    为什么单独提供：实时链路每 0.3–0.5 秒出一帧识别结果，一次合成要
    1–2 秒，同步合成会把画面卡住。后台线程 + 回调让界面线程始终空闲。

    Args:
        text: 要播报的文本。
        on_success: 合成成功时在后台线程里回调，参数是 mp3 路径。
        on_error: 合成失败时在后台线程里回调，参数是 :class:`TtsError`。
            不给回调也可以，此时只记录日志——但调用方就拿不到失败提示，
            界面场景务必传 on_error。
        config: 音色/语速配置。
        out_dir: 产出目录。

    Returns:
        已启动的守护线程（可用 ``join()`` 等待，测试里就是这么做的）。
    """

    def worker() -> None:
        try:
            path = synthesize(text, config=config, out_dir=out_dir)
        except TtsError as exc:
            _LOGGER.warning("后台语音合成失败: %s", exc)
            if on_error is not None:
                try:
                    on_error(exc)
                except Exception:  # noqa: BLE001 - 回调异常不能冒到线程顶
                    _LOGGER.exception("TTS 失败回调自身抛出异常")
            return
        except Exception as exc:  # noqa: BLE001 - 防御未预期错误
            _LOGGER.exception("后台语音合成出现未预期异常")
            if on_error is not None:
                try:
                    on_error(
                        TtsSynthesisError(
                            f"语音合成失败（{type(exc).__name__}），请检查网络后重试。"
                        )
                    )
                except Exception:  # noqa: BLE001
                    _LOGGER.exception("TTS 失败回调自身抛出异常")
            return

        if on_success is not None:
            try:
                on_success(path)
            except Exception:  # noqa: BLE001
                _LOGGER.exception("TTS 成功回调自身抛出异常")

    thread = threading.Thread(target=worker, name="rn18-tts-worker", daemon=True)
    thread.start()
    return thread


class SpeechGate:
    """实时链路的播报节流器：结果变化 **且** 冷却时间已过才播。

    P4 的静态链路不用它（每次识别都必须播），它的存在是为了让 P5 直接
    复用同一套规则，而不是在界面里再写一遍：

        gate = SpeechGate(cooldown_seconds=4.0)
        if gate.should_speak(text):
            speak_in_background(text, on_error=show_hint)
        ...
        gate.reset()   # 退出教学态后清空计时，避免立刻补播旧结果

    Attributes:
        cooldown_seconds: 两次播报之间的最小间隔（秒），建议 3–5。
    """

    def __init__(
        self,
        cooldown_seconds: float = 4.0,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """初始化节流器。

        Args:
            cooldown_seconds: 最小播报间隔，必须为非负数。
            clock: 时钟函数，默认 ``time.monotonic``；测试可注入假时钟。

        Raises:
            ValueError: ``cooldown_seconds`` 为负数。
        """
        if cooldown_seconds < 0:
            raise ValueError(f"冷却时间不能为负数，收到 {cooldown_seconds}")
        self.cooldown_seconds = float(cooldown_seconds)
        self._clock = clock
        self._last_text: str | None = None
        self._last_at = float("-inf")

    def should_speak(self, text: str) -> bool:
        """判断这段文本现在是否该播。

        规则与任务定义一致：**结果变化** + **冷却已过**，两个条件同时满足。
        文本相同（同一物体连续帧）直接不播，避免"苹果、苹果、苹果"刷屏。

        Args:
            text: 本次识别的播报文本。

        Returns:
            该播报时为 True；同时会记录本次文本与时间。
        """
        clean = (text or "").strip()
        if not clean:
            return False
        if clean == self._last_text:
            return False
        now = self._clock()
        if now - self._last_at < self.cooldown_seconds:
            return False
        self._last_text = clean
        self._last_at = now
        return True

    def reset(self) -> None:
        """清空"上次播报"记录，让下一次调用必然通过冷却判断。

        P5 退出教学态时调用：教学期间可能暂停了几十秒，恢复后不该
        因为"冷却早过了"就立刻补播一条过期结果，但也不该继续压着不播，
        清空计时是最直白的做法。
        """
        self._last_text = None
        self._last_at = float("-inf")
