"""ResNet18 + 语音物体识别 Demo 的 Gradio 界面（P2 界面 + P3 语音录名 + P4 播报）。

本文件只做两件事：**界面布局**与**事件绑定**。所有视觉逻辑都复用 ``core``：

- ``core.features``  —— ResNet18 特征提取（512 维、L2 归一化）
- ``core.library``   —— JSON 特征库的加载、录入、保存
- ``core.recognize`` —— 余弦相似度检索与未知物体判定

语音分两条独立链路，互不依赖：

1. **语音输入（P3）**：麦克风说话 -> 百度短语音识别 -> 回填名称框，
   没有配置百度密钥时按钮置灰，键盘录入路径保持原样可用。
2. **语音播报（P4）**：识别完成后用 Edge TTS 合成"当次识别结果"，
   输出到 ``gr.Audio(autoplay=True)`` 自动播放。每次识别都重新合成，
   文本随识别结果变化；合成失败时界面明示「语音播报不可用，请检查网络」，
   识别结果本身照常显示，绝不改播固定话术或提示音。

启动::

    python app.py
    # 浏览器打开 http://127.0.0.1:7860

环境变量（可选，便于把演示数据放到别的目录）::

    RN18_LIBRARY_PATH  # 特征库 JSON 路径，默认 data/features.json
    RN18_HOST          # 监听地址，默认 127.0.0.1（只给本机演示用）
    RN18_PORT          # 监听端口，默认 7860

说明：判定阈值当前是核心层里的**占位值**，最终必须由 P6 用未参与录入的
测试图校准；界面会原样显示本次实际使用的阈值，不会把占位值包装成结论。
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# 关掉 Gradio 的联网统计：教室/演示现场经常没有外网，后台请求会卡住线程。
# 必须在 import gradio 之前设置，Gradio 在导入时读取该变量。
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

# 本机演示不应走系统代理：有些 Windows 机器配了 HTTP 代理但没有把
# 127.0.0.1 放进绕过列表，Gradio 启动时自检 localhost 会超时并误报"无法访问"。
_local_bypass = {
    item.strip() for item in os.environ.get("NO_PROXY", "").split(",") if item.strip()
}
_local_bypass.update({"127.0.0.1", "localhost", "::1"})
os.environ["NO_PROXY"] = ",".join(sorted(_local_bypass))
os.environ["no_proxy"] = os.environ["NO_PROXY"]

import gradio as gr
from PIL import Image

# 允许从任意目录执行 ``python app.py``；Gradio 的工作目录不一定是仓库根目录。
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import (
    UNKNOWN_LABEL,
    AddResult,
    Encoder,
    FeatureLibrary,
    Recognition,
    StreamResult,
    add_to_library,
    annotate_image,
    load_library,
    recognize_static,
    recognize_stream,
    resolve_threshold,
    save_library,
    speech_text as _pipeline_speech_text,
    to_pil,
)
from core.library import DEFAULT_LIBRARY_PATH
from voice import (
    AsrAudioError,
    AsrDependencyError,
    AsrError,
    AsrNotConfigured,
    BaiduAsrError,
    SpeechGate,
    TtsDependencyError,
    TtsError,
    TtsSynthesisError,
    asr,
    is_asr_configured,
    speak_in_background,
    synthesize,
)

_LOGGER = logging.getLogger("rn18.app")

# 单个进程、单个特征库：Gradio 的事件在独立线程里跑，用可重入锁把
# "读库 -> 编码 -> 追加 -> 写盘" 串起来，避免连点按钮写出半份特征库。
_STORE_LOCK = threading.RLock()
_LIBRARY: FeatureLibrary | None = None

LIBRARY_PATH = Path(
    os.environ.get("RN18_LIBRARY_PATH", str(DEFAULT_LIBRARY_PATH))
).expanduser()
if not LIBRARY_PATH.is_absolute():
    LIBRARY_PATH = (PROJECT_ROOT / LIBRARY_PATH).resolve()

LIBRARY_HEADERS = ["物体名称", "特征条数", "首张注册图", "录入时间"]
NEIGHBOR_HEADERS = ["排名", "物体", "相似度", "相似度%", "最近注册图"]

# 语音降级的文案只写一份：启动时按钮提示和点击后的状态提示必须口径一致，
# 否则现场很容易出现"按钮说没配 Key、点下去又说网络错误"的自相矛盾。
ASR_MISSING_HINT = "未配置语音识别密钥，请用键盘输入"
ASR_READY_HINT = (
    "录音后点「语音识别名称」，识别结果会回填到上方名称框，确认或修改后再录入。"
)

# 播报文案同样只写一份：界面初始提示、成功提示、失败提示必须口径一致。
TTS_IDLE_HINT = "识别成功后会自动播报当次识别结果（Edge TTS）。"
TTS_UNAVAILABLE_HINT = "⚠️ 语音播报不可用，请检查网络"

# 实时链路参数（任务要求：帧流 0.3–0.5 s；教学帧间隔 0.2–0.3 s；TTS 冷却 3–5 s）。
REALTIME_STREAM_EVERY = 0.3
TEACHING_FRAME_COUNT = 5
TEACHING_MIN_GAP = 0.2
TTS_COOLDOWN_SECONDS = 4.0

REALTIME_IDLE_HINT = "打开摄像头，或上传预置帧序列后点「播放帧序列」。"
TEACHING_IDLE_HINT = "未开始教学。点「教它」后抓 5 帧，再说出物体名称。"


# --------------------------------------------------------------------------- #
# 特征库状态
# --------------------------------------------------------------------------- #
def get_library() -> FeatureLibrary:
    """返回进程内共享的特征库；首次调用时从 JSON 加载。

    重启后能继续使用，靠的就是这里：库文件存在就加载，不存在才建空库。
    加载失败（文件损坏 / schema 不匹配）会直接抛出 ``ValueError``，不静默
    覆盖磁盘上的文件，避免把用户数据弄丢。

    Returns:
        已绑定 ``LIBRARY_PATH`` 的 ``FeatureLibrary``。
    """
    global _LIBRARY
    with _STORE_LOCK:
        if _LIBRARY is None:
            _LIBRARY = load_library(LIBRARY_PATH)
        return _LIBRARY


def reload_library() -> FeatureLibrary:
    """丢弃内存缓存，重新从磁盘读取特征库（界面上的"刷新"按钮）。

    实时控制器如果已经存在，只替换它绑定的库对象，不清空教学状态：
    用户可能只是想确认"重启后还在"，不该因此把正在进行的教学打断。
    """
    global _LIBRARY
    with _STORE_LOCK:
        loaded = load_library(LIBRARY_PATH)
        _LIBRARY = loaded
    if _REALTIME is not None:
        _REALTIME.library = loaded
    return loaded


def library_rows(library: FeatureLibrary) -> list[list[str]]:
    """把特征库统计转成 ``gr.Dataframe`` 需要的二维字符串表格。

    Args:
        library: 当前特征库。

    Returns:
        每行 ``[名称, 条数, 首张图, 录入时间]``，按物体名称排序。
    """
    return [
        [row["name"], str(row["count"]), row["first_image"], row["created_at"]]
        for row in library.stats()
    ]


def library_summary(library: FeatureLibrary) -> str:
    """生成特征库概况的一行 Markdown。"""
    if library.total_vectors == 0:
        return "**特征库为空**：先在左边输入名称、上传多张图片，然后点「录入并保存」。"
    location = library.path if library.path is not None else LIBRARY_PATH
    return (
        f"**{library.object_count} 个物体 · {library.total_vectors} 条特征**  \n"
        f"特征库文件：`{location}`"
    )


# --------------------------------------------------------------------------- #
# Tab1：多图录入
# --------------------------------------------------------------------------- #
def _file_paths(files: Sequence[Any] | str | Path | None) -> list[Path]:
    """把 Gradio 的 ``gr.File`` 返回值统一成路径列表。

    Gradio 4 在 ``type="filepath"`` 时给的是 ``str`` 或 ``list[str]``；
    这里同时兼容 ``Path`` 与带 ``.name`` 的临时文件对象，避免升级小版本后
    事件函数因入参类型变化而报错。
    """
    if files is None:
        return []
    if isinstance(files, (str, Path)):
        return [Path(files)]

    paths: list[Path] = []
    for item in files:
        if isinstance(item, (str, Path)):
            paths.append(Path(item))
        elif hasattr(item, "name"):
            paths.append(Path(str(item.name)))
    return paths


def _asr_button_update(configured: bool) -> dict[str, Any]:
    """构造「语音识别名称」按钮的组件更新。

    Gradio 4 允许事件返回普通 ``dict`` 作为组件更新，等价于
    ``gr.update(...)``；用 dict 是为了让界面事件能脱离 Gradio 单测。

    Args:
        configured: 当前是否具备可用的 ASR 配置。

    Returns:
        ``{"interactive": configured}``。
    """
    return {"interactive": configured}


def _spoken_name(
    audio_path: str | Path | None,
    current_name: str = "",
) -> tuple[str, str, bool, bool]:
    """调用一次百度 ASR，并把结果/错误统一成界面可直接展示的四元组。

    Tab1 与 Tab3 教学态共用这个函数：错误文案、Key 缺失降级、空识别处理
    只写一份，避免两个入口对同一段音频给出不同结论。成功时 ``message``
    为空串，由调用方补"下一步点哪个按钮"，因为两个页面的按钮不同。

    Args:
        audio_path: ``gr.Audio(type="filepath")`` 给出的录音路径；未录音为 ``None``。
        current_name: 名称框当前内容，失败时原样保留。

    Returns:
        ``(名称框内容, 状态 Markdown, 是否已配置密钥, 是否识别成功)``。
    """
    keep_name = "" if current_name is None else str(current_name)

    if not is_asr_configured():
        return keep_name, f"⚠️ {ASR_MISSING_HINT}。", False, False

    if audio_path is None:
        return keep_name, "⚠️ 请先点击录音按钮说一遍物体名称。", True, False

    # 只捕获 voice 层定义的异常：它们的信息已经脱敏（不含 Key 与带
    # client_secret 的 URL），可以直接展示给现场用户。
    try:
        text = asr(audio_path)
    except AsrNotConfigured:
        # 启动后 .env 被移走 / Key 被清空：按同一套降级文案处理。
        return keep_name, f"⚠️ {ASR_MISSING_HINT}。", False, False
    except AsrDependencyError as exc:
        _LOGGER.warning("语音识别缺少依赖: %s", exc)
        return keep_name, f"❌ 语音识别缺少依赖：{exc}  \n（键盘录入仍可用）", True, False
    except AsrAudioError as exc:
        _LOGGER.warning("录音无法解码: %s", exc)
        return (
            keep_name,
            f"❌ 录音无法使用：{exc}  \n（请重新录音，或改用键盘输入）",
            True,
            False,
        )
    except BaiduAsrError as exc:
        _LOGGER.warning("百度语音识别失败: %s", exc)
        return keep_name, f"❌ 语音识别失败：{exc}  \n（可改用键盘输入）", True, False
    except AsrError as exc:
        # 兜底：以后新增的 AsrError 子类也不会冒到 Gradio 变成红色堆栈。
        _LOGGER.warning("语音识别失败: %s", exc)
        return keep_name, f"❌ 语音识别失败：{exc}  \n（可改用键盘输入）", True, False
    except Exception as exc:  # noqa: BLE001 - 界面事件不允许把异常抛给前端
        _LOGGER.exception("语音识别出现未预期异常")
        return (
            keep_name,
            f"❌ 语音识别异常：{type(exc).__name__}: {exc}  \n（可改用键盘输入）",
            True,
            False,
        )

    recognized = (text or "").strip()
    if not recognized:
        return (
            keep_name,
            "⚠️ 没有听清，请靠近麦克风重说一次，或直接用键盘输入。",
            True,
            False,
        )
    return recognized, "", True, True


def transcribe_name(
    audio_path: str | Path | None,
    current_name: str = "",
) -> tuple[str, str, dict[str, Any]]:
    """Tab1 事件：把麦克风录音识别成物体名称并回填输入框。

    设计成"名称框进、名称框出"而不是返回 ``gr.update``：失败时把用户
    已经输入的名称原样返回，避免一次识别失败就把键盘输入的内容清空。

    未配置密钥时只走降级分支，绝不调用 :func:`voice.asr`，因此不会联网、
    不会抛异常；键盘录入路径完全不受影响。

    Args:
        audio_path: ``gr.Audio(type="filepath")`` 给出的录音文件路径；
            未录音时为 ``None``。
        current_name: 名称框当前内容，失败时原样保留。

    Returns:
        ``(名称框内容, 状态 Markdown, 语音按钮更新)``。识别成功时名称框
        是识别文本，失败时保持 ``current_name`` 不变。
    """
    name, message, configured, ok = _spoken_name(audio_path, current_name)
    if ok:
        message = f"✅ 已识别为「{name}」，确认或修改后点「录入并保存」。"
    return name, message, _asr_button_update(configured)


def enroll_images(
    name: str,
    files: Sequence[Any] | str | Path | None,
    *,
    library: FeatureLibrary | None = None,
    encoder: Encoder | None = None,
) -> tuple[str, list[list[str]], str]:
    """Tab1 事件：录入一个物体的多张注册图并立即落盘。

    Args:
        name: 键盘输入的物体名称，首尾空白会被去掉。
        files: 上传的一张或多张图片（路径或 Gradio 文件对象）。
        library: 仅测试使用；``None`` 表示进程内共享特征库。
        encoder: 仅测试使用；``None`` 表示真实的 ResNet18。

    Returns:
        ``(状态 Markdown, 特征库表格, 特征库概况)``。
    """
    lib = library if library is not None else get_library()
    rows_before = library_rows(lib)
    summary_before = library_summary(lib)

    clean_name = (name or "").strip()
    if not clean_name:
        return "⚠️ 请先输入物体名称。", rows_before, summary_before

    paths = _file_paths(files)
    if not paths:
        return "⚠️ 请至少上传一张注册图片。", rows_before, summary_before

    try:
        with _STORE_LOCK:
            result: AddResult = add_to_library(lib, clean_name, paths, encoder=encoder)
            if result.added > 0:
                save_library(lib)
    except Exception as exc:  # 界面必须把错误说人话，而不是只留一个红色堆栈
        _LOGGER.exception("录入失败: name=%r, files=%r", clean_name, paths)
        return f"❌ 录入失败：{type(exc).__name__}: {exc}", rows_before, summary_before

    if result.added > 0:
        status = (
            f"✅ 已录入「{result.object_name}」：新增 **{result.added}** 条特征，"
            f"跳过重复 **{result.skipped}** 张。  \n"
            f"该物体 **{result.object_vectors}** 条 / 全库 **{result.total_vectors}** 条，"
            "已写入特征库。"
        )
    else:
        status = (
            f"⚠️ 「{result.object_name}」没有新增特征：{result.skipped} 张图都已在库中"
            "（同一物体内按像素摘要去重）。"
        )
    return status, library_rows(lib), library_summary(lib)


def refresh_library_view() -> tuple[str, list[list[str]], str]:
    """Tab1 事件：重新读取磁盘上的特征库，验证"重启后仍在"。"""
    try:
        lib = reload_library()
    except Exception as exc:
        _LOGGER.exception("重新加载特征库失败")
        return f"❌ 重新加载失败：{type(exc).__name__}: {exc}", [], ""
    return "🔄 已从磁盘重新加载特征库。", library_rows(lib), library_summary(lib)


# --------------------------------------------------------------------------- #
# Tab2：单图识别
# --------------------------------------------------------------------------- #
def _percent(similarity: float) -> str:
    """把余弦相似度映射成与 ``Recognition.similarity_percent`` 同口径的百分比。"""
    return f"{max(0.0, min(1.0, float(similarity))) * 100.0:.2f}%"


def format_headline(result: Recognition) -> str:
    """生成识别结论的 Markdown（名称、相似度、是否未知）。"""
    if result.is_known:
        headline = (
            f"### 识别结果：{result.name}\n\n"
            f"**相似度：{result.similarity:.4f}（{result.similarity_percent:.2f}%）**"
        )
    else:
        headline = (
            f"### 识别结果：{UNKNOWN_LABEL}\n\n"
            f"最高相似度 **{result.similarity:.4f}"
            f"（{result.similarity_percent:.2f}%）** 低于判定阈值 "
            f"**{result.threshold:.4f}**，判为未知物体。"
        )

    # 阈值是占位值这件事必须显式写在界面上，避免被当成已校准结论。
    headline += (
        f"\n\n*判定阈值：{result.threshold:.4f}"
        "（占位值，最终由 P6 用未参与录入的测试图校准）*"
    )
    return headline


def neighbor_rows(result: Recognition) -> list[list[str]]:
    """把前 K 个近邻转成表格行。"""
    return [
        [
            str(match.rank),
            match.name,
            f"{match.similarity:.4f}",
            _percent(match.similarity),
            match.image,
        ]
        for match in result.matches
    ]


def _analyze(
    image: Any,
    *,
    library: FeatureLibrary | None,
    encoder: Encoder | None,
    top_k: int,
) -> tuple[Recognition | None, str, Image.Image | None, list[list[str]]]:
    """识别 + 标注的公共前半段，静态识别与"识别并播报"共用。

    抽出来的唯一原因是：播报文本必须来自**当次** ``Recognition`` 对象，
    而 ``recognize_image`` 只返回拼好的 Markdown。若为了拿结果再识别一次，
    既浪费一次前向计算，也可能出现"图和结论对不上"的竞态。

    Args:
        image: 用户上传的图片（路径 / PIL / numpy）。
        library: 仅测试使用；``None`` 表示进程内共享特征库。
        encoder: 仅测试使用；``None`` 表示真实的 ResNet18。
        top_k: 展示前多少个近邻，界面固定 3。

    Returns:
        ``(识别结果或 None, 结论 Markdown 或错误提示, 标注图或 None, 近邻表格)``。
        识别结果与标注图同时为 None 表示没有可播报的内容。
    """
    lib = library if library is not None else get_library()
    if image is None:
        return None, "⚠️ 请先上传一张测试图片。", None, []
    if lib.total_vectors == 0:
        return None, "⚠️ 特征库为空，请先在「多图录入」页录入物体。", None, []

    try:
        with _STORE_LOCK:
            result = recognize_static(image, lib, top_k=top_k, encoder=encoder)
        annotated = annotate_image(image, result)
    except Exception as exc:  # 界面必须把错误说人话，而不是留一个红色堆栈
        _LOGGER.exception("识别失败")
        return None, f"❌ 识别失败：{type(exc).__name__}: {exc}", None, []

    return result, format_headline(result), annotated, neighbor_rows(result)


def recognize_image(
    image: Any,
    *,
    library: FeatureLibrary | None = None,
    encoder: Encoder | None = None,
    top_k: int = 3,
) -> tuple[str, Image.Image | None, list[list[str]]]:
    """Tab2 事件：识别一张图片并生成标注图与 Top-K 近邻表（不播报）。

    保留这个三返回值签名是为了让不关心语音的调用方（命令行、旧测试）
    继续工作；界面按钮改用 :func:`recognize_and_speak`。

    Args:
        image: 用户上传的图片（路径 / PIL / numpy）。
        library: 仅测试使用；``None`` 表示进程内共享特征库。
        encoder: 仅测试使用；``None`` 表示真实的 ResNet18。
        top_k: 展示前多少个近邻，界面固定 3。

    Returns:
        ``(结论 Markdown, 标注图或 None, 近邻表格)``。
    """
    _, headline, annotated, rows = _analyze(
        image, library=library, encoder=encoder, top_k=top_k
    )
    return headline, annotated, rows


def speech_text(result: Recognition) -> str:
    """把**当次**识别结果转成播报文本（兼容旧调用点，实现在核心层）。

    已知物体报「识别结果：苹果」，未知物体报「未知物体」——与界面上的
    结论文案同源（都用 ``UNKNOWN_LABEL``），因此不会出现"界面显示未知、
    语音却说苹果"的不一致。

    Args:
        result: 本次识别结果。

    Returns:
        送给 Edge TTS 的文本。
    """
    return _pipeline_speech_text(result)


def speak_result(
    result: Recognition,
    *,
    synthesizer: Callable[[str], str | Path] | None = None,
) -> tuple[str | None, str]:
    """合成当次识别结果的播报音频，返回 ``(音频路径, 状态 Markdown)``。

    失败时返回 ``(None, 「语音播报不可用，请检查网络」)``：音频组件清空，
    不播放任何替代声音，识别结果不受影响。

    Args:
        result: 本次识别结果。
        synthesizer: 仅测试使用；``None`` 表示真实的 Edge TTS。

    Returns:
        ``(mp3 路径或 None, 给用户看的状态 Markdown)``。
    """
    text = speech_text(result)
    # 在调用点读模块全局，而不是写死默认参数：测试可以直接 monkeypatch
    # ``app.synthesize`` 而不需要重新 import 模块。
    synth = synthesizer if synthesizer is not None else synthesize
    try:
        path = synth(text)
    except TtsDependencyError as exc:
        _LOGGER.warning("语音播报缺少依赖: %s", exc)
        return None, f"{TTS_UNAVAILABLE_HINT}。  \n（{exc}）"
    except TtsSynthesisError as exc:
        _LOGGER.warning("语音合成失败: %s", exc)
        return None, f"{TTS_UNAVAILABLE_HINT}。  \n（{exc}）"
    except TtsError as exc:
        # 兜底：以后新增的 TtsError 子类也不会冒到 Gradio 变成红色堆栈。
        _LOGGER.warning("语音播报失败: %s", exc)
        return None, f"{TTS_UNAVAILABLE_HINT}。  \n（{exc}）"
    except Exception as exc:  # noqa: BLE001 - 播报失败绝不能中断识别结果展示
        _LOGGER.exception("语音播报出现未预期异常")
        return None, f"{TTS_UNAVAILABLE_HINT}。  \n（{type(exc).__name__}: {exc}）"

    return str(path), f"🔊 正在播报：{text}"


def recognize_and_speak(
    image: Any,
    *,
    library: FeatureLibrary | None = None,
    encoder: Encoder | None = None,
    top_k: int = 3,
    synthesizer: Callable[[str], str | Path] | None = None,
) -> tuple[str, Image.Image | None, list[list[str]], str | None, str]:
    """Tab2 事件：识别 + 标注 + 用当次识别文本合成播报音频。

    Args:
        image: 用户上传的图片（路径 / PIL / numpy）。
        library: 仅测试使用；``None`` 表示进程内共享特征库。
        encoder: 仅测试使用；``None`` 表示真实的 ResNet18。
        top_k: 展示前多少个近邻，界面固定 3。
        synthesizer: 仅测试使用；``None`` 表示真实的 Edge TTS。

    Returns:
        ``(结论 Markdown, 标注图, 近邻表格, 播报音频路径或 None, 播报状态)``。
    """
    result, headline, annotated, rows = _analyze(
        image, library=library, encoder=encoder, top_k=top_k
    )
    if result is None:
        # 没有识别结果就没有可播报的文本：不合成、不放提示音。
        return headline, annotated, rows, None, TTS_IDLE_HINT

    audio_path, status = speak_result(result, synthesizer=synthesizer)
    return headline, annotated, rows, audio_path, status


# --------------------------------------------------------------------------- #
# Tab3：实时识别 + 教学式录入
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class FrameOutcome:
    """实时页一次帧处理的可显示结果（不依赖 Gradio，便于脱离界面测试）。

    Attributes:
        frame: 已叠加名称/相似度的 RGB 数组；``None`` 表示本帧不刷新画面
            （教学态或错误），界面应保留上一帧。
        headline: 结论 Markdown。
        stats: 本帧耗时、阈值与库容量。
        audio: 最近一次合成完成的播报 mp3 路径；没有则 ``None``。
        tts_status: 播报状态提示。
        progress: 教学进度提示。
        gate_spoken: 本帧是否真的触发了播报（供证据脚本与测试断言）。
    """

    frame: np.ndarray | None
    headline: str
    stats: str
    audio: str | None
    tts_status: str
    progress: str
    gate_spoken: bool = False


def _looks_unusable(frame: Image.Image) -> str:
    """检查抓帧入口的画面是否明显过黑/过曝。

    摄像头首帧常是黑帧或过曝是已知风险；与其把 5 张废帧写进特征库，
    不如在这里直接拒绝并提示用户调整。返回值是错误文案，空串表示可用。

    Args:
        frame: 已转 RGB 的画面。

    Returns:
        不可用时返回中文提示，否则返回 ``""``。
    """
    width, height = frame.size
    if width < 32 or height < 32:
        return "画面太小（小于 32×32），无法抓帧。"
    luminance = np.asarray(frame.convert("L"), dtype=np.float32)
    mean = float(luminance.mean())
    if mean < 8.0:
        return "画面过黑（可能是摄像头首帧），请等画面稳定后再点「教它」。"
    if mean > 250.0:
        return "画面过曝，请调整光照后再点「教它」。"
    return ""


class RealtimeController:
    """实时页状态机：持续识别 / 教学抓帧 / 防回声 / TTS 节流。

    为什么把状态机放在 ``app.py`` 而不是塞进 Gradio 回调：

    - Gradio 的 ``stream`` 事件和按钮事件在不同线程里跑，状态必须由一个
      带锁的对象统一管理，否则教学态很容易被下一帧 stream 回调覆盖画面；
    - 这个类不 import gradio，测试和证据脚本可以直接驱动它，验证"暂停识别、
      暂停 TTS、取消不写库"等判据，不需要真的开浏览器与摄像头。

    防回声规则（任务第三节）：
    - :meth:`begin_teaching` 同时置 ``teaching`` / ``recognition_paused`` /
      ``tts_paused``，并清空待播音频；
    - :meth:`_end_teaching` 恢复三个状态并 ``SpeechGate.reset()``，冷却计时清零，
      不会在教学结束后立刻补播教学期间积压的旧结果。

    Args:
        library: 进程内共享特征库；教学入库复用 ``add_to_library`` + ``save_library``。
        encoder: 自定义编码器（测试注入假编码器）；``None`` 表示 ResNet18。
        gate: 播报节流器；``None`` 表示 4 秒冷却的 :class:`SpeechGate`。
        reloader: 入库写盘失败时重新从磁盘加载库的回调；``None`` 表示不恢复。
        clock: 抓帧间隔使用的时钟，测试可注入假时钟。
        count: 一次教学抓取的帧数，固定 5。
        min_gap: 相邻教学帧的最小间隔（秒），0.2–0.3 之间。
    """

    def __init__(
        self,
        library: FeatureLibrary,
        *,
        encoder: Encoder | None = None,
        gate: SpeechGate | None = None,
        reloader: Callable[[], FeatureLibrary] | None = None,
        clock: Callable[[], float] = time.monotonic,
        count: int = TEACHING_FRAME_COUNT,
        min_gap: float = TEACHING_MIN_GAP,
        min_interval: float = REALTIME_STREAM_EVERY,
    ) -> None:
        if count < 1:
            raise ValueError("count 必须 >= 1")
        if min_gap < 0:
            raise ValueError("min_gap 不能为负数")
        if min_interval < 0:
            raise ValueError("min_interval 不能为负数")
        self.library = library
        self.encoder = encoder
        self._gate = gate if gate is not None else SpeechGate(TTS_COOLDOWN_SECONDS)
        self._reloader = reloader
        self._clock = clock
        self.count = int(count)
        self.min_gap = float(min_gap)
        self.min_interval = float(min_interval)

        self._lock = threading.RLock()
        self.teaching = False
        self.recognition_paused = False
        self.tts_paused = False

        self._capture: list[Image.Image] = []
        self._capture_times: list[float] = []
        self._last_capture_at = float("-inf")
        self._last_frame: Image.Image | None = None
        self._audio_path: str | None = None
        self._tts_status = REALTIME_IDLE_HINT
        self._last_elapsed_ms: float | None = None
        self._last_result: Recognition | None = None
        self._last_recognized_at = float("-inf")
        self._last_headline = REALTIME_IDLE_HINT

    # ------------------------------------------------------------------ 只读
    @property
    def captured_count(self) -> int:
        """本次教学已抓到的帧数。"""
        with self._lock:
            return len(self._capture)

    @property
    def capture_intervals(self) -> list[float]:
        """相邻教学帧的实测间隔（秒），供证据脚本核对 0.2–0.3 s。"""
        with self._lock:
            return [
                round(self._capture_times[index] - self._capture_times[index - 1], 3)
                for index in range(1, len(self._capture_times))
            ]

    @property
    def gate(self) -> SpeechGate:
        """当前使用的播报节流器（证据脚本读取其内部计时）。"""
        return self._gate

    @property
    def audio_path(self) -> str | None:
        """最近一次合成完成的播报路径；没有则为 ``None``。"""
        with self._lock:
            return self._audio_path

    @property
    def tts_status(self) -> str:
        """当前播报状态提示。"""
        with self._lock:
            return self._tts_status

    @property
    def last_result(self) -> Recognition | None:
        """最近一帧的识别结果；还没识别过则为 ``None``。"""
        with self._lock:
            return self._last_result

    @property
    def last_elapsed_ms(self) -> float | None:
        """最近一帧 ``recognize_stream`` 的耗时（毫秒）。"""
        with self._lock:
            return self._last_elapsed_ms

    # ------------------------------------------------------------------ 文案
    def stats_text(self) -> str:
        """本帧耗时、阈值与库容量的一行 Markdown。"""
        with self._lock:
            threshold = resolve_threshold(self.library)
            elapsed = (
                f"{self._last_elapsed_ms:.1f} ms"
                if self._last_elapsed_ms is not None
                else "—"
            )
            state = "⏸ 教学态（识别与播报已暂停）" if self.teaching else "▶ 持续识别"
            objects = self.library.object_count
            vectors = self.library.total_vectors
        return (
            f"{state} · 本帧 **{elapsed}**（含特征提取与标注）· "
            f"阈值 {threshold:.2f} · 库 **{objects}** 物体 / "
            f"**{vectors}** 条"
        )

    def teaching_progress(self) -> str:
        """教学抓帧进度文案。"""
        with self._lock:
            if not self.teaching:
                return TEACHING_IDLE_HINT
            done = len(self._capture)
        if done >= self.count:
            return (
                f"✅ 已抓 **{self.count}/{self.count}** 帧（间隔 ≥ {self.min_gap:.2f} s）。"
                "请录音并点「识别名称」，确认后点「确认入库」。"
            )
        return (
            f"📸 已抓 **{done}/{self.count}** 帧，请让物体保持不动 "
            f"（帧间隔 ≥ {self.min_gap:.2f} s）。"
        )

    # ------------------------------------------------------------------ 帧流
    def process(self, frame: Any) -> FrameOutcome:
        """处理一帧：教学态只抓帧；否则识别、标注并按冷却播报。

        任何异常都在这里收口成可显示文案，绝不冒到 Gradio 变成红色堆栈，
        因此 Tab3 的故障不会连累 Tab1/Tab2。

        Args:
            frame: 摄像头帧 / 预置帧序列的一帧（numpy / PIL / 路径）。

        Returns:
            :class:`FrameOutcome`；``frame is None`` 表示不要覆盖上一帧。
        """
        if frame is None:
            return FrameOutcome(
                None,
                "等待画面……请打开摄像头，或上传预置帧序列后点「播放帧序列」。",
                self.stats_text(),
                self._audio_path,
                self._tts_status,
                self.teaching_progress(),
            )

        try:
            pil = to_pil(frame)
        except Exception as exc:  # noqa: BLE001 - 单帧坏了不能中断整条流
            _LOGGER.warning("实时帧解码失败: %s", exc)
            return FrameOutcome(
                None,
                f"❌ 这一帧无法解码：{type(exc).__name__}: {exc}",
                self.stats_text(),
                self._audio_path,
                self._tts_status,
                self.teaching_progress(),
            )

        with self._lock:
            self._last_frame = pil
            if self.teaching:
                self._offer_frame_locked(pil)
                return FrameOutcome(
                    None,
                    "⏸ 教学态：已暂停识别与语音播报，正在抓帧。",
                    self.stats_text(),
                    self._audio_path,
                    self._tts_status,
                    self.teaching_progress(),
                )

            if self.recognition_paused:
                return FrameOutcome(
                    None,
                    "⏸ 识别已暂停。",
                    self.stats_text(),
                    self._audio_path,
                    self._tts_status,
                    self.teaching_progress(),
                )

            # Gradio 4.44 的 webcam 流由前端按自身节奏推帧，服务端再按
            # 0.3 s 节流：只处理"距上一帧已满 min_interval"的帧，保证
            # recognize_stream 的实际调用频率落在任务要求的 0.3–0.5 s。
            now = self._clock()
            if now - self._last_recognized_at < self.min_interval:
                return FrameOutcome(
                    None,
                    self._last_headline,
                    self.stats_text(),
                    self._audio_path,
                    self._tts_status,
                    self.teaching_progress(),
                )
            self._last_recognized_at = now

            try:
                stream: StreamResult = recognize_stream(
                    pil, self.library, encoder=self.encoder
                )
            except Exception as exc:  # noqa: BLE001 - 界面只展示可读错误
                _LOGGER.exception("实时识别失败")
                return FrameOutcome(
                    None,
                    f"❌ 实时识别失败：{type(exc).__name__}: {exc}",
                    self.stats_text(),
                    self._audio_path,
                    self._tts_status,
                    self.teaching_progress(),
                )

            self._last_elapsed_ms = stream.elapsed_ms
            self._last_result = stream.recognition
            self._last_headline = format_headline(stream.recognition)
            spoken = False
            if not self.tts_paused and self._gate.should_speak(stream.speech):
                self._speak_locked(stream.speech)
                spoken = True

            return FrameOutcome(
                stream.annotated,
                self._last_headline,
                self.stats_text(),
                self._audio_path,
                self._tts_status,
                self.teaching_progress(),
                gate_spoken=spoken,
            )

    # ------------------------------------------------------------ 教学：抓帧
    def begin_teaching(self, frame: Any = None) -> tuple[bool, str]:
        """进入教学态：暂停识别与 TTS、清空待播音频、抓下当前帧。

        Args:
            frame: 当前画面；``None`` 时退回最近一帧（摄像头刚推来的）。

        Returns:
            ``(是否进入成功, 给用户看的状态文案)``。
        """
        with self._lock:
            if self.teaching:
                return False, "⚠️ 教学已在进行中，请先「确认入库」或「取消教学」。"

            source = frame if frame is not None else self._last_frame
            if source is None:
                return False, "⚠️ 还没有画面：请先打开摄像头，或播放预置帧序列。"
            try:
                pil = to_pil(source)
            except Exception as exc:  # noqa: BLE001
                return False, f"❌ 当前画面无法使用：{type(exc).__name__}: {exc}"

            problem = _looks_unusable(pil)
            if problem:
                return False, f"⚠️ {problem}"

            self.teaching = True
            self.recognition_paused = True
            self.tts_paused = True
            self._capture = [pil]
            self._capture_times = [self._clock()]
            self._last_capture_at = self._capture_times[0]
            # 清空待播音频：教学态麦克风正在录音，绝不能同时播报旧结果（防回声）。
            self._audio_path = None
            self._tts_status = "⏸ 教学态已暂停语音播报，避免录到播报音。"
            self._gate.reset()

        return (
            True,
            f"✅ 已进入教学态：识别与 TTS 已暂停，开始抓帧（1/{self.count}）。",
        )

    def _offer_frame_locked(self, pil: Image.Image) -> bool:
        """在教学态追加一帧（调用方必须已持有 ``self._lock``）。"""
        if len(self._capture) >= self.count:
            return False
        now = self._clock()
        if self._capture and now - self._last_capture_at < self.min_gap:
            return False
        self._capture.append(pil)
        self._capture_times.append(now)
        self._last_capture_at = now
        return True

    def offer_frame(self, frame: Any) -> bool:
        """把摄像头/帧序列送来的一帧追加进教学抓帧缓冲。

        Args:
            frame: 一帧画面（numpy / PIL / 路径）。

        Returns:
            ``True`` 表示这次真的收下了（距离上一帧已满 ``min_gap``）。
        """
        if frame is None:
            return False
        try:
            pil = to_pil(frame)
        except Exception:  # noqa: BLE001 - 坏帧直接丢弃，不污染抓帧
            return False
        with self._lock:
            return self._offer_frame_locked(pil)

    # ------------------------------------------------------------ 教学：ASR
    def transcribe(
        self, audio_path: str | Path | None, current_name: str = ""
    ) -> tuple[str, str, str]:
        """教学态调用一次百度 ASR，把文本回填到名称框。

        只有教学动作会调用 ASR；持续识别状态不会走这里。失败时保留用户
        已输入的名称，键盘兜底路径不受影响。

        Args:
            audio_path: ``gr.Audio`` 的录音文件路径。
            current_name: 名称框当前内容。

        Returns:
            ``(名称框内容, 状态 Markdown, 教学进度)``。
        """
        with self._lock:
            if not self.teaching:
                return (
                    current_name or "",
                    "⚠️ 请先点「教它」进入教学态，再录制物体名称。",
                    self.teaching_progress(),
                )
            name, message, configured, ok = _spoken_name(audio_path, current_name)
            progress = self.teaching_progress()

        if ok:
            return (
                name,
                f"✅ 已回填「{name}」，确认或键盘修改后点「确认入库」。",
                progress,
            )
        if configured:
            message += "  \n（也可以直接在名称框输入，键盘兜底不受影响）"
        return name, message, progress

    # ------------------------------------------------------------ 教学：入库
    def commit_teaching(self, name: str) -> tuple[str, str]:
        """确认名称，把 5 帧逐条追加到特征库并持久化。

        复用 ``add_to_library`` 与 ``save_library``，不新写入库逻辑。
        写盘失败时尝试重新加载磁盘版本，避免内存里留下"磁盘上没有"的半条记录。

        Args:
            name: 确认框里的物体名称（可来自 ASR，也可来自键盘）。

        Returns:
            ``(状态 Markdown, 特征库概况 Markdown)``。
        """
        with self._lock:
            if not self.teaching:
                return "⚠️ 当前不在教学态。", library_summary(self.library)
            frames = list(self._capture)
            if len(frames) < self.count:
                return (
                    f"⚠️ 目前只抓到 {len(frames)}/{self.count} 帧，"
                    "请让物体保持不动，等抓满后再确认。",
                    library_summary(self.library),
                )

            clean_name = (name or "").strip()
            if not clean_name:
                return (
                    "⚠️ 名称为空：请先点「识别名称」回填，或直接用键盘输入。",
                    library_summary(self.library),
                )

            stamp = time.strftime("%Y%m%d-%H%M%S")
            labels = [
                f"teach-{clean_name}-{stamp}-{index + 1:02d}.jpg"
                for index in range(len(frames))
            ]
            before_vectors = self.library.total_vectors
            before_objects = self.library.object_count

            try:
                with _STORE_LOCK:
                    result: AddResult = add_to_library(
                        self.library,
                        clean_name,
                        frames,
                        labels=labels,
                        encoder=self.encoder,
                    )
                    if result.added > 0:
                        save_library(self.library)
            except Exception as exc:  # noqa: BLE001 - 失败必须说人话并保住磁盘数据
                _LOGGER.exception("教学入库失败: name=%r", clean_name)
                self._recover_library()
                self._end_teaching_locked()
                return (
                    f"❌ 入库失败：{type(exc).__name__}: {exc}  \n"
                    f"特征库未写入；当前仍为 {self.library.total_vectors} 条。",
                    library_summary(self.library),
                )

            after_vectors = self.library.total_vectors
            self._end_teaching_locked()
            if result.added < len(frames):
                note = (
                    f"⚠️ 新增 {result.added} 条，{len(frames) - result.added} 帧"
                    "因与库中图片重复被跳过。抓帧时请换角度或距离。"
                )
            else:
                note = (
                    f"✅ 已入库「{result.object_name}」：新增 **{result.added}** 条特征，"
                    f"该物体 {result.object_vectors} 条 / 全库 "
                    f"{before_vectors} → **{after_vectors}** 条"
                    f"（{before_objects} → {self.library.object_count} 个物体），"
                    "已写入特征库。"
                )
            return note, library_summary(self.library)

    def cancel_teaching(self) -> tuple[str, str, str]:
        """取消教学：丢弃抓帧，绝不调用入库函数。

        Returns:
            ``(状态 Markdown, 特征库概况, 教学进度)``。
        """
        with self._lock:
            if not self.teaching:
                return (
                    "⚠️ 当前不在教学态，没有可取消的抓帧。",
                    library_summary(self.library),
                    self.teaching_progress(),
                )
            discarded = len(self._capture)
            self._end_teaching_locked()
            return (
                f"已取消教学：丢弃 {discarded} 帧，特征库仍为 "
                f"{self.library.object_count} 个物体 / {self.library.total_vectors} 条"
                "（未写入任何记录）。",
                library_summary(self.library),
                self.teaching_progress(),
            )

    def _end_teaching_locked(self) -> None:
        """退出教学态并恢复持续识别（调用方必须持有锁）。"""
        self.teaching = False
        self.recognition_paused = False
        self.tts_paused = False
        self._capture.clear()
        self._capture_times.clear()
        self._last_capture_at = float("-inf")
        self._last_recognized_at = float("-inf")
        self._tts_status = TTS_IDLE_HINT
        # 恢复时清零冷却：教学期间可能暂停了几十秒，不清零会立刻补播旧结果。
        self._gate.reset()

    def _recover_library(self) -> None:
        """写盘失败后尽量回到磁盘版本，避免内存比磁盘多出半条记录。"""
        if self._reloader is None:
            return
        try:
            self.library = self._reloader()
        except Exception:  # noqa: BLE001 - 恢复失败只能记录，不能盖住原始错误
            _LOGGER.exception("入库失败后重新加载特征库也失败了")

    # ------------------------------------------------------------------ TTS
    def _speak_locked(self, text: str) -> None:
        """后台合成播报（调用方必须持有锁）。

        合成放后台线程：0.3 s 一帧的流不能被 1–2 s 的网络合成阻塞。
        """

        def on_success(path: Path) -> None:
            with self._lock:
                self._audio_path = str(path)
                self._tts_status = f"🔊 正在播报：{text}"

        def on_error(exc: TtsError) -> None:
            with self._lock:
                self._tts_status = f"{TTS_UNAVAILABLE_HINT}。  \n（{exc}）"

        try:
            speak_in_background(text, on_success=on_success, on_error=on_error)
        except Exception as exc:  # noqa: BLE001 - 起线程失败不能中断识别
            _LOGGER.exception("启动后台播报失败")
            self._tts_status = f"{TTS_UNAVAILABLE_HINT}。  \n（{type(exc).__name__}: {exc}）"


_REALTIME: RealtimeController | None = None


def _reload_into_global() -> FeatureLibrary:
    """从磁盘重新加载特征库并替换进程内共享对象。"""
    global _LIBRARY
    with _STORE_LOCK:
        _LIBRARY = load_library(LIBRARY_PATH)
        return _LIBRARY


def get_realtime(library: FeatureLibrary | None = None) -> RealtimeController:
    """返回进程内共享的实时控制器；特征库对象换了会自动重建。

    Args:
        library: 显式指定特征库（测试用）；``None`` 表示共享库。

    Returns:
        与当前特征库绑定的 :class:`RealtimeController`。
    """
    global _REALTIME
    target = library if library is not None else get_library()
    with _STORE_LOCK:
        if _REALTIME is None or _REALTIME.library is not target:
            _REALTIME = RealtimeController(target, reloader=_reload_into_global)
        return _REALTIME


def realtime_frame(
    frame: Any,
) -> tuple[Any, str, str, str | None, str, str]:
    """Tab3 摄像头流事件：一帧进、画面/结论/统计/播报/进度出。

    Args:
        frame: ``gr.Image(streaming=True, type="numpy")`` 推来的帧。

    Returns:
        ``(标注帧或 gr.update(), 结论, 统计, 播报音频, 播报状态, 教学进度)``。
        ``gr.update()`` 表示本帧不覆盖上一帧（教学/暂停/错误）。
    """
    outcome = get_realtime().process(frame)
    return (
        outcome.frame if outcome.frame is not None else gr.update(),
        outcome.headline,
        outcome.stats,
        outcome.audio,
        outcome.tts_status,
        outcome.progress,
    )


def play_frame_sequence(
    files: Sequence[Any] | str | Path | None,
) -> Iterator[tuple[Any, str, str, str | None, str, str]]:
    """Tab3 无摄像头兜底：把上传的图片按顺序当作视频帧播放。

    每帧之间等待 ``REALTIME_STREAM_EVERY`` 秒，与摄像头 ``stream_every``
    一致；教学态下这些帧同样会进入抓帧缓冲，因此没有摄像头也能完整走通
    "教它 -> 抓 5 帧 -> ASR -> 确认"的闭环。

    Args:
        files: ``gr.File(file_count="multiple")`` 的返回值。

    Yields:
        与 :func:`realtime_frame` 相同的六元组。
    """
    paths = _file_paths(files)
    if not paths:
        controller = get_realtime()
        yield (
            gr.update(),
            "⚠️ 请先上传预置帧序列（多张图片，按文件名顺序播放）。",
            controller.stats_text(),
            controller.audio_path,
            controller.tts_status,
            controller.teaching_progress(),
        )
        return

    controller = get_realtime()
    for index, path in enumerate(paths):
        try:
            with Image.open(path) as opened:
                frame = np.asarray(opened.convert("RGB"))
        except Exception as exc:  # noqa: BLE001 - 坏图跳过，不打断整段序列
            _LOGGER.warning("预置帧无法读取: %s", exc)
            yield (
                gr.update(),
                f"❌ 第 {index + 1} 帧无法读取：{type(exc).__name__}: {exc}",
                controller.stats_text(),
                controller.audio_path,
                controller.tts_status,
                controller.teaching_progress(),
            )
            continue

        outcome = controller.process(frame)
        yield (
            outcome.frame if outcome.frame is not None else gr.update(),
            outcome.headline,
            outcome.stats,
            outcome.audio,
            outcome.tts_status,
            outcome.progress,
        )
        time.sleep(REALTIME_STREAM_EVERY)


def teach_from_sequence(
    files: Sequence[Any] | str | Path | None,
) -> Iterator[tuple[str, str, str, Any, str, Any]]:
    """无摄像头兜底：从预置帧序列抓 5 帧进入教学态。

    摄像头现场演示应直接点「教它」；本函数只解决"这台机器没有摄像头，
    但需要把教学闭环完整走通"的问题。抓帧间隔同样是 0.3 s，且第 1 帧
    与后续帧来自不同文件，不会出现"5 张相邻帧几乎相同"的虚高问题。

    Args:
        files: ``gr.File(file_count="multiple")`` 的图片列表。

    Yields:
        与 :func:`start_teaching` 相同的六元组，逐帧刷新进度。
    """
    controller = get_realtime()
    paths = _file_paths(files)
    if not paths:
        yield (
            "⚠️ 请先上传至少 5 张预置帧。",
            controller.teaching_progress(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )
        return

    try:
        with Image.open(paths[0]) as opened:
            first = np.asarray(opened.convert("RGB"))
    except Exception as exc:  # noqa: BLE001
        yield (
            f"❌ 第 1 帧无法读取：{type(exc).__name__}: {exc}",
            controller.teaching_progress(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )
        return

    ok, message = controller.begin_teaching(first)
    yield (
        message,
        controller.teaching_progress(),
        "⏸ 教学态：已暂停识别与播报。" if ok else gr.update(),
        None if ok else gr.update(),
        "" if ok else gr.update(),
        None if ok else gr.update(),
    )
    if not ok:
        return

    for path in paths[1:]:
        if controller.captured_count >= controller.count:
            break
        # 帧间 0.3 s（任务要求 0.2–0.3 s，且禁止取相邻帧）。
        time.sleep(REALTIME_STREAM_EVERY)
        try:
            with Image.open(path) as opened:
                frame = np.asarray(opened.convert("RGB"))
        except Exception:  # noqa: BLE001 - 坏帧跳过，继续抓
            continue
        controller.offer_frame(frame)
        yield (
            "📸 教学态抓帧中……",
            controller.teaching_progress(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )


def start_teaching(frame: Any) -> tuple[str, str, str, Any, str, Any]:
    """Tab3「教它」事件：进入教学态并暂停识别/播报。

    Args:
        frame: 摄像头当前帧（可能为 ``None``，此时用控制器缓存的最近一帧）。

    Returns:
        ``(教学状态, 抓帧进度, 实时结论, 播报音频, 名称框, 录音组件)``。
        播报音频返回 ``None`` 会立即停止正在播放的声音（防回声）。
    """
    ok, message = get_realtime().begin_teaching(frame)
    controller = get_realtime()
    if not ok:
        return (
            message,
            controller.teaching_progress(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )
    return (
        message,
        controller.teaching_progress(),
        "⏸ 教学态：已暂停识别与播报。",
        None,
        "",
        None,
    )


def teaching_transcribe(
    audio_path: str | Path | None, current_name: str = ""
) -> tuple[str, str, str]:
    """Tab3「识别名称」事件：录音 -> 百度 ASR -> 回填名称框。"""
    return get_realtime().transcribe(audio_path, current_name)


def teaching_confirm(
    name: str,
) -> tuple[str, str, str, str, Any, str, Any, str]:
    """Tab3「确认入库」事件：5 帧逐条入库并持久化，随后恢复持续识别。"""
    controller = get_realtime()
    status, summary = controller.commit_teaching(name)
    return (
        status,
        summary,
        controller.teaching_progress(),
        "",
        None,
        "▶ 已恢复持续识别，等待下一帧……",
        None,
        TTS_IDLE_HINT,
    )


def teaching_cancel() -> tuple[str, str, str, str, Any, str, Any, str]:
    """Tab3「取消教学」事件：丢弃抓帧，不产生任何库记录。"""
    controller = get_realtime()
    status, summary, progress = controller.cancel_teaching()
    return (
        status,
        summary,
        progress,
        "",
        None,
        "▶ 已恢复持续识别，等待下一帧……",
        None,
        TTS_IDLE_HINT,
    )


# --------------------------------------------------------------------------- #
# 界面
# --------------------------------------------------------------------------- #
def build_demo() -> gr.Blocks:
    """构建 Gradio Blocks 界面（不启动服务，便于测试与复用）。"""
    library = get_library()
    # 只读 .env / 环境变量判断配置，不发任何网络请求：百度 token 接口
    # 在教室网络下可能很慢，绝不能拖住启动。
    asr_configured = is_asr_configured()
    _LOGGER.info("语音识别配置：%s", "已就绪" if asr_configured else "缺失（按钮置灰）")

    with gr.Blocks(title="ResNet18 物体识别 Demo") as demo:
        gr.Markdown(
            "# ResNet18 物体识别 Demo\n"
            "本地 ImageNet 预训练 ResNet18（512 维特征 + 余弦相似度），"
            "支持多图录入、单图识别与未知物体判断。"
        )

        with gr.Tab("1 · 多图录入"):
            with gr.Row():
                with gr.Column(scale=1):
                    name_input = gr.Textbox(
                        label="物体名称（键盘输入或语音识别回填）",
                        placeholder="例如：苹果",
                        max_lines=1,
                    )
                    audio_input = gr.Audio(
                        label="语音说出物体名称（麦克风录音）",
                        sources=["microphone"],
                        type="filepath",
                        format="wav",
                    )
                    asr_button = gr.Button(
                        "语音识别名称",
                        interactive=asr_configured,
                    )
                    asr_status = gr.Markdown(
                        ASR_READY_HINT
                        if asr_configured
                        else f"⚠️ {ASR_MISSING_HINT}。"
                    )
                    file_input = gr.File(
                        label="注册图片（可多选，建议同一物体 5 张以上）",
                        file_count="multiple",
                        file_types=["image"],
                        type="filepath",
                    )
                    with gr.Row():
                        enroll_button = gr.Button("录入并保存", variant="primary")
                        refresh_button = gr.Button("刷新特征库")

                with gr.Column(scale=1):
                    enroll_status = gr.Markdown("等待录入……")
                    library_total = gr.Markdown(library_summary(library))
                    library_table = gr.Dataframe(
                        headers=LIBRARY_HEADERS,
                        datatype=["str", "str", "str", "str"],
                        value=library_rows(library),
                        interactive=False,
                        wrap=True,
                        label="特征库（每个物体的特征条数）",
                    )

            # 语音只负责"把说出的名称回填到文本框"，入库仍走 enroll_images；
            # 未配置时按钮在启动阶段就被置灰，事件里还有一层同样的降级判断。
            asr_button.click(
                transcribe_name,
                inputs=[audio_input, name_input],
                outputs=[name_input, asr_status, asr_button],
            )
            enroll_button.click(
                enroll_images,
                inputs=[name_input, file_input],
                outputs=[enroll_status, library_table, library_total],
            )
            refresh_button.click(
                refresh_library_view,
                inputs=None,
                outputs=[enroll_status, library_table, library_total],
            )

        with gr.Tab("2 · 单图识别"):
            with gr.Row():
                with gr.Column(scale=1):
                    query_image = gr.Image(
                        label="测试图片（上传一张）",
                        type="filepath",
                        sources=["upload"],
                        height=340,
                    )
                    recognize_button = gr.Button("开始识别", variant="primary")
                with gr.Column(scale=1):
                    headline = gr.Markdown("上传一张图片后点击「开始识别」。")
                    annotated_image = gr.Image(
                        label="标注图（绿=已知，红=未知）",
                        height=340,
                        interactive=False,
                    )
                    # 播报播放器：识别事件每成功一次就换一个新路径，
                    # Gradio 才能因"值变化"再次触发 autoplay。
                    speech_audio = gr.Audio(
                        label="语音播报（自动播放）",
                        type="filepath",
                        autoplay=True,
                        interactive=False,
                    )
                    speech_status = gr.Markdown(TTS_IDLE_HINT)

            neighbor_table = gr.Dataframe(
                headers=NEIGHBOR_HEADERS,
                datatype=["str", "str", "str", "str", "str"],
                value=[],
                interactive=False,
                wrap=True,
                label="前 3 个近邻（图片级检索）",
            )

            recognize_button.click(
                recognize_and_speak,
                inputs=[query_image],
                outputs=[
                    headline,
                    annotated_image,
                    neighbor_table,
                    speech_audio,
                    speech_status,
                ],
            )

        with gr.Tab("3 · 实时识别"):
            gr.Markdown(
                "**摄像头实时识别（加分项）**：打开摄像头后画面按 "
                f"{REALTIME_STREAM_EVERY:.1f} s/帧送入 ResNet18，实时叠加名称与相似度，"
                "未知显示「未知物体」；播报按「结果变化 + 4 s 冷却」节流。  \n"
                "没有摄像头时，可上传一段预置帧序列（多张图片）模拟视频流，"
                "或用它完成教学抓帧。"
            )
            realtime = get_realtime(library)
            with gr.Row():
                with gr.Column(scale=1):
                    camera = gr.Image(
                        label="摄像头（流式帧，服务端 0.3 s 节流）",
                        sources=["webcam"],
                        type="numpy",
                        streaming=True,
                        height=300,
                    )
                    frame_files = gr.File(
                        label="预置帧序列（无摄像头时使用，按文件名顺序播放）",
                        file_count="multiple",
                        file_types=["image"],
                        type="filepath",
                    )
                    with gr.Row():
                        play_button = gr.Button("播放帧序列（0.3 s/帧）")
                        teach_sequence_button = gr.Button("用帧序列抓 5 帧")
                    teach_button = gr.Button(
                        "教它（抓 5 帧，暂停识别与播报）", variant="primary"
                    )

                with gr.Column(scale=1):
                    live_image = gr.Image(
                        label="识别画面（名称 + 相似度）",
                        height=300,
                        interactive=False,
                    )
                    live_headline = gr.Markdown(REALTIME_IDLE_HINT)
                    live_stats = gr.Markdown(realtime.stats_text())
                    # 实时播报：与 Tab2 独立的播放器，靠路径变化触发 autoplay。
                    live_audio = gr.Audio(
                        label="实时播报（自动播放）",
                        type="filepath",
                        autoplay=True,
                        interactive=False,
                    )
                    live_tts_status = gr.Markdown(TTS_IDLE_HINT)

            with gr.Accordion("教学式录入：说名称 → 识别 → 确认", open=True):
                with gr.Row():
                    with gr.Column(scale=1):
                        teach_audio = gr.Audio(
                            label="录音：说出物体名称（教学态）",
                            sources=["microphone"],
                            type="filepath",
                            format="wav",
                        )
                        teach_asr_button = gr.Button(
                            "识别名称", interactive=asr_configured
                        )
                        teach_name = gr.Textbox(
                            label="物体名称（ASR 回填，可键盘修改）",
                            placeholder="例如：鼠标",
                            max_lines=1,
                        )
                        with gr.Row():
                            teach_confirm_button = gr.Button(
                                "确认入库", variant="primary"
                            )
                            teach_cancel_button = gr.Button("取消教学")
                    with gr.Column(scale=1):
                        teach_status = gr.Markdown("未开始教学。点「教它」进入教学态。")
                        teach_progress = gr.Markdown(TEACHING_IDLE_HINT)
                        teach_summary = gr.Markdown(library_summary(library))

            # 摄像头流：教学态下 process() 只抓帧、不识别，返回 gr.update()
            # 保留上一帧，避免教学画面被识别结果覆盖（任务风险清单第 3 条）。
            camera.stream(
                realtime_frame,
                inputs=[camera],
                outputs=[
                    live_image,
                    live_headline,
                    live_stats,
                    live_audio,
                    live_tts_status,
                    teach_progress,
                ],
                show_progress="hidden",
            )
            play_button.click(
                play_frame_sequence,
                inputs=[frame_files],
                outputs=[
                    live_image,
                    live_headline,
                    live_stats,
                    live_audio,
                    live_tts_status,
                    teach_progress,
                ],
                show_progress="hidden",
            )
            teach_sequence_button.click(
                teach_from_sequence,
                inputs=[frame_files],
                outputs=[
                    teach_status,
                    teach_progress,
                    live_headline,
                    live_audio,
                    teach_name,
                    teach_audio,
                ],
                show_progress="hidden",
            )
            teach_button.click(
                start_teaching,
                inputs=[camera],
                outputs=[
                    teach_status,
                    teach_progress,
                    live_headline,
                    live_audio,
                    teach_name,
                    teach_audio,
                ],
            )
            teach_asr_button.click(
                teaching_transcribe,
                inputs=[teach_audio, teach_name],
                outputs=[teach_name, teach_status, teach_progress],
            )
            teach_confirm_button.click(
                teaching_confirm,
                inputs=[teach_name],
                outputs=[
                    teach_status,
                    teach_summary,
                    teach_progress,
                    teach_name,
                    teach_audio,
                    live_headline,
                    live_audio,
                    live_tts_status,
                ],
            )
            teach_cancel_button.click(
                teaching_cancel,
                inputs=None,
                outputs=[
                    teach_status,
                    teach_summary,
                    teach_progress,
                    teach_name,
                    teach_audio,
                    live_headline,
                    live_audio,
                    live_tts_status,
                ],
            )

        gr.Markdown(
            "---\n"
            "**数据口径**：特征库逐条保存每张注册图的 512 维向量（不取均值），"
            "检索时与全库所有向量比对取最高分；相似度低于阈值即判「未知物体」。  \n"
            "**注意**：当前阈值是占位值，最终必须用未参与录入的测试图校准；"
            "注册图与测试图必须严格分开。"
        )

    return demo


def main() -> None:
    """启动本地 Gradio 服务（单进程，默认 http://127.0.0.1:7860）。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _LOGGER.info("特征库路径: %s", LIBRARY_PATH)

    demo = build_demo()
    # 单进程串行处理事件：演示时不追求并发，避免两个请求同时改特征库。
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=os.environ.get("RN18_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("RN18_PORT", "7860")),
        share=False,
        inbrowser=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
