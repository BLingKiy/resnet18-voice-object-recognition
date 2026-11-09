"""ResNet18 + 语音物体识别 Demo 的 Gradio 界面（P2 界面 + P3 语音录名 + P4 语音播报）。

本文件只做两件事：**界面布局**与**事件绑定**。所有视觉逻辑都复用 ``core``：

- ``core.features``  —— ResNet18 特征提取（512 维、L2 归一化）
- ``core.library``   —— JSON 特征库的加载、录入、保存
- ``core.recognize`` —— 余弦相似度检索与未知物体判定

语音分两路，互不影响识别主流程：

- 语音输入（``voice.asr``）：说出物体名称 -> 回填名称框；
- 语音播报（``voice.tts``）：识别完成后播报当次结果，失败只显示
  「语音播报不可用，请检查网络」，识别结果照常返回。

启动::

    python app.py
    # 浏览器打开 http://127.0.0.1:7860

环境变量（可选，便于把演示数据放到别的目录）::

    RN18_LIBRARY_PATH  # 特征库 JSON 路径，默认 data/features.json
    RN18_HOST          # 监听地址，默认 127.0.0.1（只给本机演示用）
    RN18_PORT          # 监听端口，默认 7860
    RN18_TTS_DIR       # 播报音频输出目录，默认仓库内 .tts/
    RN18_TTS_VOICE     # 音色，默认 zh-CN-XiaoxiaoNeural
    RN18_TTS_COOLDOWN  # 实时播报冷却秒数，夹在 3~5 秒之间，默认 4

说明：判定阈值当前是核心层里的**占位值**，最终必须由 P6 用未参与录入的
测试图校准；界面会原样显示本次实际使用的阈值，不会把占位值包装成结论。
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

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
from PIL import Image, ImageDraw, ImageFont

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
    add_to_library,
    load_library,
    recognize,
    save_library,
    to_pil,
)
from core.library import DEFAULT_LIBRARY_PATH
from voice import (
    TTS_UNAVAILABLE_MESSAGE,
    AsrAudioError,
    AsrDependencyError,
    AsrError,
    AsrNotConfigured,
    BaiduAsrError,
    TtsError,
    asr,
    is_asr_configured,
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

# 常见中文字体；找不到时退回 PIL 默认字体（英文/数字仍可读，中文可能显示方框）。
_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)


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
    """丢弃内存缓存，重新从磁盘读取特征库（界面上的"刷新"按钮）。"""
    global _LIBRARY
    with _STORE_LOCK:
        _LIBRARY = load_library(LIBRARY_PATH)
        return _LIBRARY


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
    keep_name = "" if current_name is None else str(current_name)

    if not is_asr_configured():
        return keep_name, f"⚠️ {ASR_MISSING_HINT}。", _asr_button_update(False)

    if audio_path is None:
        return (
            keep_name,
            "⚠️ 请先点击录音按钮说一遍物体名称，再点「语音识别名称」。",
            _asr_button_update(True),
        )

    # 只捕获 voice 层定义的异常：它们的信息已经脱敏（不含 Key 与带
    # client_secret 的 URL），可以直接展示给现场用户。
    try:
        text = asr(audio_path)
    except AsrNotConfigured:
        # 启动后 .env 被移走 / Key 被清空：按同一套降级文案处理。
        return keep_name, f"⚠️ {ASR_MISSING_HINT}。", _asr_button_update(False)
    except AsrDependencyError as exc:
        _LOGGER.warning("语音识别缺少依赖: %s", exc)
        return (
            keep_name,
            f"❌ 语音识别缺少依赖：{exc}  \n（键盘录入仍可用）",
            _asr_button_update(True),
        )
    except AsrAudioError as exc:
        _LOGGER.warning("录音无法解码: %s", exc)
        return (
            keep_name,
            f"❌ 录音无法使用：{exc}  \n（请重新录音，或改用键盘输入）",
            _asr_button_update(True),
        )
    except BaiduAsrError as exc:
        _LOGGER.warning("百度语音识别失败: %s", exc)
        return (
            keep_name,
            f"❌ 语音识别失败：{exc}  \n（可改用键盘输入）",
            _asr_button_update(True),
        )
    except AsrError as exc:
        # 兜底：以后新增的 AsrError 子类也不会冒到 Gradio 变成红色堆栈。
        _LOGGER.warning("语音识别失败: %s", exc)
        return (
            keep_name,
            f"❌ 语音识别失败：{exc}  \n（可改用键盘输入）",
            _asr_button_update(True),
        )
    except Exception as exc:  # noqa: BLE001 - 界面事件不允许把异常抛给前端
        _LOGGER.exception("语音识别出现未预期异常")
        return (
            keep_name,
            f"❌ 语音识别异常：{type(exc).__name__}: {exc}  \n（可改用键盘输入）",
            _asr_button_update(True),
        )

    recognized = (text or "").strip()
    if not recognized:
        return (
            keep_name,
            "⚠️ 没有听清，请靠近麦克风重说一次，或直接用键盘输入。",
            _asr_button_update(True),
        )

    return (
        recognized,
        f"✅ 已识别为「{recognized}」，确认或修改后点「录入并保存」。",
        _asr_button_update(True),
    )


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
# 标注图
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=8)
def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """按字号加载中文字体；系统没有中文字体时退回 PIL 默认字体。"""
    for candidate in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 的 load_default 不接受 size
        return ImageFont.load_default()


def annotate_image(image: Any, result: Recognition) -> Image.Image:
    """在查询图上画出"识别结果 + 相似度"的标注条。

    没有目标检测模型，所以不做假边框：这里标注的是**整张图的识别结论**，
    绿框代表已知、红框代表未知，文字给出名称与百分比。这样教师在现场
    能一眼看到"输入图 -> 结论"的对应关系，也不会把检索包装成检测。

    Args:
        image: 用户上传的图片（路径 / PIL / numpy）。
        result: ``core.recognize`` 的返回值。

    Returns:
        新的 RGB ``PIL.Image``，原图不被修改。
    """
    canvas = to_pil(image).convert("RGB").copy()
    width, height = canvas.size
    draw = ImageDraw.Draw(canvas, "RGBA")

    color = (34, 139, 34, 255) if result.is_known else (200, 40, 40, 255)
    border = max(2, min(width, height) // 120)
    draw.rectangle([0, 0, width - 1, height - 1], outline=color, width=border)

    display_name = result.name if len(result.name) <= 20 else result.name[:19] + "…"
    label = f"{display_name}  {result.similarity_percent:.1f}%"

    font_size = max(16, int(min(width, height) * 0.055))
    font = _load_font(font_size)
    # 图很窄时逐步缩小字号，避免标注条把整张图盖住或文字被裁掉。
    while font_size > 12:
        text_width, text_height = draw.textbbox((0, 0), label, font=font)[2:]
        if text_width <= width * 0.9:
            break
        font_size -= 2
        font = _load_font(font_size)

    text_width, text_height = draw.textbbox((0, 0), label, font=font)[2:]
    padding = max(6, font_size // 3)
    banner_width = min(width, text_width + padding * 2)
    banner_height = min(height, text_height + padding * 2)
    draw.rectangle([0, 0, banner_width, banner_height], fill=color[:3] + (210,))
    draw.text((padding, padding), label, font=font, fill=(255, 255, 255, 255))
    return canvas


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


def _recognize_core(
    image: Any,
    *,
    library: FeatureLibrary | None = None,
    encoder: Encoder | None = None,
    top_k: int = 3,
) -> tuple[Recognition | None, str, Image.Image | None, list[list[str]]]:
    """识别一张图并返回 ``(结果对象, 结论 Markdown, 标注图, 近邻行)``。

    静态识别与"识别 + 播报"共用这一份逻辑：播报需要 ``Recognition`` 对象
    才能念出当次物体名，而界面展示只需要 Markdown。抽出来是为了避免两条
    路径各自识别一次、得出两个结果。

    失败或输入不合法时 ``结果对象`` 为 ``None``，其余字段是可展示的提示文案。
    """
    lib = library if library is not None else get_library()
    if image is None:
        return None, "⚠️ 请先上传一张测试图片。", None, []
    if lib.total_vectors == 0:
        return None, "⚠️ 特征库为空，请先在「多图录入」页录入物体。", None, []

    try:
        with _STORE_LOCK:
            result = recognize(lib, image, top_k=top_k, encoder=encoder)
        annotated = annotate_image(image, result)
    except Exception as exc:
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

    Args:
        image: 用户上传的图片（路径 / PIL / numpy）。
        library: 仅测试使用；``None`` 表示进程内共享特征库。
        encoder: 仅测试使用；``None`` 表示真实的 ResNet18。
        top_k: 展示前多少个近邻，界面固定 3。

    Returns:
        ``(结论 Markdown, 标注图或 None, 近邻表格)``。
    """
    _, headline, annotated, rows = _recognize_core(
        image, library=library, encoder=encoder, top_k=top_k
    )
    return headline, annotated, rows


# --------------------------------------------------------------------------- #
# Tab2：语音播报
# --------------------------------------------------------------------------- #
def announcement_text(result: Recognition) -> str:
    """把识别结果转成播报话术。

    Args:
        result: 一次识别结果。

    Returns:
        已知物体返回 ``"识别结果：<名称>"``，未知返回 ``"未知物体"``。
        文本必须来自**当次**识别结果，不能是固定话术或固定提示音。
    """
    return f"识别结果：{result.name}" if result.is_known else UNKNOWN_LABEL


def speak_result(result: Recognition) -> tuple[str, str | None]:
    """合成当次识别结果的播报音频，并把任何失败转成一句现场可读的提示。

    播报是**附加**能力：合成失败只影响这一行状态与播放器，绝不向上抛异常，
    调用方的识别结论、标注图、近邻表都已经生成好了。

    Args:
        result: 当次识别结果。

    Returns:
        ``(状态 Markdown, 音频路径)``。成功时路径指向刚合成的 mp3，
        ``gr.Audio(autoplay=True)`` 收到新路径即自动播放；失败时为 ``None``，
        同时清空上一次的播放器内容，避免旧音频被误认为本次播报。
    """
    text = announcement_text(result)
    try:
        spoken = synthesize(text)
    except TtsError as exc:
        # 依赖缺失与断网在界面上是同一句话：现场只关心"播不出来"，
        # 具体原因进日志，不把异常类型糊到学生脸上。
        _LOGGER.warning("语音播报失败: %s", exc)
        return f"⚠️ {TTS_UNAVAILABLE_MESSAGE}", None
    except Exception as exc:  # noqa: BLE001 - 播报不允许打断识别主流程
        _LOGGER.exception("语音播报出现未预期异常")
        return f"⚠️ {TTS_UNAVAILABLE_MESSAGE}", None

    status = (
        f"🔊 已播报：{spoken.text}  \n"
        f"合成 **{spoken.elapsed_ms} ms** · 音频 **{spoken.duration_ms / 1000:.2f} s**"
        f" · 音色 `{spoken.voice}`"
    )
    # 必须返回路径字符串：gr.Audio 的 postprocess 只认路径/元组，
    # 传 {"value": ...} 会在服务端报 "Cannot process ... as Audio"。
    return status, str(spoken.audio_path)


def recognize_and_speak(
    image: Any,
    *,
    library: FeatureLibrary | None = None,
    encoder: Encoder | None = None,
    top_k: int = 3,
) -> tuple[str, Image.Image | None, list[list[str]], str, str | None]:
    """Tab2 按钮事件：识别 + 播报当次结果。

    静态链路每次都重新合成、重新播报，不做文本去重——用户连续点两次
    「开始识别」，就该听到两次结果，节流只属于实时链路。

    Args:
        image: 用户上传的图片（路径 / PIL / numpy）。
        library: 仅测试使用；``None`` 表示进程内共享特征库。
        encoder: 仅测试使用；``None`` 表示真实的 ResNet18。
        top_k: 展示前多少个近邻，界面固定 3。

    Returns:
        ``(结论 Markdown, 标注图, 近邻表格, 播报状态, 音频路径)``。
    """
    result, headline, annotated, rows = _recognize_core(
        image, library=library, encoder=encoder, top_k=top_k
    )
    if result is None:
        # 没识别出东西就不播报：播一句"未知物体"会把"没图/空库"误导成
        # "识别成功但结果未知"。播报状态留空，识别提示保持可见。
        return headline, annotated, rows, "", None

    status, audio_path = speak_result(result)
    return headline, annotated, rows, status, audio_path


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

            tts_status = gr.Markdown(
                "识别后会语音播报当次结果（Edge TTS，需联网）。"
            )
            broadcast_audio = gr.Audio(
                label="语音播报（识别后自动播放）",
                type="filepath",
                autoplay=True,
                interactive=False,
            )

            neighbor_table = gr.Dataframe(
                headers=NEIGHBOR_HEADERS,
                datatype=["str", "str", "str", "str", "str"],
                value=[],
                interactive=False,
                wrap=True,
                label="前 3 个近邻（图片级检索）",
            )

            # 识别与播报在同一次事件里完成：先出结果，再合成音频。
            # 播报失败只更新状态行，不影响上面已经返回的识别结果。
            recognize_button.click(
                recognize_and_speak,
                inputs=[query_image],
                outputs=[
                    headline,
                    annotated_image,
                    neighbor_table,
                    tts_status,
                    broadcast_audio,
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
