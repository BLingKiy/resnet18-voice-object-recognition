"""识别流水线：静态单图与实时帧流共用同一套"特征 -> 检索 -> 判未知 -> 标注"。

为什么单独抽一层：

- Tab2（上传单图）与 Tab3（摄像头帧流）如果各自写一遍"识别 + 画标注"，
  两边迟早会在阈值、top-k、未知文案上出现分歧，现场会出现"同一张图两个
  页面结论不一样"。这里把顺序固定成一条链，界面只负责传图和展示。
- 实时链路需要**按帧计时**。把 ``perf_counter`` 放在这一层，界面拿到的
  ``elapsed_ms`` 就是"一帧从输入到可显示"的真实耗时，而不是把 Gradio
  的排队时间也算进去。
- 特征提取与检索本身仍完全复用 :mod:`core.features` 与 :mod:`core.recognize`，
  本模块不复制任何向量计算逻辑，只做编排、计时与绘制。

对外契约::

    from core.pipeline import recognize_static, recognize_stream

    result = recognize_static("test.jpg", library)          # Recognition
    frame = recognize_stream(camera_frame, library)         # StreamResult
    frame.recognition.name, frame.elapsed_ms, frame.annotated
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from .background import BackgroundModel
from .features import Encoder, ImageInput, extract_feature, to_pil
from .library import FeatureLibrary
from .recognize import UNKNOWN_LABEL, Recognition, recognize, resolve_threshold

# 背景门命中时的展示文案。界面、标注图与测试都引用这一个常量，
# 避免三处各写一遍中文、现场对不上。
BACKGROUND_LABEL = "画面中没有可识别物体"

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
        result: :func:`core.recognize.recognize` 的返回值。

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


def speech_text(result: Recognition) -> str:
    """把**当次**识别结果转成播报文本。

    已知物体报「识别结果：苹果」，未知物体报「未知物体」——与界面上的
    结论文案同源（都用 ``UNKNOWN_LABEL``），因此不会出现"界面显示未知、
    语音却说苹果"的不一致。实时链路把它交给 ``SpeechGate`` 做节流。

    Args:
        result: 本次识别结果。

    Returns:
        送给 Edge TTS 的文本。
    """
    if result.is_known:
        return f"识别结果：{result.name}"
    return UNKNOWN_LABEL


@dataclass(frozen=True, slots=True)
class FrameScore:
    """一帧画面的原始打分结果（尚未做背景门/裁剪采用决策）。

    这层单独抽出来，是为了让 ``tools/realtime_check.py`` 能拿到"未经过阈值
    拦截"的原始分数：如果直接调 ``recognize_stream``，背景门命中时相似度会
    被写成 0，校准脚本就看不到真实的误报分，阈值也就无从谈起。

    Attributes:
        full: 整帧检索与判未知结果。
        crop: 中心 60% 裁剪后的结果；没有背景模型时为 ``None``。
        background_similarity: 整帧与背景向量的余弦；没有背景模型时为 ``None``。
        threshold: 本次实际使用的物体判定阈值（T_rt 或库内静态值）。
    """

    full: Recognition
    crop: Recognition | None
    background_similarity: float | None
    threshold: float


def center_crop(image: ImageInput, fraction: float = 0.6) -> Image.Image:
    """裁出画面中心的正方形/矩形区域（默认 60%，即 PIL 的 0.2~0.8）。

    为什么用中心裁剪而不是检测框：本项目不引入目标检测模型，中心裁剪是
    "用户把物体放在画面中央"这一演示约定下最便宜的第二路视角；物体占比
    小时，裁掉四周背景能让 512 维特征更聚焦在物体上。

    Args:
        image: 任意支持的图像输入。
        fraction: 保留比例，``0 < fraction <= 1``，默认 0.6。

    Returns:
        新的 RGB ``PIL.Image``，原图不被修改。

    Raises:
        ValueError: ``fraction`` 不在合法范围。
    """
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError(f"fraction 必须在 (0, 1] 内，收到 {fraction!r}")

    pil = to_pil(image)
    width, height = pil.size
    box_width = max(1, round(width * float(fraction)))
    box_height = max(1, round(height * float(fraction)))
    left = max(0, (width - box_width) // 2)
    top = max(0, (height - box_height) // 2)
    return pil.crop((left, top, left + box_width, top + box_height))


def score_frame(
    frame: ImageInput,
    library: FeatureLibrary,
    *,
    threshold: float | None = None,
    encoder: Encoder | None = None,
    background: BackgroundModel | None = None,
    crop_margin: float = 0.05,
) -> FrameScore:
    """给一帧画面打原始分：全帧、中心裁剪、背景相似度。

    全帧特征只提取一次，背景相似度直接复用这个向量，避免同一帧跑两遍
    ResNet18。``background is None`` 时退化成 rc1 的单路识别，不做裁剪，
    这样老调用路径的分数与标注完全不变。

    Args:
        frame: 一帧画面（numpy / PIL / 路径）。
        library: 特征库。
        threshold: 显式物体阈值；``None`` 时用库内校准值/默认值。
        encoder: 自定义编码器；默认 ResNet18。
        background: 背景模型；``None`` 表示不做背景相关计算。
        crop_margin: 仅用于校验；实际采用决策在 :func:`select_recognition`。

    Returns:
        :class:`FrameScore`。

    Raises:
        ValueError: 帧为空、裁剪参数非法或背景/特征维度不匹配。
    """
    if frame is None:
        raise ValueError("帧为空，无法识别")
    if float(crop_margin) < 0.0:
        raise ValueError(f"crop_margin 不能为负数，收到 {crop_margin!r}")

    if background is None:
        full = recognize(library, frame, threshold=threshold, top_k=1, encoder=encoder)
        return FrameScore(
            full=full,
            crop=None,
            background_similarity=None,
            threshold=full.threshold,
        )

    pil = to_pil(frame)
    full_vector = extract_feature(pil, encoder=encoder)
    # 传一维向量给 recognize，避免为了"全帧结果"再提取一次特征。
    full = recognize(library, full_vector, threshold=threshold, top_k=1)
    background_similarity = background.similarity_from_vector(full_vector)
    crop = recognize(
        library,
        center_crop(pil),
        threshold=threshold,
        top_k=1,
        encoder=encoder,
    )
    return FrameScore(
        full=full,
        crop=crop,
        background_similarity=background_similarity,
        threshold=full.threshold,
    )


def select_recognition(
    score: FrameScore,
    crop_margin: float = 0.05,
) -> tuple[Recognition, bool]:
    """按"裁剪必须明显更好且自身过阈值"的规则选最终结果。

    Args:
        score: :func:`score_frame` 的打分结果。
        crop_margin: 裁剪分数需要超过全帧分数的最小增量。

    Returns:
        ``(采用的 Recognition, 是否采用了裁剪)``。
    """
    if score.crop is None:
        return score.full, False
    if float(crop_margin) < 0.0:
        raise ValueError(f"crop_margin 不能为负数，收到 {crop_margin!r}")

    # 两个条件缺一不可：
    # 1) crop 比 full 高出 margin，说明"裁掉背景后确实更像库里的物体"；
    # 2) crop 自身过阈值，避免把一个仍然判未知的结果硬抬成已知。
    use_crop = (
        score.crop.similarity > score.full.similarity + float(crop_margin)
        and score.crop.similarity >= score.threshold
    )
    return (score.crop if use_crop else score.full), use_crop


def annotate_background(
    image: ImageInput,
    background_similarity: float,
) -> Image.Image:
    """给"背景门命中"的帧画标注条。

    与 :func:`annotate_image` 分开实现，是为了保证普通识别路径的绘制流程
    一个像素都不变（rc1 的标注图字节级等价），同时让背景帧有明确文案。

    Args:
        image: 原始帧。
        background_similarity: 该帧与背景的余弦相似度。

    Returns:
        新的 RGB ``PIL.Image``。
    """
    canvas = to_pil(image).convert("RGB").copy()
    width, height = canvas.size
    draw = ImageDraw.Draw(canvas, "RGBA")

    # 灰色边框 + 灰色标注条：与"已知绿色 / 未知红色"区分开，
    # 表示"这一帧根本没进入物体检索"。
    color = (90, 90, 90, 255)
    border = max(2, min(width, height) // 120)
    draw.rectangle([0, 0, width - 1, height - 1], outline=color, width=border)

    label = BACKGROUND_LABEL
    percent = max(0.0, min(1.0, float(background_similarity))) * 100.0
    detail = f"背景相似度 {percent:.1f}%"

    font_size = max(16, int(min(width, height) * 0.055))
    font = _load_font(font_size)
    while font_size > 12:
        text_width = draw.textbbox((0, 0), label, font=font)[2]
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

    # 背景相似度放在第二行；画面太矮时省略，避免盖住整帧。
    small_size = max(12, font_size // 2)
    if height >= banner_height + small_size * 2:
        small = _load_font(small_size)
        draw.text(
            (padding, banner_height + 4),
            detail,
            font=small,
            fill=(255, 255, 255, 235),
        )
    return canvas


@dataclass(frozen=True, slots=True)
class StreamResult:
    """一帧实时画面的完整产出，界面只需要这个对象。

    Attributes:
        recognition: 检索与判未知结果；背景门命中时 ``name`` 为
            :data:`BACKGROUND_LABEL`。
        annotated: 已叠加名称/相似度的 RGB ``uint8`` 数组，可直接给 Gradio。
        elapsed_ms: 本帧"识别 + 标注"的墙钟耗时（毫秒）。
        speech: 该帧对应的播报文本；背景门命中时为空串，**不触发 TTS**。
        is_background: 本帧是否被背景门判定为"没有可识别物体"。
        bg_similarity: 本帧与背景向量的余弦；没有背景模型时为 ``None``。
        crop_similarity: 中心裁剪结果的最高分；没有计算裁剪时为 ``None``。
        used_crop: 最终结果是否来自中心裁剪。
        threshold: 本帧实际使用的物体判定阈值。
    """

    recognition: Recognition
    annotated: NDArray[np.uint8]
    elapsed_ms: float
    speech: str
    is_background: bool = False
    bg_similarity: float | None = None
    crop_similarity: float | None = None
    used_crop: bool = False
    threshold: float = 0.0

    @property
    def is_known(self) -> bool:
        """本帧是否识别为已知物体（背景门命中时为 ``False``）。"""
        return self.recognition.is_known

    @property
    def name(self) -> str:
        """本帧展示的名称（未知时为「未知物体」，背景时为背景文案）。"""
        return self.recognition.name


def recognize_static(
    image: ImageInput,
    library: FeatureLibrary,
    *,
    threshold: float | None = None,
    top_k: int = 3,
    encoder: Encoder | None = None,
) -> Recognition:
    """静态单图识别：一次特征提取 + 全库检索 + 判未知。

    Args:
        image: 图片路径 / PIL / numpy。
        library: 特征库。
        threshold: 显式阈值；``None`` 时用库内校准值或默认值。
        top_k: 返回的近邻条数，界面展示 3 条。
        encoder: 自定义编码器；默认 ResNet18。

    Returns:
        ``Recognition``；未知时 ``name == "未知物体"``。
    """
    return recognize(library, image, threshold=threshold, top_k=top_k, encoder=encoder)


def recognize_stream(
    frame: ImageInput,
    library: FeatureLibrary,
    *,
    threshold: float | None = None,
    encoder: Encoder | None = None,
    background: BackgroundModel | None = None,
    bg_threshold: float | None = None,
    gate_enabled: bool = True,
    crop_margin: float = 0.05,
) -> StreamResult:
    """实时帧识别：背景门 -> （可能的）中心裁剪 -> 识别 + 标注 + 计时。

    ``background is None`` 时**完整退回 rc1 的单路识别**：不计算背景相似度、
    不做中心裁剪、不做背景门，连标注绘制顺序都与旧版一致，保证已有页面与
    测试拿到的标注图字节不变。

    Args:
        frame: 摄像头帧或预置帧序列的一帧（numpy / PIL / 路径）。
        library: 特征库。
        threshold: 显式物体阈值（实时链路传 T_rt）；``None`` 时用库内校准值。
        encoder: 自定义编码器；默认 ResNet18。
        background: 背景模型；``None`` 表示保持 rc1 行为。
        bg_threshold: 背景门阈值 T_bg；仅当 ``background`` 非空时生效。
        gate_enabled: 是否启用背景门与中心裁剪；``False`` 时退回 rc1 行为。
        crop_margin: 裁剪结果要超过全帧结果多少才采用，默认 0.05。

    Returns:
        :class:`StreamResult`，``annotated`` 为 RGB ``uint8`` 数组。

    Raises:
        ValueError: 帧为空、阈值/裁剪参数非法、维度不符或特征库维度不匹配。
    """
    if frame is None:
        raise ValueError("帧为空，无法识别")

    started = time.perf_counter()

    # rc1 兼容路径：不碰背景、不裁图，识别与标注的调用顺序都与旧版相同。
    if background is None or not gate_enabled:
        recognition = recognize(
            library, frame, threshold=threshold, top_k=1, encoder=encoder
        )
        annotated = np.asarray(annotate_image(frame, recognition), dtype=np.uint8)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return StreamResult(
            recognition=recognition,
            annotated=annotated,
            elapsed_ms=elapsed_ms,
            speech=speech_text(recognition),
            is_background=False,
            bg_similarity=None,
            crop_similarity=None,
            used_crop=False,
            threshold=recognition.threshold,
        )

    if bg_threshold is not None:
        value = float(bg_threshold)
        if not np.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError(f"背景阈值必须是 [-1, 1] 内的数字，收到 {bg_threshold!r}")
    if float(crop_margin) < 0.0:
        raise ValueError(f"crop_margin 不能为负数，收到 {crop_margin!r}")

    pil = to_pil(frame)
    full_vector = extract_feature(pil, encoder=encoder)
    background_similarity = background.similarity_from_vector(full_vector)
    resolved_threshold = resolve_threshold(library, threshold)

    # 背景门必须"先门后检索"：命中时连物体检索与裁剪都不做，才符合
    # "不检索、不播报"的验收口径，也省掉一次无谓的前向。
    if bg_threshold is not None and background_similarity >= float(bg_threshold):
        recognition = Recognition(
            is_known=False,
            name=BACKGROUND_LABEL,
            similarity=0.0,
            threshold=resolved_threshold,
        )
        annotated = np.asarray(
            annotate_background(frame, background_similarity),
            dtype=np.uint8,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return StreamResult(
            recognition=recognition,
            annotated=annotated,
            elapsed_ms=elapsed_ms,
            speech="",
            is_background=True,
            bg_similarity=background_similarity,
            crop_similarity=None,
            used_crop=False,
            threshold=recognition.threshold,
        )

    full = recognize(library, full_vector, threshold=threshold, top_k=1)
    crop = recognize(
        library,
        center_crop(pil),
        threshold=threshold,
        top_k=1,
        encoder=encoder,
    )
    score = FrameScore(
        full=full,
        crop=crop,
        background_similarity=background_similarity,
        threshold=full.threshold,
    )
    recognition, used_crop = select_recognition(score, crop_margin)
    annotated = np.asarray(annotate_image(frame, recognition), dtype=np.uint8)
    speech = speech_text(recognition)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return StreamResult(
        recognition=recognition,
        annotated=annotated,
        elapsed_ms=elapsed_ms,
        speech=speech,
        is_background=False,
        bg_similarity=background_similarity,
        crop_similarity=score.crop.similarity if score.crop is not None else None,
        used_crop=used_crop,
        threshold=score.threshold,
    )
