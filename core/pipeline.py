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

from .features import Encoder, ImageInput, to_pil
from .library import FeatureLibrary
from .recognize import UNKNOWN_LABEL, Recognition, recognize

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
class StreamResult:
    """一帧实时画面的完整产出，界面只需要这个对象。

    Attributes:
        recognition: 检索与判未知结果。
        annotated: 已叠加名称/相似度的 RGB ``uint8`` 数组，可直接给 Gradio。
        elapsed_ms: 本帧"识别 + 标注"的墙钟耗时（毫秒）。
        speech: 该帧对应的播报文本；是否真的播由 ``SpeechGate`` 决定。
    """

    recognition: Recognition
    annotated: NDArray[np.uint8]
    elapsed_ms: float
    speech: str

    @property
    def is_known(self) -> bool:
        """本帧是否识别为已知物体。"""
        return self.recognition.is_known

    @property
    def name(self) -> str:
        """本帧展示的名称（未知时为「未知物体」）。"""
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
) -> StreamResult:
    """实时帧识别：识别 + 标注 + 计时，一帧一次调用。

    实时链路不展示近邻表，``top_k`` 固定为 1，少做一次 argsort 无关紧要，
    但语义上明确"只关心最高分"。耗时统计包含标注绘制，因为标注图就是
    用户最终看到的那一帧，应该按"可显示"的完整代价来算。

    Args:
        frame: 摄像头帧或预置帧序列的一帧（numpy / PIL / 路径）。
        library: 特征库。
        threshold: 显式阈值；``None`` 时用库内校准值或默认值。
        encoder: 自定义编码器；默认 ResNet18。

    Returns:
        :class:`StreamResult`，``annotated`` 为 RGB ``uint8`` 数组。

    Raises:
        ValueError: 帧为空、维度不符或特征库维度不匹配。
    """
    if frame is None:
        raise ValueError("帧为空，无法识别")

    started = time.perf_counter()
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
    )
