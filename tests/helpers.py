"""测试用的确定性假编码器与图片工具（不依赖 torch）。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image

DIM = 8

RED = (220, 40, 40)
GREEN = (40, 200, 60)
BLUE = (40, 60, 220)
YELLOW = (240, 210, 40)


def one_hot(index: int, dim: int = DIM) -> NDArray[np.float32]:
    """构造互相正交的单位向量，让相似度结果可以手算。"""
    vector = np.zeros(dim, dtype=np.float32)
    vector[index] = 1.0
    return vector


def solid(color: tuple[int, int, int], size: tuple[int, int] = (64, 64)) -> Image.Image:
    """生成纯色图片；颜色即"类别"，用于假编码器。"""
    return Image.new("RGB", size, color)


def color_encoder(
    mapping: Mapping[tuple[int, int, int], NDArray[np.floating]],
) -> Callable[[Sequence[Image.Image]], NDArray[np.float32]]:
    """按左上角像素颜色查表返回特征，模拟一个确定性的视觉编码器。"""
    table = {color: np.asarray(vector, dtype=np.float32) for color, vector in mapping.items()}

    def encode(images: Sequence[Image.Image]) -> NDArray[np.float32]:
        vectors: list[NDArray[np.float32]] = []
        for image in images:
            color = image.convert("RGB").getpixel((0, 0))
            if color not in table:
                raise KeyError(f"假编码器没有为颜色 {color} 配置向量")
            vectors.append(table[color])
        return np.stack(vectors)

    return encode
