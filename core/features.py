"""ResNet18 图像特征提取。

设计要点（也是验收时最容易被问到的三个"为什么"）：

1. **去掉 fc 层**：ImageNet 的 1000 类分类头对本项目的"自定义物体"没有意义。
   去掉后 ResNet18 输出 512 维全局平均池化特征，它描述的是通用视觉外观，
   才能用来做"以图搜图"式的检索。
2. **L2 归一化**：归一化后 ``||v|| = 1``，余弦相似度退化为点积，
   检索时一次矩阵乘法即可；同时向量模长不再影响阈值，阈值可以跨图片复用。
3. **复用 torchvision 官方预处理**：``weights.transforms()`` 与权重训练时一致
   （Resize 256 -> CenterCrop 224 -> ImageNet 均值方差归一化），
   手写预处理很容易漏掉某个细节，导致特征分布偏移、相似度整体偏低。

torch / torchvision 采用**延迟导入**：只跑特征库与检索逻辑的单元测试时，
不需要安装或加载 1100 万参数的模型，测试因此是秒级的。
"""

from __future__ import annotations

import hashlib
import importlib.util
import threading
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

# 模型标识写进特征库 JSON，换模型时可以据此判断旧库是否兼容。
MODEL_NAME = "resnet18-imagenet1k-v1"
FEATURE_DIM = 512

# 单张图片 -> (512,) 或一批图片 -> (N, 512)，输出必须已做 L2 归一化。
Encoder = Callable[[Sequence[Image.Image]], NDArray[np.float32]]

ImageInput = str | Path | Image.Image | NDArray[np.uint8] | NDArray[np.floating]


def l2_normalize(
    vectors: NDArray[np.floating] | list[float],
    *,
    eps: float = 1e-12,
) -> NDArray[np.float32]:
    """对一维向量或二维矩阵按行做 L2 归一化。

    Args:
        vectors: 形状为 ``(D,)`` 或 ``(N, D)`` 的数组。
        eps: 防止除零的下限；零向量归一化后仍是零向量。

    Returns:
        float32 数组，形状与输入一致，每行 L2 范数为 1（零向量除外）。

    Raises:
        ValueError: 输入不是一维或二维。
    """
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim not in (1, 2):
        raise ValueError(f"只支持一维或二维向量，收到 shape={array.shape}")

    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    normalized = array / np.maximum(norms, eps)
    return normalized.astype(np.float32, copy=False)


def _array_to_pil(array: NDArray[np.generic]) -> Image.Image:
    """把 numpy 图像数组转成 RGB ``PIL.Image``（内部辅助函数）。"""
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    elif not (arr.ndim == 3 and arr.shape[2] == 3):
        raise ValueError(f"无法识别的图像 shape={arr.shape}，应为 (H,W)、(H,W,1)、(H,W,3) 或 (H,W,4)")

    if not np.isfinite(arr).all():
        raise ValueError("图像数组包含 NaN 或 Inf")

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.size and float(arr.max()) <= 1.0:
            # 兼容 [0,1] 浮点图（如 matplotlib / 模型输出），避免整张图变成纯黑。
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return Image.fromarray(np.ascontiguousarray(arr))


def to_pil(image: ImageInput) -> Image.Image:
    """把任意支持的输入统一转成 RGB 的 ``PIL.Image``。

    统一入口的价值：路径、PIL、numpy 三种来源（Gradio 上传、摄像头帧、测试脚本）
    都走同一套 EXIF 方向校正与通道转换，避免同一张图得到不同特征。

    Args:
        image: 文件路径、``PIL.Image`` 或 numpy 数组。

    Returns:
        已应用 EXIF 方向、通道为 RGB 的新图像。

    Raises:
        FileNotFoundError: 路径不存在。
        ValueError: numpy 数组形状不受支持。
        TypeError: 输入类型不受支持。
    """
    if isinstance(image, Image.Image):
        # convert("RGB") 会返回副本，后面关闭原文件也不影响结果。
        return ImageOps.exif_transpose(image).convert("RGB")

    if isinstance(image, np.ndarray):
        return _array_to_pil(image)

    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(f"图片不存在: {path}")
        with Image.open(path) as opened:
            # 手机照片常带 EXIF Orientation；不校正会让同一物体横竖两个方向特征差很多。
            return ImageOps.exif_transpose(opened).convert("RGB")

    raise TypeError(f"不支持的图片类型: {type(image)!r}")


def image_digest(image: ImageInput) -> str:
    """计算图像内容的 SHA-256 摘要，用于识别重复录入。

    摘要基于解码后的 RGB 像素，而不是文件字节：同一张图换个格式另存、
    或者带不同 EXIF 元数据，只要像素一致就会被判为重复。

    Args:
        image: 任意支持的图像输入。

    Returns:
        64 位十六进制摘要字符串。
    """
    return hashlib.sha256(to_pil(image).tobytes()).hexdigest()


def build_resnet18_encoder(*, device: str = "cpu") -> Encoder:
    """加载 ImageNet 预训练 ResNet18，返回批量特征编码器。

    首次调用会从 ``download.pytorch.org`` 下载约 45 MB 权重并缓存到
    ``~/.cache/torch/hub/checkpoints``；之后离线也能用。

    Args:
        device: ``"cpu"`` 或 ``"cuda"``。课程要求 CPU 版环境，默认 CPU。

    Returns:
        可调用对象：输入一批 RGB ``PIL.Image``，返回 ``(N, 512)`` 的
        float32 已归一化特征矩阵。

    Raises:
        RuntimeError: 权重下载失败（通常是首次运行时没有网络）。
    """
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.IMAGENET1K_V1
    try:
        model = resnet18(weights=weights)
    except Exception as exc:  # 网络错误类型因 torchvision 版本而异，统一给出可操作提示
        cache_home = Path.home() / ".cache" / "torch"
        raise RuntimeError(
            "加载 ResNet18 预训练权重失败。首次运行需要联网下载约 45MB 权重，"
            f"并写入 {cache_home}（或 TORCH_HOME）。若该目录不可写，请把 TORCH_HOME "
            "指向可写目录；离线时可手动把 resnet18-f37072fd.pth 放到 "
            f"{cache_home / 'hub' / 'checkpoints'}。"
            f"原始错误: {exc}"
        ) from exc

    # 关键：分类头替换为 Identity，输出 512 维池化特征而不是 1000 类 logits。
    model.fc = torch.nn.Identity()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    transform = weights.transforms()

    def encode(images: Sequence[Image.Image]) -> NDArray[np.float32]:
        if not images:
            raise ValueError("encode 至少需要一张图片")
        batch = torch.stack([transform(to_pil(image)) for image in images]).to(device)
        with torch.inference_mode():
            features = model(batch)
        array = features.detach().cpu().numpy().astype(np.float32, copy=False)
        return l2_normalize(array)

    return encode


_encoder_lock = threading.Lock()
_default_encoder: Encoder | None = None


def get_default_encoder() -> Encoder:
    """返回进程内共享的 ResNet18 编码器（懒加载 + 双重检查锁）。

    模型只加载一次：Gradio 每次点击都重新构造模型会多花数秒，
    而且反复申请内存容易在演示现场卡顿。
    """
    global _default_encoder
    if _default_encoder is None:
        with _encoder_lock:
            if _default_encoder is None:
                _default_encoder = build_resnet18_encoder()
    return _default_encoder


def extract_features(
    images: Sequence[ImageInput],
    *,
    encoder: Encoder | None = None,
    batch_size: int = 16,
) -> NDArray[np.float32]:
    """批量提取特征。

    Args:
        images: 图片路径 / PIL / numpy 数组的序列。
        encoder: 自定义编码器；默认使用共享的 ResNet18。
        batch_size: 每批送进网络的图片数。CPU 上 8~16 比较稳。

    Returns:
        ``(N, 512)`` 的 float32 矩阵，每行 L2 范数为 1。

    Raises:
        ValueError: 图片列表为空、batch_size 非法、编码器输出形状不合法。
    """
    if batch_size < 1:
        raise ValueError("batch_size 必须 >= 1")

    items = list(images)
    if not items:
        raise ValueError("至少提供一张图片")

    pil_images = [to_pil(image) for image in items]
    encode = encoder if encoder is not None else get_default_encoder()

    chunks: list[NDArray[np.float32]] = []
    for start in range(0, len(pil_images), batch_size):
        chunk_images = pil_images[start : start + batch_size]
        raw = np.asarray(encode(chunk_images), dtype=np.float32)

        if raw.ndim == 1:
            if len(chunk_images) != 1:
                raise ValueError(
                    f"编码器对 {len(chunk_images)} 张图只返回了一维向量，形状不合法"
                )
            raw = raw[None, :]

        if raw.ndim != 2 or raw.shape[0] != len(chunk_images):
            raise ValueError(
                "编码器输出形状不合法："
                f"期望 ({len(chunk_images)}, D)，实际 {tuple(raw.shape)}"
            )
        chunks.append(raw)

    vectors = np.concatenate(chunks, axis=0)
    # 即使自定义编码器忘了归一化，这里也统一兜底，保证入库向量满足契约。
    return l2_normalize(vectors)


def extract_feature(
    image: ImageInput,
    *,
    encoder: Encoder | None = None,
) -> NDArray[np.float32]:
    """提取单张图片的特征，返回 ``(512,)`` 的已归一化向量。

    Args:
        image: 图片路径 / PIL / numpy 数组。
        encoder: 自定义编码器；默认 ResNet18。

    Returns:
        float32 一维特征向量。
    """
    return extract_features([image], encoder=encoder, batch_size=1)[0]


def torch_available() -> bool:
    """检查 torch 与 torchvision 是否可导入，供测试或界面做能力探测。"""
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("torchvision") is not None
    )
