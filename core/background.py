"""实时背景向量：启动时学一段空场景，作为「背景门」的参照。

为什么单独建一个文件、而不是写进 ``library.json``：

- ``library.json`` 的 ``objects`` 是**物体**，背景不是物体，混进去会污染
  「已录入物体数量」、物体级 top-1 聚合，以及 P6b 用测试图做的阈值校准；
- 背景是「当前这台电脑、这个机位」的现场信息，换教室/挪摄像头就该重学，
  它的生命周期和物体库完全不同；
- 文件分开后，删掉 ``background.json`` 只会让背景门回到"未学习"状态，
  不会伤到任何已录入的物体特征。

存储格式（版本 1）::

    {
      "schema": "rn18-background",
      "version": 1,
      "model": "resnet18-imagenet1k-v1",
      "dim": 512,
      "normalized": true,
      "created_at": "2026-11-09T02:30:00Z",
      "frames_used": 18,
      "vector": [0.0123, ...]
    }

为什么取**逐维中位数**而不是均值：摄像头前的画面会有人走过、有光标闪动、
有自动曝光跳变。均值会被少数离群帧整体拉偏；中位数对少量污染帧不敏感，
是课程 Demo 里最便宜也最稳的鲁棒统计量。取完中位数后再做一次 L2 归一化，
这样背景相似度仍然是余弦相似度，与物体库的检索口径一致。
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .features import (
    MODEL_NAME,
    Encoder,
    ImageInput,
    extract_feature,
    extract_features,
    l2_normalize,
)

SCHEMA_NAME = "rn18-background"
SCHEMA_VERSION = 1

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKGROUND_PATH = PROJECT_ROOT / "data" / "background.json"


def _now() -> str:
    """返回带时区的 ISO-8601 UTC 时间戳，例如 ``2026-11-09T02:30:00Z``。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(  # noqa: UP017 - 兼容 Python 3.10
        "+00:00", "Z"
    )


@dataclass(slots=True)
class BackgroundModel:
    """一段空场景的 512 维背景向量（L2 归一化）。

    Attributes:
        vector: 形状 ``(dim,)`` 的 float32 单位向量。
        frames_used: 实际参与中位数的帧数（已扣除预热帧）。
        created_at: 学习时间（UTC ISO-8601）。
        model: 特征来源模型标识，换模型时据此判断背景文件是否还可复用。
        path: 绑定的 JSON 路径；``load``/``save`` 会自动维护。
    """

    vector: NDArray[np.float32]
    frames_used: int = 0
    created_at: str = field(default_factory=_now)
    model: str = MODEL_NAME
    path: Path | None = None

    def __post_init__(self) -> None:
        vector = np.asarray(self.vector, dtype=np.float32).reshape(-1)
        if vector.size == 0:
            raise ValueError("背景向量不能为空")
        if not np.isfinite(vector).all():
            raise ValueError("背景向量包含 NaN 或 Inf")
        norm = float(np.linalg.norm(vector))
        if norm < 1e-6:
            raise ValueError("背景向量是零向量，无法做余弦相似度")
        self.vector = (vector / norm).astype(np.float32, copy=False)
        self.frames_used = int(self.frames_used)
        if self.frames_used < 0:
            raise ValueError(f"frames_used 不能为负数，收到 {self.frames_used}")

    @property
    def dim(self) -> int:
        """背景向量维度。"""
        return int(self.vector.size)

    @classmethod
    def learn(
        cls,
        frames: Sequence[ImageInput],
        *,
        encoder: Encoder | None = None,
        take: int = 20,
        drop_first: int = 2,
    ) -> BackgroundModel:
        """从一段空场景帧学习背景向量。

        先丢 ``drop_first`` 帧（摄像头首帧常见黑帧/过曝），再从剩余帧里取
        最多 ``take`` 帧提特征；逐维取中位数后做 L2 归一化。

        Args:
            frames: 空场景帧序列（路径 / PIL / numpy）。
            encoder: 自定义编码器；``None`` 表示共享的 ResNet18。
            take: 参与学习的最多帧数，默认 20。
            drop_first: 开头丢弃的帧数，默认 2。

        Returns:
            学习好的 :class:`BackgroundModel`，``frames_used`` 为实际帧数。

        Raises:
            ValueError: 帧数不足、``take``/``drop_first`` 非法或编码器输出异常。
        """
        if int(take) < 1:
            raise ValueError(f"take 必须 >= 1，收到 {take!r}")
        if int(drop_first) < 0:
            raise ValueError(f"drop_first 不能为负数，收到 {drop_first!r}")

        items = list(frames)
        if len(items) <= int(drop_first):
            raise ValueError(
                f"背景帧不足：需要至少 {int(drop_first) + 1} 帧，收到 {len(items)} 帧"
            )

        selected = items[int(drop_first) : int(drop_first) + int(take)]
        vectors = extract_features(selected, encoder=encoder)
        # extract_features 已按行归一化；对归一化向量逐维取中位数，
        # 再归一化一次，保证背景相似度仍是标准余弦。
        median = np.median(vectors, axis=0).astype(np.float32)
        return cls(
            vector=l2_normalize(median),
            frames_used=len(selected),
        )

    def similarity(self, frame: ImageInput, *, encoder: Encoder | None = None) -> float:
        """计算一帧画面与背景的余弦相似度。

        Args:
            frame: 一帧画面（路径 / PIL / numpy）。
            encoder: 自定义编码器；``None`` 表示共享的 ResNet18。

        Returns:
            ``[-1, 1]`` 内的余弦相似度，越接近 1 越像背景。
        """
        return self.similarity_from_vector(extract_feature(frame, encoder=encoder))

    def similarity_from_vector(self, vector: NDArray[np.floating]) -> float:
        """用已经提好的帧特征计算背景相似度（避免重复前向）。

        Args:
            vector: 形状 ``(dim,)`` 的帧特征；内部会再归一化一次兜底。

        Returns:
            ``[-1, 1]`` 内的余弦相似度。

        Raises:
            ValueError: 维度不匹配或包含 NaN/Inf。
        """
        query = np.asarray(vector, dtype=np.float32).reshape(-1)
        if query.size != self.dim:
            raise ValueError(
                f"帧特征维度({query.size})与背景向量({self.dim})不一致，"
                "请确认 background.json 与当前模型匹配"
            )
        if not np.isfinite(query).all():
            raise ValueError("帧特征包含 NaN 或 Inf")
        normalized = l2_normalize(query)
        return float(np.clip(np.dot(self.vector, normalized), -1.0, 1.0))

    def to_dict(self) -> dict[str, Any]:
        """序列化为与 ``background.json`` 契约一致的字典。"""
        return {
            "schema": SCHEMA_NAME,
            "version": SCHEMA_VERSION,
            "model": self.model,
            "dim": self.dim,
            "normalized": True,
            "created_at": self.created_at,
            "frames_used": int(self.frames_used),
            # 与特征库一样保留 8 位小数：余弦误差 < 1e-7，文件小很多。
            "vector": [round(float(value), 8) for value in self.vector],
        }

    def save(self, path: str | Path | None = None) -> Path:
        """原子写入背景 JSON。

        Args:
            path: 目标路径；省略时使用构造/加载时绑定的路径。

        Returns:
            实际写入的绝对路径。

        Raises:
            ValueError: 没有绑定路径且未显式传入。
            OSError: 磁盘写入失败。
        """
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("背景模型未绑定路径，save() 需要显式传入 path")
        target = target.expanduser()
        _atomic_write_json(target, self.to_dict())
        self.path = target.resolve()
        return target.resolve()

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        path: str | Path | None = None,
    ) -> BackgroundModel:
        """从字典构造背景模型并做严格校验。

        Args:
            data: ``json.load`` 得到的字典。
            path: 可选绑定路径。

        Returns:
            校验通过的背景模型。

        Raises:
            ValueError: schema/版本/维度/归一化/向量内容不合法。
        """
        if not isinstance(data, Mapping):
            raise ValueError("背景 JSON 顶层必须是对象")  # noqa: TRY004 - 与 library.py 的 schema 校验口径一致

        schema = data.get("schema")
        if schema != SCHEMA_NAME:
            raise ValueError(f"不是本程序的背景文件（schema={schema!r}）")

        version = data.get("version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"不支持的背景文件版本 {version!r}，本程序只支持版本 {SCHEMA_VERSION}"
            )

        if data.get("normalized", True) is not True:
            raise ValueError("背景文件必须保存 L2 归一化向量（normalized=true）")

        try:
            vector = np.asarray(data["vector"], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("背景文件缺少合法的 vector 字段") from exc
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError(f"背景向量必须是一维非空数组，实际 shape={vector.shape}")
        if not np.isfinite(vector).all():
            raise ValueError("背景向量包含 NaN/Inf")

        dim_value = data.get("dim", vector.size)
        try:
            dim = int(dim_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"背景文件 dim 非法: {dim_value!r}") from exc
        if dim != vector.size:
            raise ValueError(
                f"背景文件 dim={dim} 与向量长度 {vector.size} 不一致"
            )

        norm = float(np.linalg.norm(vector))
        if norm < 1e-6:
            raise ValueError("背景向量是零向量")
        if abs(norm - 1.0) > 1e-3:
            raise ValueError(f"背景向量未归一化（范数 {norm:.6f}）")

        model_name = str(data.get("model", MODEL_NAME))
        if model_name != MODEL_NAME:
            raise ValueError(
                f"背景文件来自模型 {model_name!r}，当前模型是 {MODEL_NAME!r}，"
                "请重新学背景"
            )

        try:
            frames_used = int(data.get("frames_used", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("背景文件 frames_used 非法") from exc
        if frames_used < 0:
            raise ValueError(f"背景文件 frames_used 不能为负数: {frames_used}")

        return cls(
            vector=vector / norm,
            frames_used=frames_used,
            created_at=str(data.get("created_at") or _now()),
            model=model_name,
            path=Path(path).resolve() if path is not None else None,
        )


def load_background(
    path: str | Path | None = None,
    *,
    create: bool = False,
) -> BackgroundModel | None:
    """从 JSON 加载背景模型。

    Args:
        path: 文件路径；默认 ``data/background.json``。
        create: 文件不存在时返回 ``None``（``True``）还是抛异常（``False``）。

    Returns:
        ``BackgroundModel``；``create=True`` 且文件不存在时返回 ``None``。

    Raises:
        FileNotFoundError: ``create=False`` 且文件不存在。
        ValueError: JSON 损坏或内容不符合 schema。
    """
    target = (Path(path) if path is not None else DEFAULT_BACKGROUND_PATH).expanduser()
    if not target.exists():
        if create:
            return None
        raise FileNotFoundError(f"背景文件不存在: {target}")

    try:
        raw = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise OSError(f"读取背景文件失败: {target}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"背景 JSON 解析失败（文件可能损坏）: {target}: {exc}") from exc

    return BackgroundModel.from_dict(data, path=target)


def save_background(
    model: BackgroundModel,
    path: str | Path | None = None,
) -> Path:
    """保存背景模型（``BackgroundModel.save`` 的函数式包装）。"""
    return model.save(path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """同目录临时文件 + ``os.replace`` 原子替换目标文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


__all__ = [
    "DEFAULT_BACKGROUND_PATH",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "BackgroundModel",
    "load_background",
    "save_background",
]
