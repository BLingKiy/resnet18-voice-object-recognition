"""实时背景向量：启动时学一段空场景，作为「背景门」的参照。

为什么背景要单独存 ``data/background.json``，不能写进 ``library.json``：

- ``library.json`` 里的每条记录都是**物体**；背景不是物体，混进去会污染
  「已录入物体数量」、物体级 top-1 聚合，以及 P6b 用测试图做的阈值校准。
- 背景是「当前这台电脑、这个机位」的现场信息，换教室或挪摄像头就该重学，
  生命周期与物体库完全不同。
- 文件分开后，删除 ``background.json`` 只会让背景门回到「未学习」状态，
  不会伤到任何已录入的物体特征。

本模块只做背景向量的学习、比较与持久化，不参与检索，也不改动物体库；
特征提取**复用** ``core.features.extract_features``，预处理（Resize 256 ->
CenterCrop 224 -> ImageNet 均值方差）与静态/实时链路完全一致。

为什么取逐维**中位数**而不是均值：摄像头前会有人走过、光标闪动、自动曝光
跳变。均值会被少数离群帧整体拉偏；中位数对少量污染帧不敏感，是课程 Demo
里最便宜也最稳的鲁棒统计量。取完中位数后再做一次 L2 归一化，背景相似度
仍然是标准余弦，与物体库的检索口径一致。
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .features import (
    FEATURE_DIM,
    MODEL_NAME,
    Encoder,
    extract_features,
)

SCHEMA_NAME = "rn18-background"
SCHEMA_VERSION = 1

# 任务硬要求：丢弃预热帧后不足 10 帧就报错，不能拿半截数据凑背景。
MIN_SAMPLES = 10

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKGROUND_PATH = PROJECT_ROOT / "data" / "background.json"


def _now() -> str:
    """返回带时区的 ISO-8601 UTC 时间戳，例如 ``2026-11-09T02:30:00Z``。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(  # noqa: UP017 - 兼容 Python 3.10
        "+00:00", "Z"
    )


def _as_unit_vector(
    vector: NDArray[np.floating] | Iterable[float],
    *,
    expected_dim: int | None = None,
) -> NDArray[np.float32]:
    """把输入转成一维、有限、非零、L2 归一化的 float32 向量。

    Args:
        vector: 任意可转成数组的向量。
        expected_dim: 期望维度；``None`` 表示不限制。

    Returns:
        单位向量。

    Raises:
        ValueError: 维度不符、含 NaN/Inf 或是零向量。
    """
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise ValueError("背景向量不能为空")
    if expected_dim is not None and array.size != int(expected_dim):
        raise ValueError(
            f"背景向量维度 {array.size} 与期望的 {int(expected_dim)} 不一致"
        )
    if not np.isfinite(array).all():
        raise ValueError("背景向量包含 NaN 或 Inf")
    norm = float(np.linalg.norm(array))
    if norm < 1e-6:
        raise ValueError("背景向量是零向量，无法做余弦相似度")
    return (array / norm).astype(np.float32, copy=False)


def _validate_threshold(value: float) -> float:
    """校验阈值是 ``[-1, 1]`` 内的有限数字。"""
    threshold = float(value)
    if not np.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
        raise ValueError(f"背景阈值必须是 [-1, 1] 内的数字，收到 {value!r}")
    return threshold


@dataclass(slots=True)
class BackgroundModel:
    """一段空场景的 512 维背景向量（L2 归一化）。

    Attributes:
        vector: 形状 ``(dim,)`` 的 float32 单位向量；新建时可以为 ``None``，
            调用 :meth:`collect` 或 :meth:`load` 后才有值。
        created_at: 学习/加载时间（UTC ISO-8601）。
        sample_count: 实际参与中位数的帧数（已扣除预热帧）。
        model: 特征来源模型标识，换模型时据此判断背景文件是否还可复用。
        path: 绑定的 JSON 路径；:meth:`save` / :meth:`load` 会自动维护。
    """

    vector: NDArray[np.float32] | None = None
    created_at: str = field(default_factory=_now)
    sample_count: int = 0
    model: str = MODEL_NAME
    path: Path | None = None

    def __post_init__(self) -> None:
        if self.sample_count < 0:
            raise ValueError(f"sample_count 不能为负数，收到 {self.sample_count}")
        self.sample_count = int(self.sample_count)
        # 构造函数允许传未归一化向量（调用方可能刚从保存文件里读出来），
        # 这里统一归一化，保证相似度口径始终是余弦。
        if self.vector is not None:
            self.vector = _as_unit_vector(self.vector)

    @property
    def dim(self) -> int:
        """背景向量维度；尚未学习时返回 512。"""
        return int(self.vector.size) if self.vector is not None else FEATURE_DIM

    @property
    def frames_used(self) -> int:
        """``sample_count`` 的别名，便于旧代码/报告沿用「帧数」的说法。"""
        return self.sample_count

    def collect(
        self,
        frames: Iterable[np.ndarray],
        n_frames: int = 20,
        drop_first: int = 2,
        *,
        encoder: Encoder | None = None,
    ) -> NDArray[np.float32]:
        """从一段空场景帧学习背景向量（逐维中位数 + L2 归一化）。

        Args:
            frames: 空场景帧序列（numpy / PIL / 路径均可，由特征层统一转 RGB）。
            n_frames: 最多取多少帧参与学习，默认 20。
            drop_first: 开头丢弃的帧数，默认 2（摄像头首帧常见黑帧/过曝）。
            encoder: 自定义编码器；``None`` 表示共享的 ResNet18。单元测试
                用假编码器避免加载真实权重。

        Returns:
            学到的单位背景向量，同时写入 ``self.vector``。

        Raises:
            ValueError: 帧数不足（丢弃预热帧后少于 10 帧）、参数非法，
                或编码器输出形状不合法。
        """
        if int(n_frames) < 1:
            raise ValueError(f"n_frames 必须 >= 1，收到 {n_frames!r}")
        if int(drop_first) < 0:
            raise ValueError(f"drop_first 不能为负数，收到 {drop_first!r}")
        if int(drop_first) >= int(n_frames):
            raise ValueError(
                f"drop_first({drop_first}) 必须小于 n_frames({n_frames})"
            )

        items = list(frames)
        # 先截取「采集窗口」，再丢窗口开头的预热帧：调用方多给了帧也只按
        # 本次采集的 20 帧处理，避免把不同时间段的画面混进同一个背景。
        window = items[: int(n_frames)]
        selected = window[int(drop_first) :]
        if len(selected) < MIN_SAMPLES:
            raise ValueError(
                "背景样本不足："
                f"采集窗口 {len(window)} 帧、丢弃前 {int(drop_first)} 帧后只剩 "
                f"{len(selected)} 帧，至少需要 {MIN_SAMPLES} 帧。"
                "请让画面对准空场景、延长采集后再试。"
            )

        vectors = np.asarray(
            extract_features(selected)
            if encoder is None
            else extract_features(selected, encoder=encoder),
            dtype=np.float32,
        )
        if vectors.ndim != 2 or vectors.shape[0] != len(selected):
            raise ValueError(
                "编码器输出形状不合法："
                f"期望 ({len(selected)}, D)，实际 {tuple(vectors.shape)}"
            )
        if vectors.shape[1] == 0 or not np.isfinite(vectors).all():
            raise ValueError("编码器输出为空或包含 NaN/Inf")

        # 逐维中位数：对少量离群帧（人走过、曝光跳变）不敏感。
        median = np.median(vectors, axis=0).astype(np.float32)
        self.vector = _as_unit_vector(median)
        self.sample_count = len(selected)
        self.created_at = _now()
        self.model = MODEL_NAME
        return self.vector

    def similarity(self, feat: np.ndarray) -> float:
        """计算一帧**特征向量**与背景的余弦相似度。

        参数是已经提好的特征，不是原始图片：实时链路每帧只跑一次 ResNet18，
        全帧特征同时用于背景门和物体检索，避免重复前向。

        Args:
            feat: 形状 ``(dim,)`` 的帧特征（可以是未归一化的原始输出）。

        Returns:
            ``[-1, 1]`` 内的余弦相似度，越接近 1 越像背景。

        Raises:
            RuntimeError: 背景模型尚未 :meth:`collect` / :meth:`load`。
            ValueError: 维度不匹配、含 NaN/Inf 或是零向量。
        """
        if self.vector is None:
            raise RuntimeError("背景模型还没有向量，请先 collect() 或 load()")
        query = _as_unit_vector(feat, expected_dim=self.dim)
        return float(np.clip(np.dot(self.vector, query), -1.0, 1.0))

    def is_background(self, feat: np.ndarray, t_bg: float) -> bool:
        """判断一帧特征是否属于背景（相似度严格大于 ``t_bg``）。

        用严格大于而不是大于等于：阈值校准脚本取的是两类分布的中点，
        恰好落在阈值上的样本极少数；严格大于让边界样本归入「可能有物体」，
        宁可多检索一次，也不要把刚入场的物体直接门掉。

        Args:
            feat: 形状 ``(dim,)`` 的帧特征。
            t_bg: 背景门阈值 ``T_bg``。

        Returns:
            ``True`` 表示画面中没有可识别物体。
        """
        return self.similarity(feat) > _validate_threshold(t_bg)

    def to_dict(self) -> dict[str, Any]:
        """序列化为 ``background.json`` 的字典结构。"""
        if self.vector is None:
            raise ValueError("背景模型还没有向量，无法序列化")
        return {
            "schema": SCHEMA_NAME,
            "version": SCHEMA_VERSION,
            "model": self.model,
            "created_at": self.created_at,
            "dim": self.dim,
            "sample_count": int(self.sample_count),
            "normalized": True,
            # 保留 8 位小数：余弦误差 < 1e-7，文件体积小很多。
            "vector": [round(float(value), 8) for value in self.vector],
        }

    def save(self, path: str | Path) -> None:
        """把背景模型原子写入 JSON。

        Args:
            path: 目标路径，通常为 ``data/background.json``。

        Raises:
            ValueError: 模型还没有向量。
            OSError: 磁盘写入失败。
        """
        target = Path(path).expanduser()
        _atomic_write_json(target, self.to_dict())
        self.path = target.resolve()

    @classmethod
    def load(cls, path: str | Path) -> BackgroundModel:
        """从 JSON 加载背景模型并做严格校验。

        Args:
            path: ``background.json`` 路径。

        Returns:
            校验通过的 :class:`BackgroundModel`。

        Raises:
            FileNotFoundError: 文件不存在。
            ValueError: JSON 损坏、版本不支持、维度不是 512，
                或向量没有 L2 归一化。
        """
        target = Path(path).expanduser()
        try:
            raw = target.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise OSError(f"读取背景文件失败: {target}") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"背景 JSON 解析失败（文件可能损坏）: {target}: {exc}"
            ) from exc
        return cls.from_dict(data, path=target)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        path: str | Path | None = None,
    ) -> BackgroundModel:
        """从字典构造模型；``load`` 的校验逻辑集中在这里，便于单测。"""
        if not isinstance(data, Mapping):
            raise ValueError("背景 JSON 顶层必须是对象")  # noqa: TRY004 - 与 library.py 的 schema 校验口径一致

        version = data.get("version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"不支持的背景文件版本 {version!r}，本程序只支持 {SCHEMA_VERSION}"
            )

        raw_dim = data.get("dim")
        try:
            dim = int(raw_dim)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"背景文件 dim 非法: {raw_dim!r}") from exc
        if dim != FEATURE_DIM:
            raise ValueError(
                f"背景文件维度必须是 {FEATURE_DIM}，实际 dim={dim}"
            )

        if "vector" not in data:
            raise ValueError("背景文件缺少 vector 字段")
        try:
            vector = np.asarray(data["vector"], dtype=np.float32).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError("背景文件 vector 不是合法的数字数组") from exc
        if vector.size != FEATURE_DIM:
            raise ValueError(
                f"背景向量长度必须是 {FEATURE_DIM}，实际 {vector.size}"
            )
        if not np.isfinite(vector).all():
            raise ValueError("背景向量包含 NaN/Inf")

        norm = float(np.linalg.norm(vector))
        if norm < 1e-6:
            raise ValueError("背景向量是零向量")
        if abs(norm - 1.0) > 1e-3:
            raise ValueError(
                f"背景向量未归一化（L2 范数 {norm:.6f}，要求约等于 1）"
            )

        try:
            sample_count = int(data.get("sample_count", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("背景文件 sample_count 非法") from exc
        if sample_count < 0:
            raise ValueError(f"背景文件 sample_count 不能为负数: {sample_count}")

        created_at = data.get("created_at")
        if created_at is not None and not isinstance(created_at, str):
            raise ValueError("背景文件 created_at 必须是字符串")

        return cls(
            vector=vector / norm,
            created_at=created_at or _now(),
            sample_count=sample_count,
            model=str(data.get("model", MODEL_NAME)),
            path=Path(path).resolve() if path is not None else None,
        )


def load_background(
    path: str | Path | None = None,
    *,
    create: bool = False,
) -> BackgroundModel | None:
    """函数式加载包装：文件不存在时可按 ``create`` 返回 ``None``。

    Args:
        path: 文件路径；默认 ``data/background.json``。
        create: ``True`` 时文件不存在返回 ``None``，否则抛 ``FileNotFoundError``。

    Returns:
        ``BackgroundModel`` 或 ``None``。
    """
    target = Path(path) if path is not None else DEFAULT_BACKGROUND_PATH
    if not target.exists():
        if create:
            return None
        raise FileNotFoundError(f"背景文件不存在: {target}")
    return BackgroundModel.load(target)


def save_background(
    model: BackgroundModel,
    path: str | Path | None = None,
) -> None:
    """函数式保存包装；``path`` 省略时使用模型绑定的路径。"""
    target = Path(path) if path is not None else model.path
    if target is None:
        raise ValueError("背景模型未绑定路径，save() 需要显式传入 path")
    model.save(target)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """同目录临时文件 + ``os.replace`` 原子替换，避免写一半的文件。"""
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
    "MIN_SAMPLES",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "BackgroundModel",
    "load_background",
    "save_background",
]
