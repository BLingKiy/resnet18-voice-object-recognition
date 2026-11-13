"""本地 JSON 特征库：增删查、持久化与完整性校验。

存储格式（版本 1）::

    {
      "schema": "rn18-feature-library",
      "version": 1,
      "model": "resnet18-imagenet1k-v1",
      "dim": 512,
      "normalized": true,
      "threshold": 0.62,
      "created_at": "...",
      "updated_at": "...",
      "objects": [
        {
          "name": "苹果",
          "created_at": "...",
          "entries": [
            {"image": "apple_01.jpg", "digest": "sha256...", "added_at": "...",
             "vector": [0.0123, ...]}
          ]
        }
      ]
    }

为什么用 JSON 而不是 SQLite / 向量数据库：
- 课程要求"重启后特征库仍在"，JSON 零依赖、可读、可直接在报告里展示；
- 5 个物体 × 5 张图 = 25 条 512 维向量，文件约 300KB，全量加载毫无压力；
- 用 SQLite 或 FAISS 属于工程化扩张，评分不为复杂度加分。

写入采用"临时文件 + ``os.replace``"原子替换：演示现场如果程序被强杀，
不会留下半个 JSON 把特征库彻底弄坏。
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Iterable, Mapping, Sequence
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
    ImageInput,
    extract_features,
    image_digest,
    to_pil,
)

SCHEMA_NAME = "rn18-feature-library"
SCHEMA_VERSION = 1

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIBRARY_PATH = PROJECT_ROOT / "data" / "features.json"


def _now() -> str:
    """返回带时区的 ISO-8601 UTC 时间戳，例如 ``2026-11-09T02:30:00Z``。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(slots=True)
class Entry:
    """一条注册记录：一张图片对应的 512 维特征。"""

    image: str
    vector: NDArray[np.float32]
    digest: str = ""
    added_at: str = field(default_factory=_now)


@dataclass(slots=True)
class ObjectRecord:
    """一个物体的全部注册记录。同一物体允许多张图（多视角）。"""

    name: str
    created_at: str = field(default_factory=_now)
    entries: list[Entry] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class AddResult:
    """一次录入操作的结果，供界面提示与命令行输出。"""

    object_name: str
    added: int
    skipped: int
    object_vectors: int
    total_vectors: int


class FeatureLibrary:
    """内存特征库，可绑定一个 JSON 文件路径。"""

    def __init__(
        self,
        *,
        path: str | Path | None = None,
        dim: int = FEATURE_DIM,
        threshold: float | None = None,
        model: str = MODEL_NAME,
    ) -> None:
        """
        Args:
            path: 绑定的 JSON 文件路径；``save()`` 不传参时写到这里。
            dim: 特征维度，正式环境固定 512；单元测试可用小维度。
            threshold: 判未知阈值；``None`` 表示由识别层使用默认阈值。
            model: 特征来源模型标识，写入 JSON 便于排查兼容性。

        Raises:
            ValueError: 维度不是正整数，或阈值不在 [-1, 1]。
        """
        if int(dim) < 1:
            raise ValueError(f"dim 必须是正整数，收到 {dim!r}")

        self.path: Path | None = Path(path) if path is not None else None
        self.dim = int(dim)
        self.model = str(model)
        self.threshold = self._validate_threshold(threshold)
        self.created_at = _now()
        self.updated_at = self.created_at
        self._objects: dict[str, ObjectRecord] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ 查询
    @property
    def object_count(self) -> int:
        """已录入的物体种类数。"""
        with self._lock:
            return len(self._objects)

    @property
    def total_vectors(self) -> int:
        """全库特征向量总条数（等于注册图片总数，不含重复跳过）。"""
        with self._lock:
            return sum(len(record.entries) for record in self._objects.values())

    def __len__(self) -> int:
        """``len(library)`` 返回特征向量总数，界面可直接显示"库条数"。"""
        return self.total_vectors

    def __contains__(self, name: object) -> bool:
        with self._lock:
            return isinstance(name, str) and name in self._objects

    def get(self, name: str) -> ObjectRecord | None:
        """按名称取物体记录；不存在返回 ``None``。"""
        with self._lock:
            return self._objects.get(name)

    def stats(self) -> list[dict[str, Any]]:
        """返回供界面表格 / CLI 展示的统计信息，按物体名排序。"""
        with self._lock:
            rows = [
                {
                    "name": record.name,
                    "count": len(record.entries),
                    "created_at": record.created_at,
                    "first_image": record.entries[0].image if record.entries else "",
                }
                for record in self._objects.values()
            ]
        return sorted(rows, key=lambda row: row["name"])

    def matrix(self) -> tuple[list[str], list[str], NDArray[np.float32]]:
        """把所有向量拼成检索矩阵。

        Returns:
            ``(names, images, matrix)``：
            - ``names[i]``：第 i 条向量所属物体；
            - ``images[i]``：第 i 条向量的注册图片标签；
            - ``matrix``：形状 ``(N, dim)`` 的 float32 矩阵。

            空库返回 ``([], [], zeros((0, dim)))``。
        """
        with self._lock:
            names: list[str] = []
            images: list[str] = []
            vectors: list[NDArray[np.float32]] = []
            for record in self._objects.values():
                for entry in record.entries:
                    names.append(record.name)
                    images.append(entry.image)
                    vectors.append(entry.vector)

        if not vectors:
            return names, images, np.zeros((0, self.dim), dtype=np.float32)

        # vstack 会复制一份，调用方拿到的矩阵与库内部解耦，避免迭代中被修改。
        return names, images, np.vstack(vectors).astype(np.float32, copy=False)

    # ------------------------------------------------------------------ 写入
    def add(
        self,
        name: str,
        images: Sequence[ImageInput],
        *,
        labels: Sequence[str] | None = None,
        encoder: Encoder | None = None,
    ) -> AddResult:
        """录入一个物体的多张图片并保存特征（仅内存，需再调用 ``save()``）。

        同一物体内按像素摘要去重：重复上传同一张图不会让"每物体向量条数"
        虚高，保证它始终等于有效注册图数。

        Args:
            name: 物体名称，首尾空白会被去掉；不能为空。
            images: 图片序列（路径 / PIL / numpy）。
            labels: 可选的图片标签，长度需与 ``images`` 一致；
                默认路径取文件名，其它输入生成 ``名称_001``。
            encoder: 自定义编码器，默认 ResNet18。

        Returns:
            ``AddResult``，包含新增条数、重复跳过条数与全库总数。

        Raises:
            ValueError: 名称为空、图片为空、标签数量不匹配或编码维度不符。
        """
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("物体名称不能为空")

        items = list(images)
        if not items:
            raise ValueError("至少提供一张注册图片")

        if labels is not None and isinstance(labels, str):
            raise ValueError("labels 必须是字符串列表，不能直接传单个字符串")
        if labels is not None and len(labels) != len(items):
            raise ValueError(
                f"labels 数量({len(labels)})必须与图片数量({len(items)})一致"
            )

        pil_images = [to_pil(item) for item in items]
        digests = [image_digest(image) for image in pil_images]
        image_labels = [
            str(labels[index]) if labels is not None else _default_label(item, clean_name, index)
            for index, item in enumerate(items)
        ]

        # 整个"查重 -> 编码 -> 追加"放在锁内：Gradio 多线程下不会重复写入；
        # 编码耗时约几十毫秒，串行化的代价可以接受。
        with self._lock:
            record = self._objects.get(clean_name)
            seen = {entry.digest for entry in record.entries} if record is not None else set()

            unique_indices: list[int] = []
            for index, digest in enumerate(digests):
                if digest in seen:
                    continue
                seen.add(digest)
                unique_indices.append(index)

            skipped = len(items) - len(unique_indices)
            if not unique_indices:
                total = self.total_vectors
                return AddResult(
                    object_name=clean_name,
                    added=0,
                    skipped=skipped,
                    object_vectors=len(record.entries) if record is not None else 0,
                    total_vectors=total,
                )

            vectors = extract_features(
                [pil_images[index] for index in unique_indices], encoder=encoder
            )
            if vectors.ndim != 2 or vectors.shape[1] != self.dim:
                raise ValueError(
                    f"特征维度不匹配：特征库要求 {self.dim} 维，"
                    f"编码器返回 {tuple(vectors.shape)}"
                )

            if record is None:
                record = ObjectRecord(name=clean_name)
                self._objects[clean_name] = record

            added_at = _now()
            for row, index in enumerate(unique_indices):
                record.entries.append(
                    Entry(
                        image=image_labels[index],
                        vector=vectors[row].astype(np.float32, copy=True),
                        digest=digests[index],
                        added_at=added_at,
                    )
                )
            self.updated_at = added_at
            return AddResult(
                object_name=clean_name,
                added=len(unique_indices),
                skipped=skipped,
                object_vectors=len(record.entries),
                total_vectors=self.total_vectors,
            )

    def set_threshold(self, value: float | None) -> None:
        """设置判未知阈值并更新时间戳（不自动保存）。

        Args:
            value: ``[-1, 1]`` 内的余弦相似度阈值，或 ``None`` 表示用默认值。

        Raises:
            ValueError: 阈值超出范围。
        """
        self.threshold = self._validate_threshold(value)
        self.updated_at = _now()

    # -------------------------------------------------------------- 持久化
    def to_dict(self) -> dict[str, Any]:
        """序列化为可 JSON 化的字典。

        向量保留 8 位小数：文件体积约为全精度的 1/3，而余弦相似度误差 < 1e-7，
        远小于图片本身带来的波动；加载时会再归一化一次修正舍入误差。
        """
        with self._lock:
            return {
                "schema": SCHEMA_NAME,
                "version": SCHEMA_VERSION,
                "model": self.model,
                "dim": self.dim,
                "normalized": True,
                "threshold": self.threshold,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "objects": [
                    {
                        "name": record.name,
                        "created_at": record.created_at,
                        "entries": [
                            {
                                "image": entry.image,
                                "digest": entry.digest,
                                "added_at": entry.added_at,
                                "vector": [round(float(value), 8) for value in entry.vector],
                            }
                            for entry in record.entries
                        ],
                    }
                    for record in self._objects.values()
                ],
            }

    def save(self, path: str | Path | None = None) -> Path:
        """原子写入 JSON 文件。

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
            raise ValueError("特征库未绑定路径，save() 需要显式传入 path")
        target = target.expanduser()

        with self._lock:
            self.updated_at = _now()
            payload = self.to_dict()
            _atomic_write_json(target, payload)
            # 绑定绝对路径：Gradio 的工作目录可能变化，相对路径会在下次保存时写错地方。
            self.path = target.resolve()

        return target.resolve()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, path: str | Path | None = None) -> FeatureLibrary:
        """从字典构造并做严格校验。

        Args:
            data: ``json.load`` 得到的字典。
            path: 可选绑定路径。

        Returns:
            校验通过的特征库实例。

        Raises:
            ValueError: schema/版本/维度/向量不合法，或存在重名物体。
        """
        if not isinstance(data, Mapping):
            raise ValueError("特征库 JSON 顶层必须是对象")

        schema = data.get("schema")
        if schema != SCHEMA_NAME:
            raise ValueError(f"不是本程序的特征库文件（schema={schema!r}）")

        version = data.get("version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"不支持的特征库版本 {version!r}，本程序只支持版本 {SCHEMA_VERSION}")

        try:
            dim = int(data["dim"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("特征库缺少合法的 dim 字段") from exc
        if dim < 1:
            raise ValueError(f"特征库 dim 非法: {dim}")

        if data.get("normalized", True) is not True:
            raise ValueError("特征库必须保存 L2 归一化向量（normalized=true）")

        library = cls(
            path=path,
            dim=dim,
            threshold=data.get("threshold"),
            model=str(data.get("model", MODEL_NAME)),
        )
        library.created_at = str(data.get("created_at") or _now())
        library.updated_at = str(data.get("updated_at") or library.created_at)

        objects = data.get("objects", [])
        if not isinstance(objects, list):
            raise ValueError("特征库 objects 字段必须是数组")

        for object_index, raw_object in enumerate(objects):
            if not isinstance(raw_object, Mapping):
                raise ValueError(f"objects[{object_index}] 必须是对象")
            name = str(raw_object.get("name", "")).strip()
            if not name:
                raise ValueError(f"objects[{object_index}] 缺少物体名称")
            if name in library._objects:
                raise ValueError(f"特征库中存在重名物体「{name}」")

            record = ObjectRecord(
                name=name,
                created_at=str(raw_object.get("created_at") or _now()),
            )
            raw_entries = raw_object.get("entries", [])
            if not isinstance(raw_entries, list):
                raise ValueError(f"物体「{name}」的 entries 必须是数组")

            seen_digests: set[str] = set()
            for entry_index, raw_entry in enumerate(raw_entries):
                if not isinstance(raw_entry, Mapping):
                    raise ValueError(f"物体「{name}」第 {entry_index + 1} 条记录必须是对象")

                try:
                    vector = np.asarray(raw_entry["vector"], dtype=np.float32)
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"物体「{name}」第 {entry_index + 1} 条记录缺少合法 vector"
                    ) from exc

                if vector.shape != (dim,):
                    raise ValueError(
                        f"物体「{name}」第 {entry_index + 1} 条向量维度错误："
                        f"期望 ({dim},)，实际 {tuple(vector.shape)}"
                    )
                if not np.isfinite(vector).all():
                    raise ValueError(f"物体「{name}」第 {entry_index + 1} 条向量含 NaN/Inf")

                norm = float(np.linalg.norm(vector))
                if norm < 1e-6:
                    raise ValueError(f"物体「{name}」第 {entry_index + 1} 条向量是零向量")
                if abs(norm - 1.0) > 1e-3:
                    raise ValueError(
                        f"物体「{name}」第 {entry_index + 1} 条向量未归一化（范数 {norm:.6f}）"
                    )

                digest = str(raw_entry.get("digest", ""))
                if digest and digest in seen_digests:
                    raise ValueError(f"物体「{name}」存在重复图片（digest 重复）")
                if digest:
                    seen_digests.add(digest)

                record.entries.append(
                    Entry(
                        image=str(raw_entry.get("image") or f"{name}_{entry_index + 1:03d}"),
                        vector=(vector / norm).astype(np.float32, copy=False),
                        digest=digest,
                        added_at=str(raw_entry.get("added_at") or record.created_at),
                    )
                )

            library._objects[name] = record

        return library

    @staticmethod
    def _validate_threshold(value: float | None) -> float | None:
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"阈值必须是数字或 None，收到 {value!r}") from exc
        if not np.isfinite(numeric) or not -1.0 <= numeric <= 1.0:
            raise ValueError(f"余弦相似度阈值必须在 [-1, 1] 内，收到 {numeric}")
        return numeric


def _default_label(image: ImageInput, name: str, index: int) -> str:
    """给没有显式标签的图片起一个可读名字。"""
    if isinstance(image, (str, Path)):
        return Path(image).name
    return f"{name}_{index + 1:03d}"


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
        # 失败时清理临时文件，避免特征库目录里堆满 .tmp。
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def load_library(
    path: str | Path | None = None,
    *,
    create: bool = True,
) -> FeatureLibrary:
    """从 JSON 文件加载特征库。

    Args:
        path: 文件路径；默认 ``data/features.json``。
        create: 文件不存在时是否返回空库。``False`` 时抛 ``FileNotFoundError``。

    Returns:
        加载后的特征库，已绑定文件路径。

    Raises:
        FileNotFoundError: ``create=False`` 且文件不存在。
        ValueError: JSON 损坏或内容不符合 schema。
    """
    target = (Path(path) if path is not None else DEFAULT_LIBRARY_PATH).expanduser().resolve()

    if not target.exists():
        if not create:
            raise FileNotFoundError(f"特征库不存在: {target}")
        return FeatureLibrary(path=target)

    try:
        raw = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise OSError(f"读取特征库失败: {target}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"特征库 JSON 解析失败（文件可能损坏）: {target}: {exc}") from exc

    library = FeatureLibrary.from_dict(data, path=target)
    return library


def save_library(library: FeatureLibrary, path: str | Path | None = None) -> Path:
    """保存特征库（``FeatureLibrary.save`` 的函数式包装）。"""
    return library.save(path)


def add_to_library(
    library: FeatureLibrary,
    name: str,
    images: Iterable[ImageInput],
    *,
    labels: Sequence[str] | None = None,
    encoder: Encoder | None = None,
) -> AddResult:
    """向特征库添加物体图片（``FeatureLibrary.add`` 的函数式包装）。

    与 ``add`` 一样只改内存；需要落盘时再调用 ``save_library(library)``。
    """
    return library.add(name, list(images), labels=labels, encoder=encoder)
