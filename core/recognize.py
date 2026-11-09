"""余弦相似度检索与未知物体判定。

为什么用"每物体取最大相似度"而不是"平均相似度"：
注册图是多视角的，某一张图可能恰好拍到了和查询图相同的角度。
取最大值相当于"只要有一张注册图很像，就认为这个物体很像"，
对多视角录入更友好；平均相似度会被没拍到的角度拖低，导致漏检。
代价是更容易被单张异常注册图带偏——所以注册图要真实、清晰、无重复。

判定规则：``最高相似度 < 阈值`` 即未知。阈值必须用**未参与录入**的
测试图校准（见 T6 / tools/calibrate.py），这里的 0.60 只是可运行的占位值。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .features import Encoder, ImageInput, extract_feature, l2_normalize
from .library import FeatureLibrary

UNKNOWN_LABEL = "未知物体"

# 占位阈值：ResNet18 在同类图片上的余弦相似度通常高于 0.7，
# 不同类通常低于 0.6。最终值必须用独立测试集校准后写入特征库。
DEFAULT_THRESHOLD = 0.60


@dataclass(frozen=True, slots=True)
class Match:
    """一条近邻检索结果（对应注册库里的一张图）。"""

    name: str
    image: str
    similarity: float
    rank: int


@dataclass(frozen=True, slots=True)
class ObjectScore:
    """一个物体的聚合得分。"""

    name: str
    max_similarity: float
    mean_similarity: float
    best_image: str
    count: int


@dataclass(frozen=True, slots=True)
class Recognition:
    """一次识别的完整结果，界面和命令行都只依赖这个对象。"""

    is_known: bool
    name: str
    similarity: float
    threshold: float
    matches: tuple[Match, ...] = ()
    object_scores: tuple[ObjectScore, ...] = ()

    @property
    def similarity_percent(self) -> float:
        """把余弦相似度映射成 0~100 的展示百分比（负相似度显示 0%）。"""
        return round(max(0.0, min(1.0, self.similarity)) * 100.0, 2)


def resolve_threshold(library: FeatureLibrary, threshold: float | None = None) -> float:
    """确定本次识别使用的阈值。

    优先级：显式参数 > 特征库中保存的校准值 > ``DEFAULT_THRESHOLD``。

    Args:
        library: 特征库。
        threshold: 调用方显式指定的阈值，``None`` 表示不指定。

    Returns:
        ``[-1, 1]`` 内的阈值。

    Raises:
        ValueError: 阈值不是有限数字或超出余弦相似度范围。
    """
    if threshold is not None:
        value = float(threshold)
    elif library.threshold is not None:
        value = float(library.threshold)
    else:
        value = DEFAULT_THRESHOLD

    if not np.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ValueError(f"阈值必须是 [-1, 1] 内的数字，收到 {value}")
    return value


def judge_unknown(similarity: float, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """判断相似度是否低于阈值。

    Args:
        similarity: 最高余弦相似度。
        threshold: 判定阈值。

    Returns:
        ``True`` 表示应判为未知物体；``False`` 表示认为是已知物体。
        相似度恰好等于阈值时算已知（边界偏向"认出来"）。
    """
    if not np.isfinite(similarity):
        return True
    return float(similarity) < float(threshold)


def _query_vector(
    image: ImageInput | NDArray[np.floating],
    library: FeatureLibrary,
    encoder: Encoder | None,
) -> NDArray[np.float32]:
    """把输入统一成与特征库同维度的归一化查询向量。"""
    if isinstance(image, np.ndarray) and image.ndim == 1:
        vector = np.asarray(image, dtype=np.float32)
        if vector.shape[0] != library.dim:
            raise ValueError(
                f"查询向量维度({vector.shape[0]})与特征库({library.dim})不一致"
            )
        if not np.isfinite(vector).all():
            raise ValueError("查询向量包含 NaN 或 Inf")
        if float(np.linalg.norm(vector)) < 1e-6:
            raise ValueError("查询向量是零向量")
        return l2_normalize(vector)

    # extract_feature 内部已经归一化，并在编码器输出未归一化时兜底。
    vector = extract_feature(image, encoder=encoder)
    if vector.shape[0] != library.dim:
        raise ValueError(
            f"查询特征维度({vector.shape[0]})与特征库({library.dim})不一致，"
            "请确认特征库与当前模型版本匹配"
        )
    return vector


def _object_scores(
    names: list[str],
    images: list[str],
    similarity: NDArray[np.float32],
) -> list[ObjectScore]:
    """按物体分组做 max/mean 聚合，并按最高相似度降序排列。"""
    grouped: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        grouped.setdefault(name, []).append(index)

    scores: list[ObjectScore] = []
    for name, indices in grouped.items():
        values = similarity[indices]
        best_position = int(np.argmax(values))
        scores.append(
            ObjectScore(
                name=name,
                max_similarity=float(values[best_position]),
                mean_similarity=float(np.mean(values)),
                best_image=images[indices[best_position]],
                count=len(indices),
            )
        )

    scores.sort(key=lambda item: (-item.max_similarity, item.name))
    return scores


def retrieve(
    library: FeatureLibrary,
    image: ImageInput | NDArray[np.floating],
    *,
    top_k: int = 5,
    encoder: Encoder | None = None,
) -> list[Match]:
    """检索与输入最相似的注册图片（图片级 top-k）。

    Args:
        library: 特征库。
        image: 查询图片，或已提取好的一维特征向量。
        top_k: 返回条数。
        encoder: 自定义编码器，默认 ResNet18。

    Returns:
        按相似度降序排列的 ``Match`` 列表；空库返回空列表。

    Raises:
        ValueError: top_k 非法、查询维度不匹配或查询向量是零向量。
    """
    if int(top_k) < 1:
        raise ValueError("top_k 必须 >= 1")

    if library.total_vectors == 0:
        return []

    names, images, matrix = library.matrix()
    query = _query_vector(image, library, encoder)
    similarities = matrix @ query

    limit = min(int(top_k), len(similarities))
    order = np.argsort(-similarities, kind="stable")[:limit]
    return [
        Match(
            name=names[int(index)],
            image=images[int(index)],
            similarity=float(similarities[int(index)]),
            rank=rank,
        )
        for rank, index in enumerate(order, start=1)
    ]


def object_scores(
    library: FeatureLibrary,
    image: ImageInput | NDArray[np.floating],
    *,
    encoder: Encoder | None = None,
) -> list[ObjectScore]:
    """返回每个物体的聚合得分，按最高相似度降序。

    该函数是阈值校准（T6）的主要入口：对测试图批量调用后，
    可以直接得到"同类最高分"和"未知物体最高分"两组分布。
    """
    if library.total_vectors == 0:
        return []

    names, images, matrix = library.matrix()
    query = _query_vector(image, library, encoder)
    return _object_scores(names, images, matrix @ query)


def recognize(
    library: FeatureLibrary,
    image: ImageInput | NDArray[np.floating],
    *,
    threshold: float | None = None,
    top_k: int = 5,
    encoder: Encoder | None = None,
) -> Recognition:
    """识别一张图片，返回名称、相似度与是否未知。

    Args:
        library: 特征库。
        image: 查询图片，或已提取好的一维特征向量。
        threshold: 显式阈值；``None`` 时使用特征库里的校准值/默认值。
        top_k: 返回的近邻条数，界面通常展示前 3 条。
        encoder: 自定义编码器，默认 ResNet18。

    Returns:
        ``Recognition``；未知时 ``name`` 为 ``"未知物体"``。

    Raises:
        ValueError: 阈值或 top_k 非法、查询维度不匹配。
    """
    if int(top_k) < 1:
        raise ValueError("top_k 必须 >= 1")

    resolved_threshold = resolve_threshold(library, threshold)

    if library.total_vectors == 0:
        return Recognition(
            is_known=False,
            name=UNKNOWN_LABEL,
            similarity=0.0,
            threshold=resolved_threshold,
        )

    names, images, matrix = library.matrix()
    query = _query_vector(image, library, encoder)
    similarities = matrix @ query

    scores = _object_scores(names, images, similarities)
    best = scores[0]

    limit = min(int(top_k), len(similarities))
    order = np.argsort(-similarities, kind="stable")[:limit]
    matches = tuple(
        Match(
            name=names[int(index)],
            image=images[int(index)],
            similarity=float(similarities[int(index)]),
            rank=rank,
        )
        for rank, index in enumerate(order, start=1)
    )

    unknown = judge_unknown(best.max_similarity, resolved_threshold)
    return Recognition(
        is_known=not unknown,
        name=UNKNOWN_LABEL if unknown else best.name,
        similarity=best.max_similarity,
        threshold=resolved_threshold,
        matches=matches,
        object_scores=tuple(scores),
    )
