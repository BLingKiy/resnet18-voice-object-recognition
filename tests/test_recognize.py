"""检索与判未知测试：Top-1、阈值优先级、空库与边界情况。"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import BLUE, DIM, GREEN, RED, color_encoder, one_hot, solid

from core.library import FeatureLibrary
from core.recognize import (
    DEFAULT_THRESHOLD,
    UNKNOWN_LABEL,
    Recognition,
    judge_unknown,
    object_scores,
    recognize,
    resolve_threshold,
    retrieve,
)

APPLE_SECOND = (221, 41, 40)


def _recognition_encoder():
    """苹果有两条视角差异明显的注册特征，用来验证 max/mean 聚合。"""
    return color_encoder(
        {
            RED: one_hot(0),
            APPLE_SECOND: np.array([0.8, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            GREEN: one_hot(1),
            BLUE: one_hot(2),
        }
    )


@pytest.fixture
def recognition_library(empty_library: FeatureLibrary) -> FeatureLibrary:
    encoder = _recognition_encoder()
    empty_library.add("苹果", [solid(RED), solid(APPLE_SECOND)], encoder=encoder)
    empty_library.add("香蕉", [solid(GREEN)], encoder=encoder)
    empty_library.add("杯子", [solid(BLUE)], encoder=encoder)
    return empty_library


def test_retrieve_top1_ranking_and_ranks(recognition_library: FeatureLibrary) -> None:
    matches = retrieve(recognition_library, one_hot(0), top_k=5)

    assert len(matches) == 4
    assert matches[0].name == "苹果"
    assert matches[0].similarity == pytest.approx(1.0, abs=1e-6)
    assert [match.rank for match in matches] == [1, 2, 3, 4]
    similarities = [match.similarity for match in matches]
    assert similarities == sorted(similarities, reverse=True)


def test_retrieve_top_k_limit(recognition_library: FeatureLibrary) -> None:
    assert len(retrieve(recognition_library, one_hot(0), top_k=2)) == 2
    assert len(retrieve(recognition_library, one_hot(0), top_k=99)) == 4


def test_recognize_known(recognition_library: FeatureLibrary) -> None:
    result = recognize(recognition_library, one_hot(0), threshold=0.6, top_k=2)

    assert result.is_known is True
    assert result.name == "苹果"
    assert result.similarity == pytest.approx(1.0, abs=1e-6)
    assert result.threshold == 0.6
    assert len(result.matches) == 2
    assert result.matches[0].name == "苹果"
    assert result.similarity_percent == pytest.approx(100.0)


def test_recognize_unknown_orthogonal_vector(recognition_library: FeatureLibrary) -> None:
    result = recognize(recognition_library, one_hot(7), threshold=0.6)

    assert result.is_known is False
    assert result.name == UNKNOWN_LABEL
    assert result.similarity == pytest.approx(0.0, abs=1e-6)
    assert result.similarity_percent == 0.0
    assert result.matches  # 仍然返回近邻，界面可以展示"最像的候选"


def test_recognize_negative_similarity() -> None:
    library = FeatureLibrary(dim=DIM)
    library.add("苹果", [solid(RED)], encoder=_recognition_encoder())

    result = recognize(library, -one_hot(0), threshold=0.6)

    assert result.is_known is False
    assert result.similarity == pytest.approx(-1.0, abs=1e-6)
    assert result.similarity_percent == 0.0


def test_threshold_boundary_counts_as_known(recognition_library: FeatureLibrary) -> None:
    # 与 e0 的余弦为 0.6，与苹果第二条特征 [0.8, 0.6] 只有 0.48；
    # 第三个分量放在未使用的 e7 上，避免和杯子撞上。
    query = np.zeros(DIM, dtype=np.float32)
    query[0] = 0.6
    query[7] = 0.8

    score = recognize(recognition_library, query, threshold=0.5).similarity
    assert score == pytest.approx(0.6, abs=1e-6)

    assert recognize(recognition_library, query, threshold=score).is_known is True
    assert recognize(recognition_library, query, threshold=0.61).is_known is False


def test_library_threshold_used_when_argument_is_none(recognition_library: FeatureLibrary) -> None:
    query = np.zeros(DIM, dtype=np.float32)
    query[0] = 0.6
    query[7] = 0.8
    recognition_library.set_threshold(0.95)

    by_library = recognize(recognition_library, query)
    overridden = recognize(recognition_library, query, threshold=0.5)

    assert by_library.is_known is False
    assert overridden.is_known is True


def test_explicit_threshold_overrides_library(recognition_library: FeatureLibrary) -> None:
    recognition_library.set_threshold(0.99)
    query = np.zeros(DIM, dtype=np.float32)
    query[0] = 0.6
    query[7] = 0.8

    assert recognize(recognition_library, query).is_known is False
    assert recognize(recognition_library, query, threshold=0.5).is_known is True


def test_empty_library_returns_unknown() -> None:
    library = FeatureLibrary(dim=DIM)

    result = recognize(library, one_hot(0))

    assert result.is_known is False
    assert result.name == UNKNOWN_LABEL
    assert result.similarity == 0.0
    assert result.threshold == DEFAULT_THRESHOLD
    assert result.matches == ()
    assert result.object_scores == ()


def test_retrieve_on_empty_library_does_not_call_encoder() -> None:
    library = FeatureLibrary(dim=DIM)

    def should_not_run(images):
        raise AssertionError("空库不应加载模型")

    assert retrieve(library, solid(RED), encoder=should_not_run) == []


def test_object_scores_max_and_mean(recognition_library: FeatureLibrary) -> None:
    scores = object_scores(recognition_library, one_hot(0))

    assert [score.name for score in scores] == ["苹果", "杯子", "香蕉"]
    assert scores[0].count == 2
    assert scores[0].max_similarity == pytest.approx(1.0, abs=1e-6)
    assert scores[0].mean_similarity == pytest.approx(0.9, abs=1e-6)
    assert scores[0].best_image == "苹果_001"
    assert scores[1].max_similarity == pytest.approx(0.0, abs=1e-6)


def test_recognize_image_with_encoder(recognition_library: FeatureLibrary) -> None:
    result = recognize(
        recognition_library,
        solid(GREEN),
        threshold=0.6,
        encoder=_recognition_encoder(),
    )

    assert result.is_known is True
    assert result.name == "香蕉"


def test_query_dimension_mismatch(recognition_library: FeatureLibrary) -> None:
    with pytest.raises(ValueError, match="维度"):
        recognize(recognition_library, np.zeros(DIM + 1, dtype=np.float32))


def test_zero_query_vector_rejected(recognition_library: FeatureLibrary) -> None:
    with pytest.raises(ValueError, match="零向量"):
        recognize(recognition_library, np.zeros(DIM, dtype=np.float32))


def test_invalid_top_k(recognition_library: FeatureLibrary) -> None:
    with pytest.raises(ValueError):
        retrieve(recognition_library, one_hot(0), top_k=0)
    with pytest.raises(ValueError):
        recognize(recognition_library, one_hot(0), top_k=0)


def test_resolve_threshold_priority(recognition_library: FeatureLibrary) -> None:
    assert resolve_threshold(recognition_library, 0.4) == 0.4
    assert resolve_threshold(recognition_library) == DEFAULT_THRESHOLD

    recognition_library.set_threshold(0.7)
    assert resolve_threshold(recognition_library) == 0.7
    assert resolve_threshold(recognition_library, 0.4) == 0.4


def test_judge_unknown_boundary() -> None:
    assert judge_unknown(0.59, 0.6) is True
    assert judge_unknown(0.6, 0.6) is False
    assert judge_unknown(0.61, 0.6) is False
    assert judge_unknown(float("nan"), 0.6) is True


def test_similarity_percent_is_clamped() -> None:
    assert Recognition(True, "苹果", 1.0000001, 0.6).similarity_percent == 100.0
    assert Recognition(False, UNKNOWN_LABEL, -0.5, 0.6).similarity_percent == 0.0
