"""``core/background.py`` 测试：中位数学习、丢帧、余弦边界与持久化。

全部使用 512 维假特征 / 合成纯色帧，不加载 ResNet18、不依赖摄像头。
重点验证任务契约里的四件事：

1. 收集 20 帧、丢前 2 帧、逐维中位数再 L2 归一化；
2. 丢弃预热帧后不足 10 帧必须报错；
3. 余弦相似度边界 1 / 0 / -1；
4. ``save`` -> ``load`` 往返一致，非 512 维或未归一化必须拒绝。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from helpers import color_encoder, one_hot, solid

from core.background import (
    DEFAULT_BACKGROUND_PATH,
    MIN_SAMPLES,
    BackgroundModel,
    load_background,
    save_background,
)
from core.library import FeatureLibrary

DIM = 512
E0 = one_hot(0, DIM)
E1 = one_hot(1, DIM)

# 12 帧：前 7 帧 E1（含开头 2 帧预热），后 5 帧 E0。
# 丢弃前 2 帧后剩 5 个 E1 + 5 个 E0，逐维中位数 = [0.5, 0.5, 0, ...]。
COLORS = [(index + 1, index + 1, index + 1) for index in range(12)]
ENCODER = color_encoder(
    {
        color: (E1 if index < 7 else E0)
        for index, color in enumerate(COLORS)
    }
)
FRAMES = [solid(color) for color in COLORS]


def test_collect_takes_median_after_dropping_first_two_frames() -> None:
    model = BackgroundModel()

    vector = model.collect(FRAMES, n_frames=12, drop_first=2, encoder=ENCODER)

    # 手算：5 个 [1,0] + 5 个 [0,1] 的逐维中位数是 [0.5, 0.5]，再归一化。
    expected = np.zeros(DIM, dtype=np.float32)
    expected[0] = expected[1] = 1.0
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(vector, expected, atol=1e-6)
    np.testing.assert_allclose(model.vector, expected, atol=1e-6)
    assert model.sample_count == 10
    assert model.dim == DIM
    assert float(np.linalg.norm(model.vector)) == pytest.approx(1.0, abs=1e-6)


def test_collect_uses_first_n_frames_and_rejects_too_few_samples() -> None:
    # 采集窗口只有 10 帧，丢掉前 2 帧剩 8 帧 < 10：必须报错，不能拿半截数据凑。
    with pytest.raises(ValueError, match="至少需要 10 帧"):
        BackgroundModel().collect(FRAMES[:10], n_frames=10, drop_first=2, encoder=ENCODER)

    # 参数边界：drop_first 不能吃掉整个采集窗口。
    with pytest.raises(ValueError, match="drop_first"):
        BackgroundModel().collect(FRAMES[:12], n_frames=2, drop_first=2, encoder=ENCODER)


def test_similarity_boundaries_are_one_zero_minus_one() -> None:
    model = BackgroundModel(vector=E0)

    assert model.similarity(E0) == pytest.approx(1.0, abs=1e-6)
    assert model.similarity(E1) == pytest.approx(0.0, abs=1e-6)
    assert model.similarity(-E0) == pytest.approx(-1.0, abs=1e-6)
    # 未归一化的查询向量也按余弦处理。
    assert model.similarity(E0 * 7.5) == pytest.approx(1.0, abs=1e-6)


def test_is_background_uses_strict_greater_than() -> None:
    model = BackgroundModel(vector=E0)

    assert model.is_background(E0, 0.9) is True
    assert model.is_background(E1, 0.9) is False
    # 恰好等于阈值不算背景：边界样本交给物体检索再判一次。
    assert model.is_background(E0, 1.0) is False


def test_similarity_rejects_dimension_mismatch() -> None:
    model = BackgroundModel(vector=E0)

    with pytest.raises(ValueError, match="维度"):
        model.similarity(np.ones(8, dtype=np.float32))


def test_save_load_roundtrip_keeps_vector_and_metadata(tmp_path: Path) -> None:
    model = BackgroundModel()
    model.collect(FRAMES, n_frames=12, drop_first=2, encoder=ENCODER)
    path = tmp_path / "background.json"

    model.save(path)

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert {"version", "created_at", "dim", "vector", "sample_count"} <= set(raw)
    assert raw["version"] == 1
    assert raw["dim"] == DIM
    assert raw["sample_count"] == 10
    assert len(raw["vector"]) == DIM
    assert "objects" not in raw  # 背景文件绝不是物体库

    loaded = BackgroundModel.load(path)
    np.testing.assert_allclose(loaded.vector, model.vector, atol=1e-6)
    assert loaded.sample_count == model.sample_count
    assert loaded.created_at == model.created_at
    assert loaded.model == model.model
    assert loaded.path == path.resolve()
    assert loaded.dim == DIM


def test_load_rejects_wrong_dim_and_unnormalized_vector(tmp_path: Path) -> None:
    base = {
        "version": 1,
        "created_at": "2026-11-09T00:00:00Z",
        "dim": DIM,
        "sample_count": 18,
        "vector": [1.0] + [0.0] * (DIM - 1),
    }

    wrong_dim = dict(base, dim=256)
    path = tmp_path / "wrong_dim.json"
    path.write_text(json.dumps(wrong_dim), encoding="utf-8")
    with pytest.raises(ValueError, match="维度"):
        BackgroundModel.load(path)

    wrong_length = dict(base, vector=[1.0] + [0.0] * (DIM - 2))
    path.write_text(json.dumps(wrong_length), encoding="utf-8")
    with pytest.raises(ValueError, match="长度"):
        BackgroundModel.load(path)

    unnormalized = dict(base, vector=[3.0] + [0.0] * (DIM - 1))
    path.write_text(json.dumps(unnormalized), encoding="utf-8")
    with pytest.raises(ValueError, match="未归一化"):
        BackgroundModel.load(path)

    bad_version = dict(base, version=2)
    path.write_text(json.dumps(bad_version), encoding="utf-8")
    with pytest.raises(ValueError, match="版本"):
        BackgroundModel.load(path)


def test_load_missing_file_can_return_none(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"

    assert load_background(missing, create=True) is None
    with pytest.raises(FileNotFoundError):
        load_background(missing)
    with pytest.raises(FileNotFoundError):
        BackgroundModel.load(missing)


def test_functional_save_wrapper_and_default_path(tmp_path: Path) -> None:
    model = BackgroundModel(vector=E0, sample_count=18)
    path = tmp_path / "background.json"

    save_background(model, path)

    assert json.loads(path.read_text(encoding="utf-8"))["sample_count"] == 18
    assert DEFAULT_BACKGROUND_PATH.name == "background.json"
    assert DEFAULT_BACKGROUND_PATH.parent.name == "data"
    assert MIN_SAMPLES == 10


def test_background_does_not_touch_library(tmp_path: Path) -> None:
    """背景向量与 library.json 彻底分离：库哈希、物体数、条数都不变。"""
    encoder = color_encoder({(10, 10, 10): E0})
    library = FeatureLibrary(path=tmp_path / "features.json", dim=DIM)
    library.add("苹果", [solid((10, 10, 10))], encoder=encoder)
    library.save()
    before = library.path.read_bytes()

    model = BackgroundModel(vector=E1, sample_count=18)
    model.save(tmp_path / "background.json")

    assert library.path.read_bytes() == before
    assert library.object_count == 1
    assert library.total_vectors == 1
    assert "objects" not in json.loads(
        (tmp_path / "background.json").read_text(encoding="utf-8")
    )
