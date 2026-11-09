"""``core/background.py`` 测试：中位数学习、丢帧、相似度与持久化。

全部使用 ``tests/helpers.py`` 的假编码器，不加载 ResNet18；
重点验证"背景文件与特征库彻底分离"这条红线。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from helpers import DIM, one_hot, solid

from core.background import (
    DEFAULT_BACKGROUND_PATH,
    BackgroundModel,
    load_background,
    save_background,
)
from core.library import FeatureLibrary

A = (10, 10, 10)
B = (60, 60, 60)
C = (120, 120, 120)
D = (200, 200, 200)


def _encoder():
    """按颜色返回可手算的向量；extract_features 会统一归一化。"""
    from helpers import color_encoder

    return color_encoder(
        {
            A: np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            B: np.array([0.6, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            C: np.array([0.8, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            D: np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }
    )


def test_learn_takes_median_after_dropping_first_frames() -> None:
    frames = [solid(A), solid(B), solid(C), solid(D)]

    model = BackgroundModel.learn(frames, encoder=_encoder(), take=2, drop_first=1)

    # 丢 A，取 B/C；逐维中位数 = [0.7, 0.7]，归一化后每维 1/sqrt(2)。
    expected = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(model.vector, expected, atol=1e-6)
    assert model.frames_used == 2
    assert model.dim == DIM
    assert float(np.linalg.norm(model.vector)) == pytest.approx(1.0, abs=1e-6)


def test_learn_take_is_capped_by_available_frames() -> None:
    frames = [solid(A), solid(B), solid(C)]

    model = BackgroundModel.learn(frames, encoder=_encoder(), take=20, drop_first=2)

    assert model.frames_used == 1  # 只剩 C 一帧


def test_learn_rejects_insufficient_frames() -> None:
    with pytest.raises(ValueError, match="背景帧不足"):
        BackgroundModel.learn([solid(A), solid(B)], encoder=_encoder(), drop_first=2)
    with pytest.raises(ValueError, match="take"):
        BackgroundModel.learn([solid(A)], encoder=_encoder(), take=0)


def test_similarity_of_same_scene_is_one_and_orthogonal_is_zero() -> None:
    model = BackgroundModel(vector=one_hot(0), frames_used=18)

    assert model.similarity(solid(A), encoder=_encoder()) == pytest.approx(1.0, abs=1e-6)
    assert model.similarity(solid(D), encoder=_encoder()) == pytest.approx(0.0, abs=1e-6)
    assert model.similarity_from_vector(one_hot(0)) == pytest.approx(1.0, abs=1e-6)


def test_similarity_rejects_dimension_mismatch() -> None:
    model = BackgroundModel(vector=one_hot(0))

    with pytest.raises(ValueError, match="维度"):
        model.similarity_from_vector(np.ones(4, dtype=np.float32))


def test_save_and_load_roundtrip_contract(tmp_path: Path) -> None:
    model = BackgroundModel.learn(
        [solid(A), solid(B), solid(C)], encoder=_encoder(), take=2, drop_first=1
    )
    path = tmp_path / "background.json"

    saved = save_background(model, path)
    assert saved == path.resolve()

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert set(raw) == {
        "schema",
        "version",
        "model",
        "dim",
        "normalized",
        "created_at",
        "frames_used",
        "vector",
    }
    assert raw["schema"] == "rn18-background"
    assert raw["version"] == 1
    assert raw["dim"] == DIM
    assert raw["normalized"] is True
    assert raw["frames_used"] == 2
    assert "objects" not in raw

    loaded = load_background(path)
    assert loaded is not None
    np.testing.assert_allclose(loaded.vector, model.vector, atol=1e-6)
    assert loaded.frames_used == model.frames_used
    assert loaded.path == path.resolve()


def test_load_missing_file_can_return_none(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"

    assert load_background(missing, create=True) is None
    with pytest.raises(FileNotFoundError):
        load_background(missing)


def test_background_is_separate_from_library_and_does_not_change_bytes(
    tmp_path: Path,
) -> None:
    library = FeatureLibrary(path=tmp_path / "features.json", dim=DIM)
    library.add("苹果", [solid(A)], encoder=_encoder())
    library.save()
    digest_before = hashlib.sha256(library.path.read_bytes()).hexdigest()

    model = BackgroundModel.learn([solid(B), solid(C)], encoder=_encoder(), take=1, drop_first=1)
    model.save(tmp_path / "background.json")

    digest_after = hashlib.sha256(library.path.read_bytes()).hexdigest()
    assert digest_after == digest_before
    assert library.object_count == 1
    assert library.total_vectors == 1

    background_raw = json.loads((tmp_path / "background.json").read_text(encoding="utf-8"))
    library_raw = json.loads(library.path.read_text(encoding="utf-8"))
    assert background_raw["schema"] == "rn18-background"
    assert library_raw["schema"] == "hifox-resnet18-feature-library"
    assert "vector" not in library_raw
    assert "objects" not in background_raw


def test_from_dict_rejects_wrong_schema_and_unnormalized_vector() -> None:
    base = BackgroundModel(vector=one_hot(0), frames_used=3).to_dict()

    with pytest.raises(ValueError, match="schema"):
        BackgroundModel.from_dict({**base, "schema": "something-else"})

    unnormalized = {**base, "vector": [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
    with pytest.raises(ValueError, match="未归一化"):
        BackgroundModel.from_dict(unnormalized)

    wrong_dim = {**base, "dim": 512}
    with pytest.raises(ValueError, match="dim"):
        BackgroundModel.from_dict(wrong_dim)


def test_default_path_is_sibling_of_library() -> None:
    assert DEFAULT_BACKGROUND_PATH.name == "background.json"
    assert DEFAULT_BACKGROUND_PATH.parent.name == "data"
