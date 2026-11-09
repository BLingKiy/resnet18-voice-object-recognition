"""特征工具层测试：归一化、输入统一、批处理与摘要。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from helpers import DIM, RED, solid
from PIL import Image

from core.features import (
    extract_feature,
    extract_features,
    image_digest,
    l2_normalize,
    to_pil,
)


def test_l2_normalize_matrix_rows() -> None:
    matrix = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)

    normalized = l2_normalize(matrix)

    assert normalized.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), [1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(normalized[0], [0.6, 0.8], atol=1e-6)


def test_l2_normalize_single_vector() -> None:
    vector = l2_normalize(np.array([0.0, 2.0, 0.0], dtype=np.float32))

    assert vector.shape == (3,)
    np.testing.assert_allclose(vector, [0.0, 1.0, 0.0], atol=1e-6)


def test_l2_normalize_rejects_three_dimensions() -> None:
    with pytest.raises(ValueError):
        l2_normalize(np.zeros((2, 2, 2), dtype=np.float32))


def test_to_pil_from_grayscale_array() -> None:
    image = to_pil(np.full((8, 8), 128, dtype=np.uint8))

    assert image.mode == "RGB"
    assert image.size == (8, 8)
    assert image.getpixel((0, 0)) == (128, 128, 128)


def test_to_pil_from_rgba_and_float() -> None:
    rgba = np.zeros((4, 5, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 3] = 255
    assert to_pil(rgba).getpixel((0, 0)) == (255, 0, 0)

    floating = np.ones((4, 5, 3), dtype=np.float32)
    assert to_pil(floating).getpixel((0, 0)) == (255, 255, 255)


def test_to_pil_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        to_pil(np.zeros((4, 4, 2), dtype=np.uint8))


def test_to_pil_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        to_pil(tmp_path / "missing.jpg")


def test_to_pil_applies_exif_orientation(tmp_path: Path) -> None:
    # 手机竖拍常见 Orientation=6（顺时针旋转 90 度显示），不校正会让长宽颠倒。
    image = Image.new("RGB", (20, 10), RED)
    exif = image.getexif()
    exif[274] = 6
    path = tmp_path / "rotated.jpg"
    image.save(path, exif=exif)

    assert to_pil(path).size == (10, 20)


def test_image_digest_is_content_based() -> None:
    image = solid(RED)
    array = np.asarray(image)

    assert image_digest(image) == image_digest(array)
    assert image_digest(image) != image_digest(solid((0, 0, 255)))
    assert len(image_digest(image)) == 64


def test_extract_features_batches_and_normalizes() -> None:
    calls: list[int] = []

    def encoder(images: list[Image.Image]) -> np.ndarray:
        calls.append(len(images))
        return np.ones((len(images), DIM), dtype=np.float32)

    vectors = extract_features([solid(RED)] * 5, encoder=encoder, batch_size=2)

    assert calls == [2, 2, 1]
    assert vectors.shape == (5, DIM)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)


def test_extract_features_accepts_one_dimensional_single_output() -> None:
    def encoder(images: list[Image.Image]) -> np.ndarray:
        assert len(images) == 1
        return np.ones(DIM, dtype=np.float32)

    vectors = extract_features([solid(RED)], encoder=encoder)

    assert vectors.shape == (1, DIM)
    np.testing.assert_allclose(np.linalg.norm(vectors[0]), 1.0, atol=1e-6)


def test_extract_features_rejects_wrong_row_count() -> None:
    def encoder(images: list[Image.Image]) -> np.ndarray:
        return np.ones((2, DIM), dtype=np.float32)

    with pytest.raises(ValueError, match="输出形状不合法"):
        extract_features([solid(RED)], encoder=encoder)


def test_extract_features_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        extract_features([], encoder=lambda images: np.zeros((0, DIM)))


def test_extract_features_rejects_bad_batch_size() -> None:
    with pytest.raises(ValueError):
        extract_features([solid(RED)], encoder=lambda images: np.zeros((1, DIM)), batch_size=0)


def test_extract_feature_single_shape() -> None:
    def encoder(images: list[Image.Image]) -> np.ndarray:
        return np.ones((len(images), DIM), dtype=np.float32)

    vector = extract_feature(solid(RED), encoder=encoder)

    assert vector.shape == (DIM,)
    assert vector.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vector), 1.0, atol=1e-6)
