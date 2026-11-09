"""特征库测试：计数、去重、持久化、损坏检测与原子写入。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from helpers import BLUE, DIM, GREEN, RED, YELLOW, color_encoder, one_hot, solid

import core.library as library_module
from core.library import (
    SCHEMA_NAME,
    FeatureLibrary,
    add_to_library,
    load_library,
    save_library,
)


def _encoder():
    return color_encoder({RED: one_hot(0), GREEN: one_hot(1), BLUE: one_hot(2)})


def test_add_counts_vectors_and_stats(empty_library: FeatureLibrary) -> None:
    result = empty_library.add(
        "苹果",
        [solid(RED), solid(GREEN), solid(BLUE)],
        encoder=_encoder(),
    )

    assert (result.added, result.skipped) == (3, 0)
    assert (result.object_vectors, result.total_vectors) == (3, 3)
    assert empty_library.object_count == 1
    assert empty_library.total_vectors == 3
    assert len(empty_library) == 3

    record = empty_library.get("苹果")
    assert record is not None
    assert [entry.image for entry in record.entries] == ["苹果_001", "苹果_002", "苹果_003"]
    for entry in record.entries:
        assert entry.vector.shape == (DIM,)
        np.testing.assert_allclose(np.linalg.norm(entry.vector), 1.0, atol=1e-6)

    stats = empty_library.stats()
    assert stats == [
        {"name": "苹果", "count": 3, "created_at": record.created_at, "first_image": "苹果_001"}
    ]


def test_add_from_file_paths_uses_basenames(empty_library: FeatureLibrary, tmp_path: Path) -> None:
    first = tmp_path / "red.png"
    second = tmp_path / "green.png"
    solid(RED).save(first)
    solid(GREEN).save(second)

    result = add_to_library(empty_library, "苹果", [first, second], encoder=_encoder())

    assert result.added == 2
    record = empty_library.get("苹果")
    assert record is not None
    assert [entry.image for entry in record.entries] == ["red.png", "green.png"]


def test_duplicate_images_are_skipped(empty_library: FeatureLibrary) -> None:
    first = empty_library.add("苹果", [solid(RED), solid(RED)], encoder=_encoder())
    assert (first.added, first.skipped) == (1, 1)

    second = empty_library.add("苹果", [solid(RED)], encoder=_encoder())
    assert (second.added, second.skipped) == (0, 1)
    assert second.total_vectors == 1


def test_encoder_failure_leaves_library_untouched(empty_library: FeatureLibrary) -> None:
    def broken(images):
        raise RuntimeError("模拟模型崩溃")

    with pytest.raises(RuntimeError):
        empty_library.add("苹果", [solid(RED)], encoder=broken)

    assert empty_library.total_vectors == 0
    assert empty_library.object_count == 0


def test_dimension_mismatch_leaves_library_untouched(empty_library: FeatureLibrary) -> None:
    def wrong_dim(images):
        return np.ones((len(images), 4), dtype=np.float32)

    with pytest.raises(ValueError, match="特征维度不匹配"):
        empty_library.add("苹果", [solid(RED)], encoder=wrong_dim)

    assert empty_library.total_vectors == 0


def test_add_validates_inputs(empty_library: FeatureLibrary) -> None:
    with pytest.raises(ValueError, match="名称不能为空"):
        empty_library.add("   ", [solid(RED)], encoder=_encoder())
    with pytest.raises(ValueError, match="至少提供一张"):
        empty_library.add("苹果", [], encoder=_encoder())
    with pytest.raises(ValueError, match="labels 数量"):
        empty_library.add("苹果", [solid(RED)], labels=["a", "b"], encoder=_encoder())


def test_save_and_reload_roundtrip(empty_library: FeatureLibrary) -> None:
    empty_library.add("苹果", [solid(RED), solid(GREEN)], encoder=_encoder())
    empty_library.add("香蕉", [solid(BLUE)], encoder=_encoder())
    empty_library.set_threshold(0.62)

    path = empty_library.save()
    raw = json.loads(path.read_text(encoding="utf-8"))

    assert raw["schema"] == SCHEMA_NAME
    assert raw["version"] == 1
    assert raw["dim"] == DIM
    assert raw["normalized"] is True
    assert raw["threshold"] == 0.62
    assert [len(obj["entries"]) for obj in raw["objects"]] == [2, 1]
    assert all(len(entry["vector"]) == DIM for obj in raw["objects"] for entry in obj["entries"])
    assert "苹果" in path.read_text(encoding="utf-8")  # ensure_ascii=False，报告里可直接阅读

    reloaded = load_library(path)
    assert reloaded.object_count == 2
    assert reloaded.total_vectors == 3
    assert reloaded.threshold == 0.62
    assert reloaded.get("苹果") is not None
    assert reloaded.get("苹果").entries[0].image == "苹果_001"
    np.testing.assert_allclose(
        reloaded.get("苹果").entries[0].vector,
        empty_library.get("苹果").entries[0].vector,
        atol=1e-6,
    )


def test_save_requires_path() -> None:
    library = FeatureLibrary(dim=DIM)

    with pytest.raises(ValueError, match="未绑定路径"):
        library.save()


def test_load_missing_library(tmp_path: Path) -> None:
    path = tmp_path / "features.json"

    empty = load_library(path)
    assert empty.total_vectors == 0
    assert empty.path == path

    with pytest.raises(FileNotFoundError):
        load_library(path, create=False)


def test_load_corrupt_json(tmp_path: Path) -> None:
    path = tmp_path / "features.json"
    path.write_text("{ not valid json", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON 解析失败"):
        load_library(path)


def test_load_rejects_wrong_schema(tmp_path: Path) -> None:
    path = tmp_path / "features.json"
    path.write_text(
        json.dumps({"schema": "something-else", "version": 1, "dim": DIM, "objects": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="不是本程序的特征库"):
        load_library(path)


def test_load_rejects_wrong_version(tmp_path: Path) -> None:
    path = tmp_path / "features.json"
    path.write_text(
        json.dumps({"schema": SCHEMA_NAME, "version": 99, "dim": DIM, "objects": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="不支持的特征库版本"):
        load_library(path)


def test_load_rejects_dimension_mismatch(empty_library: FeatureLibrary) -> None:
    empty_library.add("苹果", [solid(RED)], encoder=_encoder())
    path = empty_library.save()
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["dim"] = DIM + 1
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="向量维度错误"):
        load_library(path)


def test_load_rejects_unnormalized_vector(empty_library: FeatureLibrary) -> None:
    empty_library.add("苹果", [solid(RED)], encoder=_encoder())
    path = empty_library.save()
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["objects"][0]["entries"][0]["vector"][0] = 2.0
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="未归一化"):
        load_library(path)


def test_load_rejects_duplicate_digest(empty_library: FeatureLibrary) -> None:
    empty_library.add("苹果", [solid(RED), solid(GREEN)], encoder=_encoder())
    path = empty_library.save()
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw["objects"][0]["entries"]
    entries[1]["digest"] = entries[0]["digest"]
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="重复图片"):
        load_library(path)


def test_load_rejects_duplicate_object_names(empty_library: FeatureLibrary) -> None:
    empty_library.add("苹果", [solid(RED)], encoder=_encoder())
    path = empty_library.save()
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["objects"].append(raw["objects"][0])
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="重名物体"):
        load_library(path)


def test_threshold_validation(empty_library: FeatureLibrary) -> None:
    empty_library.set_threshold(0.0)
    assert empty_library.threshold == 0.0
    empty_library.set_threshold(None)
    assert empty_library.threshold is None

    for invalid in (-1.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            empty_library.set_threshold(invalid)


def test_save_is_atomic_and_preserves_old_file(
    empty_library: FeatureLibrary, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty_library.add("苹果", [solid(RED)], encoder=_encoder())
    path = empty_library.save()
    original = path.read_text(encoding="utf-8")

    def broken_dump(*args, **kwargs):
        raise RuntimeError("模拟磁盘写满")

    monkeypatch.setattr(library_module.json, "dump", broken_dump)

    with pytest.raises(RuntimeError):
        empty_library.save()

    assert path.read_text(encoding="utf-8") == original
    assert list(path.parent.glob("*.tmp")) == []
    load_library(path)  # 旧文件仍然可读


def test_stats_sorted_by_name_and_matrix_shape(empty_library: FeatureLibrary) -> None:
    encoder = color_encoder(
        {RED: one_hot(0), GREEN: one_hot(1), BLUE: one_hot(2), YELLOW: one_hot(3)}
    )
    empty_library.add("香蕉", [solid(RED)], encoder=encoder)
    empty_library.add("苹果", [solid(GREEN), solid(BLUE)], encoder=encoder)

    names = [row["name"] for row in empty_library.stats()]
    assert names == sorted(names)

    vector_names, images, matrix = empty_library.matrix()
    assert matrix.shape == (3, DIM)
    assert len(vector_names) == len(images) == 3
    assert set(vector_names) == {"苹果", "香蕉"}

    empty_names, empty_images, empty_matrix = FeatureLibrary(dim=DIM).matrix()
    assert empty_names == [] and empty_images == []
    assert empty_matrix.shape == (0, DIM)


def test_norm_after_reload_and_json_counts_per_object(empty_library: FeatureLibrary) -> None:
    """直接对应验收判据：每物体向量条数 == 注册图数，且向量 512 维、范数≈1。

    单测里把维度缩到 8；真实 512 维由集成测试覆盖。
    """
    colors = [RED, GREEN, BLUE, YELLOW, (10, 20, 30)]
    encoder = color_encoder({color: one_hot(index) for index, color in enumerate(colors)})
    for object_index, color in enumerate(colors):
        images = [solid(color, (64 + offset, 64 + offset)) for offset in range(5)]
        empty_library.add(f"物体{object_index + 1}", images, encoder=encoder)

    path = empty_library.save()
    raw = json.loads(path.read_text(encoding="utf-8"))
    reloaded = load_library(path)

    assert reloaded.total_vectors == 25
    assert all(len(obj["entries"]) == 5 for obj in raw["objects"])
    for record in (reloaded.get(f"物体{index + 1}") for index in range(5)):
        assert record is not None
        assert len(record.entries) == 5
        for entry in record.entries:
            assert entry.vector.shape == (DIM,)
            np.testing.assert_allclose(np.linalg.norm(entry.vector), 1.0, atol=1e-5)


def test_save_library_wrapper_returns_absolute_path(empty_library: FeatureLibrary) -> None:
    saved = save_library(empty_library)
    assert saved.is_absolute()
    assert saved.exists()
