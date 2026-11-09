"""真实 ResNet18 端到端测试（默认跳过）。

运行方式::

    RUN_RESNET_TESTS=1 python -m pytest tests/test_integration_resnet.py -v

第一次运行会下载约 45MB 的 ImageNet 权重。测试使用程序生成的 5 类图案，
只验证"特征契约 + 检索链路 + 持久化"是否成立；**真实课程数据上的
Top-1 与阈值必须用 T2 采集的照片重新跑**，不能拿这里的合成图代替。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from core.features import FEATURE_DIM, extract_features, get_default_encoder
from core.library import FeatureLibrary, load_library
from core.recognize import UNKNOWN_LABEL, recognize, retrieve
from tools.demo_cli import main

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_RESNET_TESTS") != "1",
    reason="设置 RUN_RESNET_TESTS=1 才运行真实模型测试（首次会下载权重）",
)

CLASSES = ("circle", "square", "triangle", "stripes", "checker")


def _pattern(kind: str, shift: int, size: int = 160) -> Image.Image:
    """生成带颜色/形状/位置变化的图案，注册图与测试图使用不同 shift。"""
    image = Image.new("RGB", (size, size), (246, 246, 242))
    draw = ImageDraw.Draw(image)
    base = shift * 3

    if kind == "circle":
        draw.ellipse((30 + base, 30, 110 + base, 110), fill=(210, 45, 45))
    elif kind == "square":
        draw.rectangle((35, 35 + base, 115, 115 + base), fill=(40, 70, 210))
    elif kind == "triangle":
        draw.polygon(
            [(80, 25 + base), (25 + base, 130), (135, 130)], fill=(40, 185, 80)
        )
    elif kind == "stripes":
        for offset in range(20, size, 28):
            draw.rectangle((0, offset + base, size, offset + base + 12), fill=(235, 200, 30))
    elif kind == "checker":
        cell = 20
        for row in range(-cell, size, cell):
            for column in range(-cell, size, cell):
                if (row // cell + column // cell) % 2 == 0:
                    # 加 base 让测试图与注册图不是同一张（数据硬约束）。
                    left, top = column + base, row + base
                    draw.rectangle((left, top, left + cell, top + cell), fill=(60, 30, 90))
    else:  # pragma: no cover - 防御性分支
        raise ValueError(kind)

    return image


@pytest.fixture(scope="module")
def encoder():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    return get_default_encoder()


def test_resnet18_feature_contract(encoder) -> None:
    red = _pattern("circle", 0)
    red_variant = _pattern("circle", 1)
    blue = _pattern("square", 0)

    features = extract_features([red, red_variant, blue], encoder=encoder)

    assert features.shape == (3, FEATURE_DIM)
    assert features.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1.0, atol=1e-5)
    same = float(features[0] @ features[1])
    different = float(features[0] @ features[2])
    assert same > different, f"同物体相似度({same:.4f})应高于不同物体({different:.4f})"


def test_full_pipeline_counts_top1_persistence_and_unknown(
    tmp_path: Path, encoder
) -> None:
    library = FeatureLibrary(path=tmp_path / "features.json")
    for kind in CLASSES:
        images = [_pattern(kind, shift) for shift in range(5)]
        result = library.add(kind, images, encoder=encoder)
        assert result.added == 5

    path = library.save()
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert len(raw["objects"]) == 5
    assert all(len(obj["entries"]) == 5 for obj in raw["objects"])

    # 模拟进程重启：丢掉内存对象，只从 JSON 重新加载。
    reloaded = load_library(path)
    assert reloaded.total_vectors == 25
    assert reloaded.dim == FEATURE_DIM
    for kind in CLASSES:
        record = reloaded.get(kind)
        assert record is not None
        assert len(record.entries) == 5
        for entry in record.entries:
            assert entry.vector.shape == (FEATURE_DIM,)
            np.testing.assert_allclose(np.linalg.norm(entry.vector), 1.0, atol=1e-5)

    correct = 0
    for kind in CLASSES:
        test_image = _pattern(kind, 9)  # 注册时只用 0~4，测试用 9，保证不重复
        matches = retrieve(reloaded, test_image, top_k=1, encoder=encoder)
        if matches and matches[0].name == kind:
            correct += 1
    assert correct == len(CLASSES), f"Top-1 只对了 {correct}/{len(CLASSES)}"

    rng = np.random.default_rng(20261109)
    noise = Image.fromarray(rng.integers(0, 256, (160, 160, 3), dtype=np.uint8))
    unknown = recognize(reloaded, noise, threshold=0.99, encoder=encoder)
    assert unknown.is_known is False
    assert unknown.name == UNKNOWN_LABEL


def test_cli_add_recognize_eval(tmp_path: Path) -> None:
    data = tmp_path / "data"
    library_path = data / "features.json"
    for kind in ("circle", "square"):
        class_dir = data / "known" / kind
        class_dir.mkdir(parents=True)
        for shift in range(2):
            _pattern(kind, shift).save(class_dir / f"{shift}.jpg")

    for kind in ("circle", "square"):
        code = main(
            [
                "add",
                "--name",
                kind,
                "--dir",
                str(data / "known" / kind),
                "--library",
                str(library_path),
                "--json",
            ]
        )
        assert code == 0, kind

    test_root = data / "test"
    for kind in ("circle", "square"):
        class_dir = test_root / kind
        class_dir.mkdir(parents=True)
        _pattern(kind, 7).save(class_dir / "held_out.jpg")

    code = main(
        [
            "recognize",
            "--image",
            str(test_root / "circle" / "held_out.jpg"),
            "--library",
            str(library_path),
            "--json",
        ]
    )
    assert code == 0
    assert load_library(library_path).total_vectors == 4

    code = main(["eval", "--dir", str(test_root), "--library", str(library_path), "--json"])
    assert code == 0
