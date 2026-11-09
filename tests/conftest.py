"""pytest 共享夹具。

测试策略：
- 特征库 / 检索 / 判未知的逻辑用**假编码器**验证，不依赖 torch，秒级完成；
- 真实 ResNet18 的端到端测试单独放在 ``test_integration_resnet.py``，
  默认跳过，设置 ``RUN_RESNET_TESTS=1`` 才下载权重运行。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers import BLUE, DIM, GREEN, RED, color_encoder, one_hot, solid

from core.library import FeatureLibrary


@pytest.fixture
def empty_library(tmp_path: Path) -> FeatureLibrary:
    """绑定临时 JSON 路径的空特征库，维度缩到 8 便于手算。"""
    return FeatureLibrary(path=tmp_path / "features.json", dim=DIM)


@pytest.fixture
def three_object_library(empty_library: FeatureLibrary) -> FeatureLibrary:
    """苹果/RED、香蕉/GREEN、杯子/BLUE 各一条特征。"""
    encoder = color_encoder({RED: one_hot(0), GREEN: one_hot(1), BLUE: one_hot(2)})
    empty_library.add("苹果", [solid(RED)], encoder=encoder)
    empty_library.add("香蕉", [solid(GREEN)], encoder=encoder)
    empty_library.add("杯子", [solid(BLUE)], encoder=encoder)
    return empty_library
