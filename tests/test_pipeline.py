"""``core/pipeline.py`` 测试：静态/实时两条链路共用同一套识别与标注。

全部使用 ``tests/helpers.py`` 的假编码器，不加载 ResNet18。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from helpers import BLUE, DIM, GREEN, RED, YELLOW, color_encoder, one_hot, solid
from PIL import Image

from core.library import FeatureLibrary
from core.pipeline import annotate_image, recognize_static, recognize_stream, speech_text
from core.recognize import Recognition

ENCODER = color_encoder(
    {
        RED: one_hot(0),
        GREEN: one_hot(0),
        BLUE: one_hot(1),
        YELLOW: one_hot(2),
    }
)


@pytest.fixture
def library(tmp_path: Path) -> FeatureLibrary:
    lib = FeatureLibrary(path=tmp_path / "features.json", dim=DIM)
    lib.add("苹果", [solid(RED)], encoder=ENCODER)
    lib.add("杯子", [solid(BLUE)], encoder=ENCODER)
    return lib


def test_recognize_stream_returns_overlay_and_timing(library: FeatureLibrary) -> None:
    result = recognize_stream(solid(RED, (80, 60)), library, encoder=ENCODER)

    assert result.is_known and result.name == "苹果"
    assert result.speech == "识别结果：苹果"
    assert result.elapsed_ms >= 0.0
    assert isinstance(result.annotated, np.ndarray)
    assert result.annotated.shape == (60, 80, 3)
    assert result.annotated.dtype == np.uint8
    # 已知物体用绿色标注条：左上角像素绿色分量最高。
    top_left = result.annotated[0, 0]
    assert int(top_left[1]) > int(top_left[0])


def test_unknown_frame_is_labelled_and_red(library: FeatureLibrary) -> None:
    encoder = color_encoder({YELLOW: one_hot(7), RED: one_hot(0), BLUE: one_hot(1)})
    result = recognize_stream(solid(YELLOW), library, encoder=encoder)

    assert result.is_known is False
    assert result.name == "未知物体"
    assert result.speech == "未知物体"
    top_left = result.annotated[0, 0]
    assert int(top_left[0]) > int(top_left[1])


def test_recognize_static_matches_stream_conclusion(library: FeatureLibrary) -> None:
    image = solid(BLUE, (50, 50))
    static = recognize_static(image, library, encoder=ENCODER, top_k=1)
    stream = recognize_stream(image, library, encoder=ENCODER)

    assert static.name == stream.name == "杯子"
    assert static.similarity == pytest.approx(stream.recognition.similarity)
    assert len(static.matches) == 1


def test_empty_library_is_unknown_without_crash(tmp_path: Path) -> None:
    lib = FeatureLibrary(path=tmp_path / "features.json", dim=DIM)
    result = recognize_stream(solid(RED), lib, encoder=ENCODER)

    assert result.name == "未知物体" and result.is_known is False
    assert result.recognition.similarity == 0.0


def test_annotate_does_not_mutate_source_and_handles_long_names() -> None:
    source = solid(RED, (120, 90))
    original = source.copy()
    result = Recognition(
        is_known=True,
        name="一个非常非常长的物体名称用来测试省略号",
        similarity=0.93,
        threshold=0.6,
    )

    annotated = annotate_image(source, result)

    assert isinstance(annotated, Image.Image)
    assert annotated.size == source.size
    assert source.tobytes() == original.tobytes()


def test_blank_frame_is_rejected() -> None:
    with pytest.raises(ValueError):
        recognize_stream(None, FeatureLibrary(dim=DIM))  # type: ignore[arg-type]


def test_speech_text_covers_known_and_unknown() -> None:
    known = Recognition(is_known=True, name="苹果", similarity=0.9, threshold=0.6)
    unknown = Recognition(
        is_known=False, name="未知物体", similarity=0.1, threshold=0.6
    )

    assert speech_text(known) == "识别结果：苹果"
    assert speech_text(unknown) == "未知物体"
