"""``core/pipeline.py`` 测试：静态/实时两条链路共用同一套识别与标注。

全部使用 ``tests/helpers.py`` 的假编码器，不加载 ResNet18。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from helpers import BLUE, DIM, GREEN, RED, YELLOW, color_encoder, one_hot, solid
from PIL import Image, ImageDraw

from core.background import BackgroundModel
from core.library import FeatureLibrary
from core.pipeline import (
    BACKGROUND_LABEL,
    annotate_image,
    center_crop,
    recognize_static,
    recognize_stream,
    score_frame,
    select_recognition,
    speech_text,
)
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


# --------------------------------------------------------------------------- #
# 背景自适应（P5.1）
# --------------------------------------------------------------------------- #
GRAY = (128, 128, 128)


def test_background_gate_skips_retrieval_and_speech(library: FeatureLibrary) -> None:
    background = BackgroundModel(vector=one_hot(0), sample_count=18)  # RED

    result = recognize_stream(
        solid(RED),
        library,
        encoder=ENCODER,
        bg_model=background,
        t_bg=0.9,
    )

    assert result.is_background is True
    assert result.name == BACKGROUND_LABEL
    assert result.recognition.is_known is False
    assert result.speech == ""
    assert result.bg_similarity == pytest.approx(1.0, abs=1e-6)
    assert result.crop_similarity is None  # 命中背景门后不再做裁剪/检索
    assert result.used_crop is False
    assert result.annotated.shape == (64, 64, 3)


def test_background_gate_miss_still_recognizes(library: FeatureLibrary) -> None:
    background = BackgroundModel(vector=one_hot(0), sample_count=18)  # RED

    result = recognize_stream(
        solid(BLUE),
        library,
        encoder=ENCODER,
        bg_model=background,
        t_bg=0.9,
    )

    assert result.is_background is False
    assert result.name == "杯子"
    assert result.speech == "识别结果：杯子"
    assert result.bg_similarity == pytest.approx(0.0, abs=1e-6)


def test_gate_disabled_is_byte_identical_to_no_gate(library: FeatureLibrary) -> None:
    frame = solid(BLUE, (80, 60))
    background = BackgroundModel(vector=one_hot(0), sample_count=18)

    baseline = recognize_stream(frame, library, encoder=ENCODER)
    disabled = recognize_stream(
        frame,
        library,
        encoder=ENCODER,
        bg_model=background,
        t_bg=0.9,
        bg_gate_enabled=False,
    )

    assert disabled.annotated.tobytes() == baseline.annotated.tobytes()
    assert disabled.recognition == baseline.recognition
    assert disabled.speech == baseline.speech
    assert disabled.is_background is False
    assert disabled.bg_similarity is None
    assert disabled.threshold == baseline.threshold


def test_center_crop_uses_middle_sixty_percent() -> None:
    cropped = center_crop(solid(RED, (100, 80)))

    assert cropped.size == (60, 48)
    with pytest.raises(ValueError):
        center_crop(solid(RED), fraction=0.0)


def test_crop_used_only_when_better_by_margin(
    library: FeatureLibrary,
) -> None:
    """四周 GRAY、中心 RED：全帧不认识，中心裁剪能认出苹果。"""
    encoder = color_encoder(
        {RED: one_hot(0), GRAY: one_hot(3), BLUE: one_hot(1)}
    )
    frame = Image.new("RGB", (100, 100), GRAY)
    ImageDraw.Draw(frame).rectangle([20, 20, 79, 79], fill=RED)
    background = BackgroundModel(vector=one_hot(2), sample_count=18)  # BLUE

    adopted = recognize_stream(
        frame,
        library,
        encoder=encoder,
        bg_model=background,
        t_bg=0.9,
        bg_crop_margin=0.05,
    )
    assert adopted.used_crop is True
    assert adopted.name == "苹果"
    assert adopted.crop_similarity == pytest.approx(1.0, abs=1e-6)

    # margin 调到 1.1：裁剪必须比全帧高 1.1，任何余弦都不可能满足 -> 不采用。
    rejected = recognize_stream(
        frame,
        library,
        encoder=encoder,
        bg_model=background,
        t_bg=0.9,
        bg_crop_margin=1.1,
    )
    assert rejected.used_crop is False
    assert rejected.name == "未知物体"
    assert rejected.crop_similarity == pytest.approx(1.0, abs=1e-6)


def test_crop_adopted_still_judged_unknown_by_realtime_threshold() -> None:
    """裁剪分 0.5 高于全帧、也高于库内 0.4，但仍低于 T_rt 0.75 -> 未知。

    这条用例同时证明实时链路用的是 ``t_rt``，不是库里的静态阈值。
    """
    half = np.array([0.5, np.sqrt(0.75), 0, 0, 0, 0, 0, 0], dtype=np.float32)
    orange = (255, 140, 0)
    encoder = color_encoder(
        {RED: one_hot(0), GRAY: one_hot(3), orange: half, BLUE: one_hot(1)}
    )
    # 只放「苹果」，避免 DIM=1 上的「杯子」把 0.5 的裁剪分抬到 0.866。
    library = FeatureLibrary(dim=DIM)
    library.add("苹果", [solid(RED)], encoder=encoder)
    library.set_threshold(0.4)
    frame = Image.new("RGB", (100, 100), GRAY)
    ImageDraw.Draw(frame).rectangle([20, 20, 79, 79], fill=orange)
    background = BackgroundModel(vector=one_hot(2), sample_count=18)

    result = recognize_stream(
        frame,
        library,
        encoder=encoder,
        bg_model=background,
        t_bg=0.9,
        t_rt=0.75,
        bg_crop_margin=0.05,
    )

    assert result.crop_similarity == pytest.approx(0.5, abs=1e-6)
    assert result.source == "crop"
    assert result.name == "未知物体"
    assert result.threshold_used == pytest.approx(0.75)


def test_score_frame_returns_raw_scores_for_calibration(library: FeatureLibrary) -> None:
    """校准工具需要"未过门"的原始分，不能被背景门改写成 0。"""
    background = BackgroundModel(vector=one_hot(0), sample_count=18)

    score = score_frame(
        solid(RED),
        library,
        encoder=ENCODER,
        bg_model=background,
    )

    assert score.full.similarity == pytest.approx(1.0, abs=1e-6)
    assert score.crop is not None and score.crop.similarity == pytest.approx(1.0)
    assert score.bg_sim == pytest.approx(1.0, abs=1e-6)
    recognition, source = select_recognition(score, 0.05)
    assert recognition.name == "苹果"
    # crop 与 full 同分，不满足"高于 margin"，所以不采用裁剪。
    assert source == "full"


def test_stream_reports_explicit_realtime_threshold(library: FeatureLibrary) -> None:
    result = recognize_stream(
        solid(RED), library, encoder=ENCODER, threshold=0.9
    )

    assert result.threshold == pytest.approx(0.9)
    assert result.recognition.threshold == pytest.approx(0.9)
    assert result.is_known is True
