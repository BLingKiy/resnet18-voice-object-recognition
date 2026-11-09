"""背景门 + 中心裁剪第二路的契约测试（合成帧 / 假特征，不依赖摄像头）。

覆盖任务要求的三个硬点：

1. ``bg_sim > t_bg`` 时输出 ``state="no_object"``，检索与 TTS 调用次数均为 0；
2. 裁剪分 - 全帧分 ``> margin`` 才用裁剪，否则用全帧，且分数来自被采用那一路；
3. ``bg_gate_enabled=False`` 时与改动前的单路识别逐帧等价。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from helpers import DIM, color_encoder, one_hot, solid
from PIL import Image, ImageDraw

from core import pipeline
from core.background import BackgroundModel
from core.library import FeatureLibrary
from core.pipeline import BACKGROUND_LABEL, recognize_stream

RED = (220, 40, 40)
BLUE = (40, 60, 220)
GRAY = (128, 128, 128)

ENCODER = color_encoder({RED: one_hot(0), BLUE: one_hot(1), GRAY: one_hot(3)})


@pytest.fixture
def library(tmp_path: Path) -> FeatureLibrary:
    lib = FeatureLibrary(path=tmp_path / "features.json", dim=DIM)
    lib.add("苹果", [solid(RED)], encoder=ENCODER)
    lib.add("杯子", [solid(BLUE)], encoder=ENCODER)
    lib.set_threshold(0.6)  # 静态阈值；实时链路不应套用它
    return lib


def _center_object_image(background: tuple[int, int, int], obj: tuple[int, int, int]) -> Image.Image:
    """100×100 画布：四周背景色，中心 60% 是物体色。"""
    frame = Image.new("RGB", (100, 100), background)
    ImageDraw.Draw(frame).rectangle([20, 20, 80, 80], fill=obj)
    return frame


def test_gate_hit_skips_retrieval_and_tts(
    library: FeatureLibrary, monkeypatch: pytest.MonkeyPatch
) -> None:
    background = BackgroundModel(vector=one_hot(0), sample_count=18)  # RED
    calls = {"recognize": 0, "speech": 0}

    def forbidden_recognize(*args: object, **kwargs: object) -> None:
        calls["recognize"] += 1
        raise AssertionError("背景门命中时不应进入物体检索")

    def forbidden_speech(*args: object, **kwargs: object) -> str:
        calls["speech"] += 1
        raise AssertionError("背景门命中时不应生成播报文本")

    monkeypatch.setattr(pipeline, "recognize", forbidden_recognize)
    monkeypatch.setattr(pipeline, "speech_text", forbidden_speech)

    result = recognize_stream(
        solid(RED),
        library,
        encoder=ENCODER,
        bg_model=background,
        t_bg=0.5,
        t_rt=0.5,
    )

    assert calls == {"recognize": 0, "speech": 0}
    assert result.state == "no_object"
    assert result.is_no_object is True
    assert result.name == BACKGROUND_LABEL
    assert result.recognition.is_known is False
    assert result.speech == ""
    assert result.score is None
    assert result.source == "full"
    assert result.bg_sim == pytest.approx(1.0, abs=1e-6)
    assert result.threshold_used == pytest.approx(0.5)  # 触发背景门的是 T_bg
    assert result.annotated.shape == (64, 64, 3)


def test_gate_miss_recognizes_with_full_frame(library: FeatureLibrary) -> None:
    background = BackgroundModel(vector=one_hot(0), sample_count=18)  # RED

    result = recognize_stream(
        solid(BLUE),
        library,
        encoder=ENCODER,
        bg_model=background,
        t_bg=0.9,
        t_rt=0.6,
    )

    assert result.state == "known"
    assert result.name == "杯子"
    assert result.speech == "识别结果：杯子"
    assert result.source == "full"
    assert result.bg_sim == pytest.approx(0.0, abs=1e-6)
    assert result.score == pytest.approx(1.0, abs=1e-6)


def test_crop_used_only_when_score_improves_by_margin(library: FeatureLibrary) -> None:
    """四周 GRAY、中心 RED：全帧不认识，中心裁剪能认出苹果。"""
    background = BackgroundModel(vector=one_hot(2), sample_count=18)  # 与画面正交
    frame = _center_object_image(GRAY, RED)

    adopted = recognize_stream(
        frame,
        library,
        encoder=ENCODER,
        bg_model=background,
        t_bg=1.0,  # 背景门不拦截，专门验证裁剪采用逻辑
        t_rt=0.5,
        bg_crop_margin=0.05,
    )
    assert adopted.source == "crop"
    assert adopted.name == "苹果"
    assert adopted.score == pytest.approx(1.0, abs=1e-6)
    assert adopted.crop_sim == pytest.approx(1.0, abs=1e-6)

    # margin 1.1：余弦差最大 2.0，但本例全帧 0、裁剪 1.0，差 1.0 不够 -> 全帧。
    rejected = recognize_stream(
        frame,
        library,
        encoder=ENCODER,
        bg_model=background,
        t_bg=1.0,
        t_rt=0.5,
        bg_crop_margin=1.1,
    )
    assert rejected.source == "full"
    assert rejected.name == "未知物体"
    assert rejected.score == pytest.approx(0.0, abs=1e-6)
    assert rejected.crop_sim == pytest.approx(1.0, abs=1e-6)


def test_equal_scores_keep_full_frame(library: FeatureLibrary) -> None:
    background = BackgroundModel(vector=one_hot(2), sample_count=18)
    frame = solid(RED)  # 全帧与中心裁剪都是 RED，差值 0，不大于 margin

    result = recognize_stream(
        frame,
        library,
        encoder=ENCODER,
        bg_model=background,
        t_bg=1.0,
        t_rt=0.5,
        bg_crop_margin=0.05,
    )

    assert result.source == "full"
    assert result.name == "苹果"
    assert result.score == pytest.approx(1.0, abs=1e-6)


def test_gate_disabled_is_equivalent_to_previous_path(library: FeatureLibrary) -> None:
    frame = solid(BLUE, (80, 60))
    background = BackgroundModel(vector=one_hot(0), sample_count=18)

    baseline = recognize_stream(frame, library, encoder=ENCODER)
    disabled = recognize_stream(
        frame,
        library,
        encoder=ENCODER,
        bg_model=background,
        t_bg=0.5,
        t_rt=0.99,  # 关闭时必须忽略新阈值，完整退回旧路径
        bg_gate_enabled=False,
        bg_crop_margin=0.5,
    )

    assert disabled.recognition == baseline.recognition
    assert disabled.annotated.tobytes() == baseline.annotated.tobytes()
    assert disabled.speech == baseline.speech
    assert disabled.state == baseline.state
    assert disabled.source == "full"
    assert disabled.bg_sim is None
    assert disabled.score == pytest.approx(baseline.score)
    assert disabled.threshold_used == pytest.approx(baseline.threshold_used)


def test_realtime_threshold_is_used_not_static() -> None:
    """候选分 0.9：静态阈值 0.6 会认成已知；T_rt 0.99 必须判未知。"""
    from core.library import FeatureLibrary as _Library

    near = np.zeros(DIM, dtype=np.float32)
    near[0] = 0.9
    near[1] = float(np.sqrt(1.0 - 0.9**2))
    encoder = color_encoder({RED: one_hot(0), BLUE: near})
    library = _Library(dim=DIM)
    library.add("苹果", [solid(RED)], encoder=encoder)
    library.set_threshold(0.6)
    background = BackgroundModel(vector=one_hot(2), sample_count=18)  # 正交

    result = recognize_stream(
        solid(BLUE),
        library,
        encoder=encoder,
        bg_model=background,
        t_bg=1.0,
        t_rt=0.99,
    )

    assert result.state == "unknown"
    assert result.name == "未知物体"
    assert result.threshold_used == pytest.approx(0.99)
    assert result.score == pytest.approx(0.9, abs=1e-6)


def test_empty_library_gate_miss_returns_unknown(tmp_path: Path) -> None:
    library = FeatureLibrary(path=tmp_path / "features.json", dim=DIM)
    background = BackgroundModel(vector=one_hot(2), sample_count=18)

    result = recognize_stream(
        solid(RED),
        library,
        encoder=ENCODER,
        bg_model=background,
        t_bg=1.0,
        t_rt=0.5,
    )

    assert result.state == "unknown"
    assert result.name == "未知物体"
    assert result.score == pytest.approx(0.0)
    assert np.isfinite(result.elapsed_ms)
