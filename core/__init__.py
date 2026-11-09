"""ResNet18 物体识别核心层。

对外契约（界面层/Tab1、Tab2 只需依赖这些名字）::

    from core import (
        FeatureLibrary, add_to_library, extract_feature, extract_features,
        judge_unknown, load_library, recognize, retrieve, save_library,
    )

    library = load_library("data/features.json")      # 不存在则返回空库
    add_to_library(library, "苹果", ["a.jpg", "b.jpg"])
    save_library(library)
    result = recognize(library, "test.jpg")
    result.name, result.similarity, result.is_known
"""

from .background import (
    DEFAULT_BACKGROUND_PATH,
    BackgroundModel,
    load_background,
    save_background,
)
from .features import (
    FEATURE_DIM,
    MODEL_NAME,
    Encoder,
    build_resnet18_encoder,
    extract_feature,
    extract_features,
    get_default_encoder,
    image_digest,
    l2_normalize,
    to_pil,
    torch_available,
)
from .library import (
    AddResult,
    Entry,
    FeatureLibrary,
    ObjectRecord,
    add_to_library,
    load_library,
    save_library,
)
from .pipeline import (
    BACKGROUND_LABEL,
    FrameScore,
    MatchSource,
    StreamResult,
    StreamState,
    annotate_background,
    annotate_image,
    center_crop,
    recognize_static,
    recognize_stream,
    score_frame,
    select_recognition,
    speech_text,
)
from .realtime_config import (
    DEFAULT_BG_CROP_MARGIN,
    DEFAULT_REALTIME_CONFIG_PATH,
    DEFAULT_THRESHOLD_BG,
    DEFAULT_THRESHOLD_REALTIME,
    DEFAULT_THRESHOLD_STATIC,
    RealtimeConfig,
    load_realtime_config,
    save_realtime_config,
)
from .recognize import (
    DEFAULT_THRESHOLD,
    UNKNOWN_LABEL,
    Match,
    ObjectScore,
    Recognition,
    judge_unknown,
    object_scores,
    recognize,
    resolve_threshold,
    retrieve,
)

__all__ = [
    "BACKGROUND_LABEL",
    "DEFAULT_BACKGROUND_PATH",
    "DEFAULT_BG_CROP_MARGIN",
    "DEFAULT_REALTIME_CONFIG_PATH",
    "DEFAULT_THRESHOLD",
    "DEFAULT_THRESHOLD_BG",
    "DEFAULT_THRESHOLD_REALTIME",
    "DEFAULT_THRESHOLD_STATIC",
    "FEATURE_DIM",
    "MODEL_NAME",
    "UNKNOWN_LABEL",
    "AddResult",
    "BackgroundModel",
    "Encoder",
    "Entry",
    "FeatureLibrary",
    "FrameScore",
    "Match",
    "MatchSource",
    "ObjectRecord",
    "ObjectScore",
    "RealtimeConfig",
    "Recognition",
    "StreamResult",
    "StreamState",
    "add_to_library",
    "annotate_background",
    "annotate_image",
    "build_resnet18_encoder",
    "center_crop",
    "extract_feature",
    "extract_features",
    "get_default_encoder",
    "image_digest",
    "judge_unknown",
    "l2_normalize",
    "load_background",
    "load_library",
    "load_realtime_config",
    "object_scores",
    "recognize",
    "recognize_static",
    "recognize_stream",
    "resolve_threshold",
    "retrieve",
    "save_background",
    "save_library",
    "save_realtime_config",
    "score_frame",
    "select_recognition",
    "speech_text",
    "to_pil",
    "torch_available",
]
