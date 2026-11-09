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
    "DEFAULT_THRESHOLD",
    "FEATURE_DIM",
    "MODEL_NAME",
    "UNKNOWN_LABEL",
    "AddResult",
    "Encoder",
    "Entry",
    "FeatureLibrary",
    "Match",
    "ObjectRecord",
    "ObjectScore",
    "Recognition",
    "add_to_library",
    "build_resnet18_encoder",
    "extract_feature",
    "extract_features",
    "get_default_encoder",
    "image_digest",
    "judge_unknown",
    "l2_normalize",
    "load_library",
    "object_scores",
    "recognize",
    "resolve_threshold",
    "retrieve",
    "save_library",
    "to_pil",
    "torch_available",
]
