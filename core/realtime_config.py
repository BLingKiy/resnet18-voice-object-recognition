"""实时链路的三个阈值配置（与静态识别阈值分离）。

为什么必须分离：

- **T_static** 是用整张测试图校准的（本项目 0.8124）。静态图里整幅画面都是
  物体，分数天然高；实时帧里物体只占画面一部分，整帧特征被背景稀释，分数
  整体偏低。直接把 0.8124 用到实时链路，现场会几乎认不出物体。
- **T_rt** 必须在真实摄像头帧上重新校准，数据来源与静态不同，所以键名分开。
- **T_bg** 是「像不像背景」的阈值，量纲同样是余弦相似度，但语义与物体检索
  相反（越高越像背景），必须独立。

配置文件 ``data/realtime.json`` 与 ``library.json`` 并列：它只存"这台机器
当前的运行参数"，不碰特征库。``threshold_static`` 的**真值仍以特征库里的
``threshold`` 为准**；读配置时会用库值覆盖文件值，避免两处数字打架。

默认值说明（明确标注，不冒充校准结论）：

- ``threshold_static``：取特征库已校准值；库里没有时回落到
  ``core.recognize.DEFAULT_THRESHOLD``（0.60）。
- ``threshold_realtime``：**尚未真机校准**。默认暂等于 ``threshold_static``，
  这是"唯一已有校准依据的数字"，不是实时链路的结论；报告与界面都必须标注
  「待真机校准」。
- ``threshold_bg``：**尚未真机校准**。0.90 是保守初值，需用
  ``tools/realtime_check.py`` 的空场景/物体帧分布确定。
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .recognize import DEFAULT_THRESHOLD, resolve_threshold

SCHEMA_NAME = "rn18-realtime-config"
SCHEMA_VERSION = 1

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REALTIME_CONFIG_PATH = PROJECT_ROOT / "data" / "realtime.json"

# 未校准初值：仅用于让程序开箱可跑，不代表本项目结论。
DEFAULT_BG_THRESHOLD = 0.90
DEFAULT_CROP_MARGIN = 0.05


@dataclass(frozen=True, slots=True)
class RealtimeConfig:
    """实时链路的三个阈值与两个运行开关。

    Attributes:
        threshold_static: 静态识别阈值（真值来自特征库，本文件只做镜像）。
        threshold_realtime: 实时识别阈值 T_rt；**待真机校准**。
        threshold_bg: 背景门阈值 T_bg；**待真机校准**。
        gate_enabled: 是否启用背景门 + 中心裁剪；``False`` 时行为退回 rc1。
        crop_margin: 裁剪结果要超过全帧结果多少才采用，默认 0.05。
    """

    threshold_static: float
    threshold_realtime: float
    threshold_bg: float
    gate_enabled: bool = True
    crop_margin: float = DEFAULT_CROP_MARGIN

    def __post_init__(self) -> None:
        for name in ("threshold_static", "threshold_realtime", "threshold_bg"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} 必须是 [-1, 1] 内的数字，收到 {value!r}")
            object.__setattr__(self, name, value)

        margin = float(self.crop_margin)
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError(f"crop_margin 必须是非负数字，收到 {margin!r}")
        object.__setattr__(self, "crop_margin", margin)
        object.__setattr__(self, "gate_enabled", bool(self.gate_enabled))

    @classmethod
    def defaults(
        cls,
        library: Any | None = None,
        *,
        gate_enabled: bool = True,
    ) -> RealtimeConfig:
        """返回一份可运行的默认配置。

        Args:
            library: 特征库；用于把 ``threshold_static`` 对齐到已校准值。
            gate_enabled: 背景门默认开关。产品默认开；单元测试兼容旧行为时
                可显式传 ``False``。

        Returns:
            ``threshold_realtime`` 暂等于 ``threshold_static``、
            ``threshold_bg=0.90`` 的默认配置（两者均待真机校准）。
        """
        static = resolve_threshold(library) if library is not None else DEFAULT_THRESHOLD
        return cls(
            threshold_static=static,
            threshold_realtime=static,
            threshold_bg=DEFAULT_BG_THRESHOLD,
            gate_enabled=gate_enabled,
            crop_margin=DEFAULT_CROP_MARGIN,
        )

    def to_dict(self) -> dict[str, Any]:
        """序列化为 ``realtime.json`` 契约。"""
        return {
            "schema": SCHEMA_NAME,
            "version": SCHEMA_VERSION,
            "threshold_static": round(float(self.threshold_static), 8),
            "threshold_realtime": round(float(self.threshold_realtime), 8),
            "threshold_bg": round(float(self.threshold_bg), 8),
            "gate_enabled": bool(self.gate_enabled),
            "crop_margin": round(float(self.crop_margin), 8),
        }


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise OSError(f"读取实时配置失败: {path}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"实时配置 JSON 解析失败（文件可能损坏）: {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ValueError("实时配置 JSON 顶层必须是对象")  # noqa: TRY004 - 与 library.py 的 schema 校验口径一致
    return data


def load_realtime_config(
    path: str | Path | None = None,
    *,
    library: Any | None = None,
) -> RealtimeConfig:
    """读取实时配置；文件不存在时返回默认值（不自动写盘）。

    Args:
        path: 配置文件路径；默认 ``data/realtime.json``。
        library: 特征库；提供时用库里的 ``threshold`` 覆盖
            ``threshold_static``，保证静态阈值只有一个真值来源。

    Returns:
        :class:`RealtimeConfig`。

    Raises:
        ValueError: schema/版本/字段不合法。
        OSError: 文件存在但读取失败。
    """
    target = (Path(path) if path is not None else DEFAULT_REALTIME_CONFIG_PATH).expanduser()
    if not target.exists():
        return RealtimeConfig.defaults(library, gate_enabled=True)

    data = _load_json(target)
    schema = data.get("schema")
    if schema != SCHEMA_NAME:
        raise ValueError(f"不是本程序的实时配置文件（schema={schema!r}）")
    version = data.get("version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"不支持的实时配置版本 {version!r}，本程序只支持版本 {SCHEMA_VERSION}"
        )

    defaults = RealtimeConfig.defaults(library, gate_enabled=True)
    static = resolve_threshold(library) if library is not None else _number(
        data.get("threshold_static"), defaults.threshold_static, "threshold_static"
    )
    # 注意：即使文件里写了别的 threshold_static，只要传了 library，就以库为准。
    return RealtimeConfig(
        threshold_static=static,
        threshold_realtime=_number(
            data.get("threshold_realtime"), defaults.threshold_realtime, "threshold_realtime"
        ),
        threshold_bg=_number(
            data.get("threshold_bg"), defaults.threshold_bg, "threshold_bg"
        ),
        gate_enabled=bool(data.get("gate_enabled", True)),
        crop_margin=_number(
            data.get("crop_margin"), defaults.crop_margin, "crop_margin"
        ),
    )


def save_realtime_config(
    config: RealtimeConfig,
    path: str | Path | None = None,
) -> Path:
    """原子写入实时配置。

    Args:
        config: 待写入配置。
        path: 目标路径；默认 ``data/realtime.json``。

    Returns:
        实际写入的绝对路径。

    Raises:
        OSError: 磁盘写入失败。
    """
    target = (Path(path) if path is not None else DEFAULT_REALTIME_CONFIG_PATH).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(config.to_dict(), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return target.resolve()


def _number(value: Any, default: float, name: str) -> float:
    """把配置里的数字字段转成 float，缺失时用默认值。"""
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"实时配置字段 {name} 必须是数字，收到 {value!r}") from exc


__all__ = [
    "DEFAULT_BG_THRESHOLD",
    "DEFAULT_CROP_MARGIN",
    "DEFAULT_REALTIME_CONFIG_PATH",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "RealtimeConfig",
    "load_realtime_config",
    "save_realtime_config",
]
