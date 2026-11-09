"""实时链路的三个阈值与背景门开关（与静态识别阈值分离）。

为什么必须分成三个键：

- **T_static** 用整张测试图校准（本项目 0.8124）。静态图里整幅画面都是
  物体，分数天然高；实时帧里物体只占一部分，整帧特征被背景稀释，分数
  整体偏低。直接把 0.8124 用到实时链路，现场会几乎认不出物体。
- **T_rt** 必须在真实摄像头帧上重新校准，数据来源不同，所以键名分开。
- **T_bg** 是「像不像背景」的阈值，量纲同样是余弦相似度，但语义与物体
  检索相反（越高越像背景），必须独立。

配置文件 ``data/realtime.json`` 与 ``library.json`` 并列：它只存这台机器
当前的运行参数，不碰特征库。``threshold_static`` 的**真值仍以特征库里的
``threshold`` 为准**；读配置时会用库值覆盖文件值，避免两处数字打架。

默认值说明（明确标注，不冒充校准结论）：

- ``threshold_static``：取特征库已校准值；库里没有时回落到
  ``core.recognize.DEFAULT_THRESHOLD``（0.60）。
- ``threshold_realtime``：**尚未真机校准**。初值 0.75 是低于静态阈值的
  保守占位，只用于让程序开箱可跑，必须用 ``tools/realtime_check.py``
  在真实摄像头帧上重新确定。**不能直接把 0.8124 搬过来。**
- ``threshold_bg``：**尚未真机校准**。0.90 是保守初值（要求与背景非常像
  才门掉），需用空场景/物体帧分布确定。
- ``bg_crop_margin``：**尚未真机校准**。0.05 是「裁剪结果至少要比全帧高
  5 个百分点才采用」的占位值。
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

from .recognize import resolve_threshold

SCHEMA_NAME = "rn18-realtime-config"
SCHEMA_VERSION = 1

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REALTIME_CONFIG_PATH = PROJECT_ROOT / "data" / "realtime.json"

# 以下三个都是「待真机校准」的占位值，不是本项目结论。
DEFAULT_THRESHOLD_STATIC = 0.8124
DEFAULT_THRESHOLD_REALTIME = 0.75
DEFAULT_THRESHOLD_BG = 0.90
DEFAULT_BG_CROP_MARGIN = 0.05


def _number(value: Any, default: float, name: str) -> float:
    """把配置里的数字字段转成 float；缺失时用默认值。"""
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"实时配置字段 {name} 必须是数字，收到 {value!r}") from exc


@dataclass(frozen=True, slots=True)
class RealtimeConfig:
    """实时链路的三个阈值与两个运行参数。

    Attributes:
        threshold_static: 静态识别阈值（真值来自特征库，本文件只做镜像）。
        threshold_realtime: 实时识别阈值 ``T_rt``；**待真机校准**。
        threshold_bg: 背景门阈值 ``T_bg``；**待真机校准**。
        bg_crop_margin: 裁剪分数要超过全帧分数多少才采用；**待校准**。
        bg_gate_enabled: 是否启用背景门 + 中心裁剪；``False`` 时实时链路
            退回 P5.1 之前的单路行为。
    """

    threshold_static: float
    threshold_realtime: float = DEFAULT_THRESHOLD_REALTIME
    threshold_bg: float = DEFAULT_THRESHOLD_BG
    bg_crop_margin: float = DEFAULT_BG_CROP_MARGIN
    bg_gate_enabled: bool = True

    def __post_init__(self) -> None:
        for name in ("threshold_static", "threshold_realtime", "threshold_bg"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} 必须是 [-1, 1] 内的数字，收到 {value!r}")
            object.__setattr__(self, name, value)

        margin = float(self.bg_crop_margin)
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError(f"bg_crop_margin 必须是非负数字，收到 {margin!r}")
        object.__setattr__(self, "bg_crop_margin", margin)
        object.__setattr__(self, "bg_gate_enabled", bool(self.bg_gate_enabled))

    @classmethod
    def defaults(
        cls,
        library: Any | None = None,
        *,
        bg_gate_enabled: bool = True,
    ) -> RealtimeConfig:
        """返回一份可运行的默认配置。

        Args:
            library: 特征库；用于把 ``threshold_static`` 对齐到已校准值。
            bg_gate_enabled: 背景门默认开关。产品默认开；旧行为对照时传
                ``False``。

        Returns:
            ``threshold_realtime=0.75``、``threshold_bg=0.90`` 的默认配置
            （两者均待真机校准）。
        """
        if library is not None:
            static = resolve_threshold(library)
        else:
            static = DEFAULT_THRESHOLD_STATIC
        return cls(
            threshold_static=static,
            threshold_realtime=DEFAULT_THRESHOLD_REALTIME,
            threshold_bg=DEFAULT_THRESHOLD_BG,
            bg_crop_margin=DEFAULT_BG_CROP_MARGIN,
            bg_gate_enabled=bg_gate_enabled,
        )

    def to_dict(self) -> dict[str, Any]:
        """序列化为 ``data/realtime.json`` 的字典结构。"""
        return {
            "schema": SCHEMA_NAME,
            "version": SCHEMA_VERSION,
            "threshold_static": round(float(self.threshold_static), 8),
            "threshold_realtime": round(float(self.threshold_realtime), 8),
            "threshold_bg": round(float(self.threshold_bg), 8),
            "bg_crop_margin": round(float(self.bg_crop_margin), 8),
            "bg_gate_enabled": bool(self.bg_gate_enabled),
        }


def _load_json(path: Path) -> Mapping[str, Any]:
    """读取 JSON 并确保顶层是对象。"""
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
        return RealtimeConfig.defaults(library, bg_gate_enabled=True)

    data = _load_json(target)
    schema = data.get("schema")
    if schema is not None and schema != SCHEMA_NAME:
        raise ValueError(f"不是本程序的实时配置文件（schema={schema!r}）")
    version = data.get("version")
    if version is not None and version != SCHEMA_VERSION:
        raise ValueError(
            f"不支持的实时配置版本 {version!r}，本程序只支持版本 {SCHEMA_VERSION}"
        )

    defaults = RealtimeConfig.defaults(library, bg_gate_enabled=True)
    static = (
        resolve_threshold(library)
        if library is not None
        else _number(data.get("threshold_static"), defaults.threshold_static, "threshold_static")
    )
    # 即使文件里写了别的 threshold_static，只要传了 library，就以库为准。
    return RealtimeConfig(
        threshold_static=static,
        threshold_realtime=_number(
            data.get("threshold_realtime"),
            defaults.threshold_realtime,
            "threshold_realtime",
        ),
        threshold_bg=_number(
            data.get("threshold_bg"), defaults.threshold_bg, "threshold_bg"
        ),
        bg_crop_margin=_number(
            data.get("bg_crop_margin"), defaults.bg_crop_margin, "bg_crop_margin"
        ),
        bg_gate_enabled=bool(data.get("bg_gate_enabled", True)),
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


__all__ = [
    "DEFAULT_BG_CROP_MARGIN",
    "DEFAULT_REALTIME_CONFIG_PATH",
    "DEFAULT_THRESHOLD_BG",
    "DEFAULT_THRESHOLD_REALTIME",
    "DEFAULT_THRESHOLD_STATIC",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "RealtimeConfig",
    "load_realtime_config",
    "save_realtime_config",
]
