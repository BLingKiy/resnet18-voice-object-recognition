"""实时识别自检：用空场景帧 + 物体帧给出 T_bg / T_rt 的建议值。

这个脚本只做"量分数、给建议"，**不替人拍板**：它会把两条分布、空场景
误报次数和三个阈值的来源全部打印出来，人只需要判断是否合理，或在真机上
补拍一轮。它不参与应用运行，``opencv-python`` 也只在 ``--camera`` 模式
才需要；``--frames`` 纯离线，CI 也能跑。

两种用法::

    # 离线：帧目录按物体名分子目录，空场景放 background/
    python tools/realtime_check.py --frames ~/.rn18/data/realtime_frames \
        --library ~/.rn18/data/features.json

    # 真机：前 1/3 时间空场景，之后按库里的物体顺序逐个放入画面
    python tools/realtime_check.py --camera 0 --seconds 30 \
        --library ~/.rn18/data/features.json

帧目录约定::

    frames/
      background/   # 空场景；没有这个目录时整个 frames/ 视为背景帧
      mouse/        # 目录名与特征库里的物体名一致
      keyboard/
      unknown/      # 可选：未录入物体，按物体名再分子目录
        剪刀/
        充电头/

``--write`` 只写 ``data/realtime.json``，不动 ``library.json``，也不改
``threshold_static``（始终取特征库里的已校准值）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 允许 ``python tools/realtime_check.py`` 直接运行（此时 sys.path[0] 是 tools/）。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.background import BackgroundModel
from core.features import Encoder, ImageInput, get_default_encoder
from core.library import DEFAULT_LIBRARY_PATH, load_library
from core.pipeline import score_frame, select_recognition
from core.realtime_config import (
    DEFAULT_REALTIME_CONFIG_PATH,
    RealtimeConfig,
    load_realtime_config,
    save_realtime_config,
)
from core.recognize import resolve_threshold

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BACKGROUND_ALIASES = {"background", "bg", "empty", "背景", "空场景"}
UNKNOWN_ALIASES = {"unknown", "unknowns", "未知", "未录入"}

DEFAULT_TAKE = 20
DEFAULT_DROP_FIRST = 2


@dataclass(slots=True)
class FrameSet:
    """一次离线自检用到的所有帧路径。"""

    root: Path
    background: list[Path] = field(default_factory=list)
    known: dict[str, list[Path]] = field(default_factory=dict)
    unknown: dict[str, list[Path]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _images(directory: Path) -> list[Path]:
    """取目录下排序后的图片文件。"""
    if not directory.is_dir():
        return []
    return sorted(
        item
        for item in directory.iterdir()
        if item.is_file()
        and item.suffix.lower() in IMAGE_SUFFIXES
        and not item.name.startswith(".")
    )


def _class_dirs(root: Path) -> list[Path]:
    """取根目录下按物体名分好的子目录（忽略隐藏目录）。"""
    if not root.is_dir():
        raise FileNotFoundError(f"帧目录不存在: {root}")
    return sorted(
        item for item in root.iterdir() if item.is_dir() and not item.name.startswith(".")
    )


def discover_frames(
    root: str | Path,
    *,
    background_dir: str | Path | None = None,
) -> FrameSet:
    """按约定扫描帧目录，返回背景/已知物体/未知物体三组帧。

    Args:
        root: 帧根目录。
        background_dir: 显式背景目录；省略时找 ``root/background`` 等别名，
            再找不到就把 ``root`` 下的散图当背景帧。

    Returns:
        :class:`FrameSet`。

    Raises:
        FileNotFoundError: 根目录不存在。
        ValueError: 目录里没有任何图片。
    """
    root_path = Path(root).expanduser()
    if not root_path.is_dir():
        raise FileNotFoundError(f"帧目录不存在: {root_path}")

    frame_set = FrameSet(root=root_path)
    if background_dir is not None:
        bg_dir = Path(background_dir).expanduser()
        frame_set.background = _images(bg_dir)
        if not frame_set.background:
            raise ValueError(f"背景目录里没有图片: {bg_dir}")

    for directory in _class_dirs(root_path):
        loose = _images(directory)
        name = directory.name
        lowered = name.lower()
        if lowered in BACKGROUND_ALIASES:
            frame_set.background.extend(loose)
        elif lowered in UNKNOWN_ALIASES:
            for unknown_dir in _class_dirs(directory):
                images = _images(unknown_dir)
                if images:
                    frame_set.unknown[unknown_dir.name] = images
        elif loose:
            frame_set.known[name] = loose

    if not frame_set.background:
        # 没有 background/ 子目录：把根目录下的散图当空场景帧（常见于
        # 直接把一段空场景截图丢进来的情况）。
        frame_set.background = _images(root_path)
        if frame_set.background:
            frame_set.warnings.append(
                "未找到 background/ 子目录，已把根目录散图当作空场景帧。"
            )

    if not frame_set.background and not frame_set.known and not frame_set.unknown:
        raise ValueError(f"目录里没有可用的图片: {root_path}")
    if not frame_set.background:
        frame_set.warnings.append("缺少空场景帧，T_bg 无法给出可靠建议。")
    if not frame_set.known:
        frame_set.warnings.append("缺少已知物体帧，T_rt 无法校准时会保留当前值。")

    return frame_set


def _stats(values: Sequence[float | None]) -> dict[str, Any]:
    """返回 min/mean/max/n，空列表时全部为 ``None``。"""
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return {"n": 0, "min": None, "mean": None, "max": None}
    return {
        "n": len(numbers),
        "min": round(min(numbers), 6),
        "mean": round(sum(numbers) / len(numbers), 6),
        "max": round(max(numbers), 6),
    }


def _score_items(
    items: Sequence[ImageInput],
    library: Any,
    config: RealtimeConfig,
    encoder: Encoder,
    background: BackgroundModel | None,
    crop_margin: float,
    *,
    label: str,
    name_for: Callable[[int], str],
) -> dict[str, Any]:
    """逐帧打分并汇总一组画面的分布（路径/数组都支持）。"""
    bg_values: list[float] = []
    full_values: list[float] = []
    crop_values: list[float] = []
    effective_values: list[float] = []
    used_crop = 0
    correct = 0
    rows: list[dict[str, Any]] = []

    for index, item in enumerate(items):
        score = score_frame(
            item,
            library,
            threshold=config.threshold_realtime,
            encoder=encoder,
            background=background,
            crop_margin=crop_margin,
        )
        effective, adopted = select_recognition(score, crop_margin)
        if score.background_similarity is not None:
            bg_values.append(float(score.background_similarity))
        full_values.append(float(score.full.similarity))
        if score.crop is not None:
            crop_values.append(float(score.crop.similarity))
        effective_values.append(float(effective.similarity))
        used_crop += int(adopted)
        is_correct = effective.is_known and effective.name == label
        correct += int(is_correct)
        rows.append(
            {
                "image": name_for(index),
                "bg": (
                    round(float(score.background_similarity), 6)
                    if score.background_similarity is not None
                    else None
                ),
                "full": round(float(score.full.similarity), 6),
                "crop": (
                    round(float(score.crop.similarity), 6)
                    if score.crop is not None
                    else None
                ),
                "effective": round(float(effective.similarity), 6),
                "name": effective.name,
                "is_known": bool(effective.is_known),
                "used_crop": bool(adopted),
                "correct": bool(is_correct),
            }
        )

    return {
        "label": label,
        "n": len(items),
        "bg_sim": _stats(bg_values),
        "full": _stats(full_values),
        "crop": _stats(crop_values),
        "effective": _stats(effective_values),
        "used_crop": used_crop,
        "correct": correct,
        "rows": rows,
    }


def _evaluate_group(
    paths: Sequence[Path],
    library: Any,
    config: RealtimeConfig,
    encoder: Encoder,
    background: BackgroundModel | None,
    crop_margin: float,
    *,
    label: str,
    limit: int | None = None,
) -> dict[str, Any]:
    """路径版分组打分：截取 ``limit`` 后交给 :func:`_score_items`。"""
    items = list(paths)
    if limit is not None:
        items = items[: int(limit)]
    return _score_items(
        items,
        library,
        config,
        encoder,
        background,
        crop_margin,
        label=label,
        name_for=lambda index: items[index].name,
    )


def evaluate_frames(
    frame_set: FrameSet,
    library: Any,
    config: RealtimeConfig,
    encoder: Encoder,
    *,
    crop_margin: float | None = None,
    limit: int | None = None,
    take: int = DEFAULT_TAKE,
    drop_first: int = DEFAULT_DROP_FIRST,
) -> dict[str, Any]:
    """对 :class:`FrameSet` 做一次完整打分，返回可 JSON 化的报告。

    Args:
        frame_set: 已扫描好的帧集合。
        library: 特征库（必须非空，否则检索分全为 0）。
        config: 当前实时配置（T_rt/T_bg 的现值）。
        encoder: 特征编码器；测试注入假编码器。
        crop_margin: 覆盖配置里的裁剪 margin；``None`` 用配置值。
        limit: 每个物体/背景组最多取多少帧（调试用）。
        take: 背景学习最多取多少帧。
        drop_first: 背景学习开头丢弃多少帧。

    Returns:
        报告字典；字段见 :func:`format_report`。
    """
    if library.total_vectors == 0:
        raise ValueError("特征库为空：请先录入物体再自检")

    margin = float(config.crop_margin if crop_margin is None else crop_margin)
    bg_paths = list(frame_set.background)
    if limit is not None:
        bg_paths = bg_paths[: int(limit)]

    warnings = list(frame_set.warnings)
    background: dict[str, Any] = {
        "frames": len(bg_paths),
        "learned": 0,
        "bg_sim": _stats([]),
        "bg_sim_holdout": _stats([]),
        "false_positive": {
            "threshold": config.threshold_realtime,
            "max": None,
            "count": 0,
        },
    }

    background_model: BackgroundModel | None = None
    if len(bg_paths) > drop_first:
        try:
            background_model = BackgroundModel.learn(
                bg_paths,
                encoder=encoder,
                take=take,
                drop_first=drop_first,
            )
            background["learned"] = background_model.frames_used
        except Exception as exc:  # noqa: BLE001 - 任何编码失败都写进报告，不中断整轮自检
            warnings.append(f"背景学习失败：{type(exc).__name__}: {exc}")

    if background_model is not None:
        bg_report = _evaluate_group(
            bg_paths,
            library,
            config,
            encoder,
            background_model,
            margin,
            label="background",
        )
        background["bg_sim"] = bg_report["bg_sim"]
        background["false_positive"] = {
            "threshold": config.threshold_realtime,
            "max": bg_report["effective"]["max"],
            "count": sum(1 for row in bg_report["rows"] if row["is_known"]),
        }

        # 留出帧：学习用掉的帧不参与背景相似度上界，避免"自己测自己"偏乐观。
        holdout = bg_paths[drop_first + background_model.frames_used :]
        if holdout:
            holdout_report = _evaluate_group(
                holdout,
                library,
                config,
                encoder,
                background_model,
                margin,
                label="background",
            )
            background["bg_sim_holdout"] = holdout_report["bg_sim"]
        else:
            warnings.append(
                "空场景帧全部用于学习，背景相似度上界略偏乐观；"
                "建议至少留 3 张未参与学习的空场景帧。"
            )

    known_reports = {
        name: _evaluate_group(
            paths,
            library,
            config,
            encoder,
            background_model,
            margin,
            label=name,
            limit=limit,
        )
        for name, paths in frame_set.known.items()
    }
    unknown_reports = {
        name: _evaluate_group(
            paths,
            library,
            config,
            encoder,
            background_model,
            margin,
            label=name,
            limit=limit,
        )
        for name, paths in frame_set.unknown.items()
    }

    thresholds = recommend_thresholds(
        background=background,
        known=known_reports,
        unknown=unknown_reports,
        config=config,
        library=library,
    )

    return {
        "library": str(getattr(library, "path", "") or ""),
        "frames_root": str(frame_set.root),
        "config": config.to_dict(),
        "background": background,
        "known": known_reports,
        "unknown": unknown_reports,
        "thresholds": thresholds,
        "warnings": warnings,
    }


def recommend_thresholds(
    *,
    background: dict[str, Any],
    known: dict[str, dict[str, Any]],
    unknown: dict[str, dict[str, Any]],
    config: RealtimeConfig,
    library: Any,
) -> dict[str, Any]:
    """根据分布给出 T_bg / T_rt 建议值，并判定是否可分。

    取法与 ``tools/calibrate.py`` 一致：两条分布能分开时取中点；分不开时
    给出"偏保守"的边界值并明确标注需要重拍，不硬编一个好看的数字。
    """
    reasons: dict[str, str] = {}
    static = resolve_threshold(library)

    bg_holdout_min = background.get("bg_sim_holdout", {}).get("min")
    bg_all_min = background.get("bg_sim", {}).get("min")
    bg_min = bg_holdout_min if bg_holdout_min is not None else bg_all_min

    object_bg_maxes = [
        report["bg_sim"]["max"]
        for report in known.values()
        if report["bg_sim"]["max"] is not None
    ]
    object_bg_max = max(object_bg_maxes) if object_bg_maxes else None

    threshold_bg: float | None = None
    bg_ok = False
    if bg_min is None:
        reasons["bg"] = "缺少空场景帧，无法给出 T_bg。"
    elif object_bg_max is None:
        # 没有物体帧时，把阈值压到空场景最小值以下，保证空场景一定被门住；
        # 代价是可能把背景相似度很高的物体也误判成背景，必须补物体帧复核。
        threshold_bg = round(max(-1.0, float(bg_min) - 0.02), 4)
        reasons["bg"] = (
            f"只有空场景帧：取空场景 min {float(bg_min):.4f} - 0.02 的保守值；"
            "需补物体帧验证不会把物体误判成背景。"
        )
    elif object_bg_max < float(bg_min):
        threshold_bg = round((float(object_bg_max) + float(bg_min)) / 2.0, 4)
        bg_ok = True
        reasons["bg"] = (
            f"物体帧背景相似度 max {float(object_bg_max):.4f} 与空场景 min "
            f"{float(bg_min):.4f} 之间取中点。"
        )
    else:
        # 两分布重叠时不给数字：任何取值都会牺牲一侧，写进配置会被误当成
        # 校准结论。保留当前值，让人重拍空场景/物体帧后再跑。
        reasons["bg"] = (
            f"物体帧背景相似度 max {float(object_bg_max):.4f} ≥ 空场景 min "
            f"{float(bg_min):.4f}，两分布重叠：无可靠建议，"
            "请重拍空场景或换机位后重跑。"
        )

    known_min = min(
        (
            report["effective"]["min"]
            for report in known.values()
            if report["effective"]["min"] is not None
        ),
        default=None,
    )
    unknown_max = max(
        (
            report["effective"]["max"]
            for report in unknown.values()
            if report["effective"]["max"] is not None
        ),
        default=None,
    )
    bg_false_max = background.get("false_positive", {}).get("max")

    threshold_rt: float | None = None
    rt_ok = False
    rt_unknown_checked = unknown_max is not None
    if known_min is None:
        reasons["rt"] = "缺少已知物体帧，无法给出 T_rt；保留当前值。"
    elif unknown_max is not None and unknown_max < known_min:
        threshold_rt = round((float(known_min) + float(unknown_max)) / 2.0, 4)
        rt_ok = True
        reasons["rt"] = (
            f"已知 min {float(known_min):.4f} 与未知 max {float(unknown_max):.4f} "
            "之间取中点。"
        )
    elif unknown_max is not None:
        # 已知/未知重叠：不给数字，避免把"必然误伤一侧"的阈值写进配置。
        reasons["rt"] = (
            f"已知 min {float(known_min):.4f} ≤ 未知 max {float(unknown_max):.4f}，"
            "两分布重叠：无可靠建议，请补拍测试帧或检查未知样本是否与已知太像。"
        )
    else:
        threshold_rt = round(float(known_min) - 0.03, 4)
        if bg_false_max is not None and float(bg_false_max) + 0.02 <= float(known_min):
            threshold_rt = max(threshold_rt, round(float(bg_false_max) + 0.02, 4))
        threshold_rt = min(threshold_rt, round(float(known_min), 4))
        rt_ok = True
        reasons["rt"] = (
            f"缺少未录入物体帧：取已知 min {float(known_min):.4f} - 0.03，"
            "并抬到空场景误报 max 之上；能保证已知通过，但**未验证未知拒识**。"
        )

    if threshold_rt is not None:
        threshold_rt = max(-1.0, min(1.0, float(threshold_rt)))
    if threshold_bg is not None:
        threshold_bg = max(-1.0, min(1.0, float(threshold_bg)))

    return {
        "static": round(float(static), 6),
        "realtime_current": round(float(config.threshold_realtime), 6),
        "bg_current": round(float(config.threshold_bg), 6),
        "realtime_recommended": threshold_rt,
        "bg_recommended": threshold_bg,
        "reasons": reasons,
        "reasonable": {
            "bg": bool(bg_ok),
            "realtime": bool(rt_ok),
            "unknown_checked": bool(rt_unknown_checked),
            "overall": bool(bg_ok and rt_ok and rt_unknown_checked),
        },
    }


def format_report(report: dict[str, Any]) -> str:
    """把报告字典转成人类可读的中文文本。"""
    lines: list[str] = []
    lines.append("== 数据 ==")
    lines.append(f"特征库: {report['library'] or '(未绑定路径)'}")
    lines.append(f"帧目录: {report['frames_root']}")
    background = report["background"]
    lines.append(
        f"背景帧: {background['frames']} 张（学习用 {background['learned']} 张）"
    )
    lines.append(
        "已知物体帧: "
        + (
            ", ".join(f"{name}={item['n']}" for name, item in report["known"].items())
            or "(无)"
        )
    )
    lines.append(
        "未知物体帧: "
        + (
            ", ".join(
                f"{name}={item['n']}" for name, item in report["unknown"].items()
            )
            or "(无)"
        )
    )
    current = report["config"]
    lines.append(
        f"当前阈值: T_static={current['threshold_static']:.4f} "
        f"T_rt={current['threshold_realtime']:.4f} T_bg={current['threshold_bg']:.4f}"
    )

    lines.append("")
    lines.append("== 背景相似度基线（均值/最大） ==")
    for key, label in (
        ("bg_sim", "空场景（含学习帧）"),
        ("bg_sim_holdout", "空场景（留出帧）"),
    ):
        stats = background[key]
        if stats["n"]:
            lines.append(
                f"{label}: n={stats['n']} mean={stats['mean']:.4f} "
                f"max={stats['max']:.4f} min={stats['min']:.4f}"
            )
        else:
            lines.append(f"{label}: 无数据")
    for name, item in report["known"].items():
        stats = item["bg_sim"]
        if stats["n"]:
            lines.append(
                f"物体「{name}」在场: n={stats['n']} mean={stats['mean']:.4f} "
                f"max={stats['max']:.4f} min={stats['min']:.4f}"
            )

    lines.append("")
    lines.append("== 空场景误报（全帧最高分 vs 当前 T_rt） ==")
    false_positive = background["false_positive"]
    if false_positive["max"] is None:
        lines.append("无空场景打分数据。")
    else:
        lines.append(
            f"空场景最高分 max={false_positive['max']:.4f}，"
            f"≥ T_rt({false_positive['threshold']:.4f}) 的误报次数 "
            f"{false_positive['count']}/{background['frames']}"
        )

    lines.append("")
    lines.append("== 物体在场分数分布（min / mean / max） ==")
    lines.append(
        f"{'物体':<12} {'n':>3} {'全帧':<24} {'中心裁剪':<24} {'采用':<24} 裁剪次数"
    )

    def _fmt(stats: dict[str, Any]) -> str:
        if not stats["n"]:
            return "-"
        return f"{stats['min']:.4f}/{stats['mean']:.4f}/{stats['max']:.4f}"

    for group_name, groups in (("已知", report["known"]), ("未知", report["unknown"])):
        for name, item in groups.items():
            lines.append(
                f"{group_name}:{name:<8} {item['n']:>3} "
                f"{_fmt(item['full']):<24} {_fmt(item['crop']):<24} "
                f"{_fmt(item['effective']):<24} {item['used_crop']}"
            )

    lines.append("")
    lines.append("== 阈值建议（写回 data/realtime.json 前请人工复核） ==")
    thresholds = report["thresholds"]
    lines.append(
        f"T_static = {thresholds['static']:.4f}（特征库已校准值，保持不变）"
    )
    bg_value = thresholds["bg_recommended"]
    rt_value = thresholds["realtime_recommended"]
    if bg_value is not None:
        lines.append(
            f"T_bg     = {bg_value:.4f}（当前 {thresholds['bg_current']:.4f}）"
        )
    else:
        lines.append(f"T_bg     = 无建议（当前 {thresholds['bg_current']:.4f}）")
    lines.append(f"  依据: {thresholds['reasons'].get('bg', '-')}")
    if rt_value is not None:
        lines.append(
            f"T_rt     = {rt_value:.4f}"
            f"（当前 {thresholds['realtime_current']:.4f}，待真机校准）"
        )
    else:
        lines.append(f"T_rt     = 无建议（当前 {thresholds['realtime_current']:.4f}）")
    lines.append(f"  依据: {thresholds['reasons'].get('rt', '-')}")

    reasonable = thresholds["reasonable"]
    verdict = "合理" if reasonable["overall"] else "需要复核"
    lines.append(
        f"判定: {verdict}（T_bg={'可分' if reasonable['bg'] else '不可分/缺数据'}，"
        f"T_rt={'可分' if reasonable['realtime'] else '不可分/缺数据'}，"
        f"未知拒识={'已验证' if reasonable['unknown_checked'] else '未验证'}）"
    )

    if report["warnings"]:
        lines.append("")
        lines.append("== 警告 ==")
        for warning in report["warnings"]:
            lines.append(f"- {warning}")

    lines.append("")
    lines.append(
        "提示：以上只是阈值建议，不是现场验收结论；"
        "T_rt 必须在真实摄像头帧上复核，T_bg 必须在空场景下复核。"
    )
    return "\n".join(lines)


def _write_config(report: dict[str, Any], config: RealtimeConfig, path: Path) -> Path:
    """把建议值写回 realtime.json；没有建议的项保留当前值。"""
    thresholds = report["thresholds"]
    rt_value = thresholds["realtime_recommended"]
    bg_value = thresholds["bg_recommended"]
    updated = RealtimeConfig(
        threshold_static=thresholds["static"],
        threshold_realtime=(
            float(rt_value) if rt_value is not None else config.threshold_realtime
        ),
        threshold_bg=float(bg_value) if bg_value is not None else config.threshold_bg,
        gate_enabled=config.gate_enabled,
        crop_margin=config.crop_margin,
    )
    return save_realtime_config(updated, path)


def cmd_frames(args: argparse.Namespace) -> int:
    """``--frames`` 模式主流程。"""
    library = load_library(args.library, create=False)
    config = load_realtime_config(args.config, library=library)
    frame_set = discover_frames(args.frames, background_dir=args.background)
    encoder = get_default_encoder()

    report = evaluate_frames(
        frame_set,
        library,
        config,
        encoder,
        crop_margin=args.crop_margin,
        limit=args.limit,
        take=args.take,
        drop_first=args.drop_first,
    )

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_report(report))

    if args.write:
        target = _write_config(report, config, args.config)
        print(f"\n已写入: {target}", file=sys.stderr if args.json else sys.stdout)

    return 0 if report["thresholds"]["reasonable"]["overall"] else 1


def _open_camera(index: int) -> Any:
    """打开摄像头；缺少 opencv 时给出明确安装提示而不是 ImportError 堆栈。"""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "未安装 opencv-python，无法使用 --camera。"
            "请先执行 `pip install opencv-python`（仅自检脚本需要，"
            "应用本身不依赖 opencv）；或改用 --frames 离线帧目录。"
        ) from exc

    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"无法打开摄像头 {index}：请确认设备存在、未被其他程序占用，"
            "或改用 --frames 离线帧目录。"
        )
    return capture


def _read_rgb(capture: Any) -> Any | None:
    """读一帧并转 RGB；失败返回 ``None``。"""
    import cv2

    ok, frame = capture.read()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _capture_for(capture: Any, seconds: float, *, every: float = 0.3) -> list[Any]:
    """在 ``seconds`` 秒内按 ``every`` 间隔抓 RGB 帧。"""
    frames: list[Any] = []
    deadline = time.monotonic() + max(0.0, float(seconds))
    while time.monotonic() < deadline:
        frame = _read_rgb(capture)
        if frame is not None:
            frames.append(frame)
        time.sleep(every)
    return frames


def cmd_camera(args: argparse.Namespace) -> int:
    """``--camera`` 模式：空场景 -> 逐个物体 -> 可选未知物体。"""
    library = load_library(args.library, create=False)
    if library.total_vectors == 0:
        raise ValueError("特征库为空：请先录入物体再自检")

    config = load_realtime_config(args.config, library=library)
    encoder = get_default_encoder()

    known_names = (
        [item.strip() for item in args.objects.split(",") if item.strip()]
        if args.objects
        else [row["name"] for row in library.stats()]
    )
    unknown_names = (
        [item.strip() for item in args.unknowns.split(",") if item.strip()]
        if args.unknowns
        else []
    )
    total = max(1.0, float(args.seconds))
    bg_seconds = (
        float(args.bg_seconds)
        if args.bg_seconds is not None
        else max(6.0, min(10.0, total * 0.34))
    )
    bg_seconds = min(bg_seconds, total * 0.6)
    object_seconds = max(0.0, total - bg_seconds)
    slots = len(known_names) + len(unknown_names)
    per_slot = object_seconds / slots if slots else 0.0

    capture = _open_camera(args.camera)
    try:
        print(f"[1/2] 空场景：请让画面对准空场景，保持 {bg_seconds:.0f} 秒……")
        bg_frames = _capture_for(capture, bg_seconds)
        if len(bg_frames) <= args.drop_first:
            raise RuntimeError("空场景采到的帧太少，无法学背景；请延长 --seconds。")

        print(f"[2/2] 物体阶段：共 {slots} 个物体，每个约 {per_slot:.0f} 秒。")
        known_frames: dict[str, list[Any]] = {}
        unknown_frames: dict[str, list[Any]] = {}
        for name in known_names:
            print(f"  请把「{name}」放入画面并保持……")
            known_frames[name] = _capture_for(capture, per_slot)
        for name in unknown_names:
            print(f"  请把未录入物体「{name}」放入画面并保持……")
            unknown_frames[name] = _capture_for(capture, per_slot)
    finally:
        capture.release()

    try:
        background_model = BackgroundModel.learn(
            bg_frames,
            encoder=encoder,
            take=args.take,
            drop_first=args.drop_first,
        )
    except Exception as exc:
        raise RuntimeError(f"背景学习失败：{exc}") from exc

    margin = float(config.crop_margin if args.crop_margin is None else args.crop_margin)
    bg_report = _score_items(
        bg_frames,
        library,
        config,
        encoder,
        background_model,
        margin,
        label="background",
        name_for=lambda index: f"bg_{index:03d}",
    )
    background: dict[str, Any] = {
        "frames": len(bg_frames),
        "learned": background_model.frames_used,
        "bg_sim": bg_report["bg_sim"],
        "bg_sim_holdout": _stats([]),
        "false_positive": {
            "threshold": config.threshold_realtime,
            "max": bg_report["effective"]["max"],
            "count": sum(1 for row in bg_report["rows"] if row["is_known"]),
        },
    }
    holdout = bg_frames[args.drop_first + background_model.frames_used :]
    if holdout:
        holdout_report = _score_items(
            holdout,
            library,
            config,
            encoder,
            background_model,
            margin,
            label="background",
            name_for=lambda index: f"bg_holdout_{index:03d}",
        )
        background["bg_sim_holdout"] = holdout_report["bg_sim"]

    known_reports = {
        name: _score_items(
            frames,
            library,
            config,
            encoder,
            background_model,
            margin,
            label=name,
            name_for=lambda index, name=name: f"{name}_{index:03d}",
        )
        for name, frames in known_frames.items()
    }
    unknown_reports = {
        name: _score_items(
            frames,
            library,
            config,
            encoder,
            background_model,
            margin,
            label=name,
            name_for=lambda index, name=name: f"{name}_{index:03d}",
        )
        for name, frames in unknown_frames.items()
    }

    thresholds = recommend_thresholds(
        background=background,
        known=known_reports,
        unknown=unknown_reports,
        config=config,
        library=library,
    )
    report = {
        "library": str(getattr(library, "path", "") or ""),
        "frames_root": f"camera:{args.camera}",
        "config": config.to_dict(),
        "background": background,
        "known": known_reports,
        "unknown": unknown_reports,
        "thresholds": thresholds,
        "warnings": [],
    }

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_report(report))
    if args.write:
        target = _write_config(report, config, args.config)
        print(f"\n已写入: {target}", file=sys.stderr if args.json else sys.stdout)
    return 0 if thresholds["reasonable"]["overall"] else 1


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(
        prog="realtime_check.py",
        description="实时识别背景/阈值自检：空场景误报、物体分数分布、T_bg/T_rt 建议",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--frames", type=Path, help="离线帧目录（background/ + 物体名/）")
    source.add_argument("--camera", type=int, help="摄像头索引，例如 0")
    parser.add_argument(
        "--seconds", type=float, default=30.0, help="摄像头采集总时长（秒），默认 30"
    )
    parser.add_argument(
        "--bg-seconds",
        type=float,
        default=None,
        help="摄像头模式下空场景时长（秒）；默认取总时长的 1/3",
    )
    parser.add_argument(
        "--objects",
        type=str,
        default=None,
        help="摄像头模式要依次放入的已知物体名，逗号分隔；默认用库里的全部物体",
    )
    parser.add_argument(
        "--unknowns",
        type=str,
        default=None,
        help="摄像头模式要依次放入的未录入物体名，逗号分隔（用于验证拒识）",
    )
    parser.add_argument(
        "--background",
        type=Path,
        default=None,
        help="显式指定空场景帧目录（离线模式）",
    )
    parser.add_argument(
        "--library",
        type=Path,
        default=DEFAULT_LIBRARY_PATH,
        help=f"特征库 JSON 路径（默认 {DEFAULT_LIBRARY_PATH}）",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_REALTIME_CONFIG_PATH,
        help=f"实时配置 JSON 路径（默认 {DEFAULT_REALTIME_CONFIG_PATH}）",
    )
    parser.add_argument(
        "--crop-margin",
        type=float,
        default=None,
        help="覆盖配置里的裁剪 margin（默认用 realtime.json）",
    )
    parser.add_argument(
        "--take",
        type=int,
        default=DEFAULT_TAKE,
        help=f"背景学习最多取多少帧，默认 {DEFAULT_TAKE}",
    )
    parser.add_argument(
        "--drop-first",
        type=int,
        default=DEFAULT_DROP_FIRST,
        help=f"背景学习开头丢弃多少帧，默认 {DEFAULT_DROP_FIRST}",
    )
    parser.add_argument("--limit", type=int, default=None, help="每组最多取多少帧（调试）")
    parser.add_argument("--write", action="store_true", help="把建议阈值写回 realtime.json")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出完整报告")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口：0 表示建议合理，1 表示需要人工复核，2 表示运行错误。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.frames is not None:
            return int(cmd_frames(args))
        return int(cmd_camera(args))
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
