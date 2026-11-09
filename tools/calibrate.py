"""阈值校准工具（T6 / P6 的一部分）。

用途：用**未参与录入**的测试图，分别统计"已知类最高相似度"与"未知类最高相似度"
两组分布，给出建议阈值（两组中点）。阈值是判定"未知物体"的唯一依据，必须如此校准，
不能随意拍脑袋或沿用代码里的占位值。

典型用法::

    # 只给已知类测试图（当前阶段）：输出已知类分布，未知类显式标"待补数据"
    python tools/calibrate.py --known-dir data/test

    # 未知类样本补齐后：两组分布都算，给出建议阈值
    python tools/calibrate.py --known-dir data/test --unknown-dir data/unknown

    # 机器可读输出，便于贴进报告
    python tools/calibrate.py --known-dir data/test --unknown-dir data/unknown --json

数据布局（与 demo_cli 一致）::

    data/test/<物体名>/*.jpg      已知类测试图（每物体 ≥2 张，与注册图严格分开）
    data/unknown/<物体名>/*.jpg   未知类样本（≥3 类、每类 2~3 张，验收前补齐）

全部命令支持 ``--json``；阈值优先使用特征库里已校准的值，否则用占位值 0.60。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# 允许 ``python tools/calibrate.py`` 直接运行（此时 sys.path[0] 是 tools/，找不到 core）。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.library import (
    DEFAULT_LIBRARY_PATH,
    FeatureLibrary,
    load_library,
)
from core.recognize import (
    Recognition,
    recognize,
    resolve_threshold,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PENDING_UNKNOWN_TEXT = "待补数据（未知类样本未采集，验收前补齐 ≥3 类各 2~3 张）"


def _collect_images(path: Path) -> list[Path]:
    """收集单张图片或目录下所有图片文件（与 demo_cli 同款逻辑）。"""
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"不支持的图片格式: {path.suffix}（{path}）")
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"图片路径不存在: {path}")

    files = sorted(
        item
        for item in path.iterdir()
        if item.is_file()
        and item.suffix.lower() in IMAGE_SUFFIXES
        and not item.name.startswith(".")
    )
    if not files:
        raise ValueError(f"目录中没有图片: {path}")
    return files


def _class_dirs(root: Path) -> list[Path]:
    """取根目录下按物体名分好的子目录。"""
    if not root.is_dir():
        raise FileNotFoundError(f"目录不存在: {root}")
    dirs = sorted(
        item for item in root.iterdir() if item.is_dir() and not item.name.startswith(".")
    )
    if not dirs:
        raise ValueError(f"目录下没有按物体名分好的子目录: {root}")
    return dirs


def _gather_similarities(
    library: FeatureLibrary,
    root: Path,
    *,
    by_class: bool,
) -> tuple[list[float], dict[str, list[float]], int]:
    """对每个测试图取最高相似度，返回（扁平列表, 按类分组, 图片总数）。"""
    flat: list[float] = []
    per_class: dict[str, list[float]] = {}
    count = 0

    scan_dirs = _class_dirs(root) if by_class else [root]
    for class_dir in scan_dirs:
        label = class_dir.name if by_class else "_all"
        for image in _collect_images(class_dir):
            result: Recognition = recognize(library, image, top_k=1)
            similarity = float(result.similarity)
            flat.append(similarity)
            per_class.setdefault(label, []).append(similarity)
            count += 1
    return flat, per_class, count


def _stats(values: list[float]) -> dict[str, float]:
    """一组相似度的 min/max/mean（空列表返回全 0）。"""
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "count": 0.0}
    return {
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "mean": round(sum(values) / len(values), 6),
        "count": float(len(values)),
    }


def _emit(payload: dict[str, Any], as_json: bool, human_text: str) -> int:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(human_text.rstrip())
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """校准主流程。"""
    library = load_library(args.library, create=False)
    if library.total_vectors == 0:
        raise ValueError(f"特征库为空: {args.library}，请先录入物体再校准阈值")

    threshold = resolve_threshold(library, args.threshold)

    # ---- 已知类分布（必须有）----
    known_flat, known_per_class, known_count = _gather_similarities(
        library, args.known_dir, by_class=True
    )
    known_stats = _stats(known_flat)

    # ---- 未知类分布（可能缺）----
    unknown_pending = False
    unknown_stats: dict[str, float] = {}
    unknown_per_class: dict[str, list[float]] = {}
    unknown_count = 0
    if args.unknown_dir is None or not Path(args.unknown_dir).is_dir():
        unknown_pending = True
    else:
        unknown_flat, unknown_per_class, unknown_count = _gather_similarities(
            library, args.unknown_dir, by_class=True
        )
        if unknown_count == 0:
            unknown_pending = True
        else:
            unknown_stats = _stats(unknown_flat)

    # ---- 建议阈值（仅当两组都有数据时）----
    suggested = None
    if not unknown_pending and known_flat and unknown_flat:
        suggested = round((known_stats["min"] + unknown_stats["max"]) / 2.0, 4)

    payload: dict[str, Any] = {
        "library": str(args.library),
        "effective_threshold": round(threshold, 6),
        "known": {
            "dir": str(args.known_dir),
            "count": known_count,
            "stats": known_stats,
            "per_class": {
                name: {
                    "count": len(vals),
                    "min": round(min(vals), 6),
                    "max": round(max(vals), 6),
                }
                for name, vals in sorted(known_per_class.items())
            },
        },
        "unknown": {
            "pending": unknown_pending,
            "count": unknown_count,
            "stats": unknown_stats if not unknown_pending else None,
        },
        "suggested_threshold": suggested,
    }

    lines = [
        f"特征库: {args.library}",
        f"当前（占位/已校准）阈值: {threshold:.4f}",
        "",
        f"[已知类] 测试图 {known_count} 张",
        f"  最高相似度  min={known_stats['min']:.4f}  max={known_stats['max']:.4f}"
        f"  mean={known_stats['mean']:.4f}",
    ]
    for name, vals in sorted(known_per_class.items()):
        lines.append(
            f"    {name:<10} n={len(vals):<3} min={min(vals):.4f} max={max(vals):.4f}"
        )

    if unknown_pending:
        lines.append("")
        lines.append(f"[未知类] {PENDING_UNKNOWN_TEXT}")
    else:
        lines.append("")
        lines.append(f"[未知类] 样本 {unknown_count} 张")
        lines.append(
            f"  最高相似度  min={unknown_stats['min']:.4f}"
            f"  max={unknown_stats['max']:.4f}  mean={unknown_stats['mean']:.4f}"
        )

    lines.append("")
    if suggested is not None:
        lines.append(
            f"建议阈值（已知 min 与未知 max 中点）: {suggested:.4f}"
        )
        lines.append(
            "→ 用 `python tools/demo_cli.py threshold --set <值>` 写回特征库。"
        )
    else:
        lines.append(
            "建议阈值：未知类样本补齐后才会计算。当前不得把占位值当作最终结论。"
        )

    return _emit(payload, args.json, "\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(
        prog="calibrate.py",
        description="用未参与录入的测试图校准判未知阈值",
    )
    parser.add_argument(
        "--library",
        type=Path,
        default=DEFAULT_LIBRARY_PATH,
        help=f"特征库 JSON 路径（默认 {DEFAULT_LIBRARY_PATH}）",
    )
    parser.add_argument(
        "--known-dir",
        type=Path,
        required=True,
        help="已知类测试图根目录，结构 data/test/<物体名>/*.jpg",
    )
    parser.add_argument(
        "--unknown-dir",
        type=Path,
        default=None,
        help="未知类样本根目录，结构 data/unknown/<物体名>/*.jpg（验收前补齐）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="临时覆盖判未知阈值（仅影响本次统计展示，不写库）",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 输出，便于脚本处理")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口，返回进程退出码（0 成功，2 业务错误）。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(cmd_calibrate(args))
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
