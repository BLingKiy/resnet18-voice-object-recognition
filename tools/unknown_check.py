"""未知样本拒识验证：统计误报率（把未录入物体误判成已知的比例）。

只做一件事——把 ``data/unknown/<物体名>/`` 里每张图按当前阈值判一遍，
逐张给出最近邻、最高分与判定，最后汇总误报率。它是 ``calibrate.py`` 的
**下游复核**：calibrate 只统计分数分布，本工具验证"阈值写进库之后，
这些未知图是不是真的被拒掉了"。

典型用法::

    python tools/unknown_check.py --library ~/.rn18/data/features.json \
        --dir ~/.rn18/data/unknown

    # 机器可读，便于贴进报告或做断言
    python tools/unknown_check.py --dir ~/.rn18/data/unknown --json

数据布局（与 calibrate.py 一致）::

    data/unknown/<物体名>/*.jpg

判定口径：``recognize`` 返回的 ``is_known`` 为 ``True`` 即**误报**（未知物体被
当成已知物）。阈值一律从特征库里已校准的值取（可用 ``--threshold`` 临时覆盖），
不在此处写死任何数字。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# 允许 ``python tools/unknown_check.py`` 直接运行（此时 sys.path[0] 是 tools/）。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.library import DEFAULT_LIBRARY_PATH, load_library
from core.recognize import recognize, resolve_threshold

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _class_dirs(root: Path) -> list[Path]:
    """取根目录下按物体名分好的子目录。"""
    if not root.is_dir():
        raise FileNotFoundError(f"目录不存在: {root}")
    dirs = sorted(item for item in root.iterdir() if item.is_dir() and not item.name.startswith("."))
    if not dirs:
        raise ValueError(f"目录下没有按物体名分好的子目录: {root}")
    return dirs


def _images(class_dir: Path) -> list[Path]:
    """取单个类目录下的图片文件（排序后返回，保证输出可复现）。"""
    files = sorted(
        item
        for item in class_dir.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES and not item.name.startswith(".")
    )
    if not files:
        raise ValueError(f"目录中没有图片: {class_dir}")
    return files


def collect_rows(library: Any, root: Path, threshold: float | None) -> list[dict[str, Any]]:
    """逐图跑一次识别，返回每张图的结果行。"""
    rows: list[dict[str, Any]] = []
    for class_dir in _class_dirs(root):
        for image in _images(class_dir):
            result = recognize(library, image, threshold=threshold, top_k=1)
            rows.append(
                {
                    "class": class_dir.name,
                    "image": image.name,
                    "nearest": result.name if result.is_known else result.object_scores[0].name,
                    "similarity": round(float(result.similarity), 6),
                    "is_known": bool(result.is_known),
                    "false_positive": bool(result.is_known),
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """汇总拒识率与误报率，并给出最高分样本（决定阈值上界的那张）。"""
    total = len(rows)
    false_positive = sum(1 for row in rows if row["false_positive"])
    worst = max(rows, key=lambda row: row["similarity"])
    return {
        "threshold": round(threshold, 6),
        "total": total,
        "rejected": total - false_positive,
        "false_positive": false_positive,
        "false_positive_rate": round(false_positive / total, 6) if total else 0.0,
        "rejection_rate": round((total - false_positive) / total, 6) if total else 0.0,
        "highest": {
            "class": worst["class"],
            "image": worst["image"],
            "nearest": worst["nearest"],
            "similarity": worst["similarity"],
            "margin": round(threshold - worst["similarity"], 6),
        },
    }


def cmd_check(args: argparse.Namespace) -> int:
    """拒识验证主流程。"""
    library = load_library(args.library, create=False)
    if library.total_vectors == 0:
        raise ValueError(f"特征库为空: {args.library}，请先录入物体再验证")
    threshold = resolve_threshold(library, args.threshold)

    rows = collect_rows(library, args.dir, args.threshold)
    summary = summarize(rows, threshold)

    if args.json:
        print(json.dumps({"dir": str(args.dir), **summary, "rows": rows}, ensure_ascii=False, indent=2))
        return 0

    print(f"特征库: {args.library}")
    print(f"阈值: {threshold:.4f}")
    print(f"未知样本: {args.dir}（{summary['total']} 张）")
    print("")
    for row in rows:
        verdict = "误报（当成已知）" if row["false_positive"] else "正确拒识"
        print(
            f"  {row['class']:<8} {row['image']:<14} 最近邻={row['nearest']:<12} "
            f"sim={row['similarity']:.4f} -> {verdict}"
        )
    print("")
    print(
        f"拒识 {summary['rejected']}/{summary['total']}，"
        f"误报 {summary['false_positive']}/{summary['total']}"
        f"（误报率 {summary['false_positive_rate']:.2%}）"
    )
    highest = summary["highest"]
    print(
        f"最高分样本: {highest['class']}/{highest['image']} "
        f"sim={highest['similarity']:.4f}（最近邻 {highest['nearest']}），"
        f"距阈值 {highest['margin']:+.4f}"
    )
    return 0 if summary["false_positive"] == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(
        prog="unknown_check.py",
        description="用已校准阈值验证未知样本的拒识率与误报率",
    )
    parser.add_argument(
        "--library",
        type=Path,
        default=DEFAULT_LIBRARY_PATH,
        help=f"特征库 JSON 路径（默认 {DEFAULT_LIBRARY_PATH}）",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="未知样本根目录，结构 data/unknown/<物体名>/*.jpg",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="临时覆盖阈值（仅用于对比，不写库）",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 输出，便于脚本处理")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口：0 表示全部正确拒识，1 表示存在误报，2 表示业务错误。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(cmd_check(args))
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
