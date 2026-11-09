"""命令行验收工具：录入 -> 检索 -> 判未知 -> 重启复现。

典型用法::

    # 录入（目录或显式文件列表二选一）
    python tools/demo_cli.py add --name 苹果 --dir data/known/苹果
    python tools/demo_cli.py add --name 香蕉 --images data/known/香蕉/01.jpg data/known/香蕉/02.jpg

    # 查看特征库
    python tools/demo_cli.py stats

    # 单图识别
    python tools/demo_cli.py recognize --image data/test/苹果/t01.jpg --top-k 3

    # 批量评测（目录结构 data/test/<物体名>/*.jpg）
    python tools/demo_cli.py eval --dir data/test

    # 写入 T6 校准出的阈值
    python tools/demo_cli.py threshold --set 0.63

所有命令都支持 ``--json``，方便把验收结果贴进报告或做自动化断言。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# 允许 ``python tools/demo_cli.py`` 直接运行（此时 sys.path[0] 是 tools/，找不到 core）。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.library import (
    DEFAULT_LIBRARY_PATH,
    FeatureLibrary,
    add_to_library,
    load_library,
)
from core.recognize import (
    Recognition,
    recognize,
    resolve_threshold,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    """给每个子命令加上通用的 --library / --json。"""
    parser.add_argument(
        "--library",
        type=Path,
        default=DEFAULT_LIBRARY_PATH,
        help=f"特征库 JSON 路径（默认 {DEFAULT_LIBRARY_PATH}）",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 输出，便于脚本处理")


def _collect_images(path: Path) -> list[Path]:
    """收集单张图片或目录下所有图片文件。"""
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


def _load_for_read(path: Path) -> FeatureLibrary:
    """读取已有特征库；不存在时报错而不是静默建空库。"""
    return load_library(path, create=False)


def _emit(payload: dict[str, Any], as_json: bool, human_text: str) -> int:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(human_text.rstrip())
    return 0


def _format_threshold(library: FeatureLibrary) -> str:
    if library.threshold is None:
        return f"{resolve_threshold(library):.4f}（默认值，尚未校准）"
    return f"{library.threshold:.4f}"


def cmd_add(args: argparse.Namespace) -> int:
    """录入一个物体的多张图片，成功后立即写盘。"""
    source = args.dir if args.dir is not None else None
    if source is not None:
        images = _collect_images(source)
    else:
        images = []
        for item in args.images:
            images.extend(_collect_images(item))

    library = load_library(args.library, create=True)
    result = add_to_library(library, args.name, images)

    saved_path: str | None = None
    if result.added > 0:
        saved_path = str(library.save())

    payload = {
        "object": result.object_name,
        "added": result.added,
        "skipped_duplicates": result.skipped,
        "object_vectors": result.object_vectors,
        "total_vectors": result.total_vectors,
        "library": saved_path or str(args.library),
    }

    if result.added > 0:
        human = (
            f"[OK] 已录入「{result.object_name}」：新增 {result.added} 条特征，"
            f"跳过重复 {result.skipped} 张\n"
            f"     该物体 {result.object_vectors} 条 / 全库 {result.total_vectors} 条\n"
            f"     特征库: {saved_path}"
        )
    else:
        human = (
            f"[提示] 「{result.object_name}」没有新增特征：{result.skipped} 张图都已在库中\n"
            f"     该物体 {result.object_vectors} 条 / 全库 {result.total_vectors} 条"
        )
    return _emit(payload, args.json, human)


def _recognition_payload(result: Recognition) -> dict[str, Any]:
    return {
        "is_known": result.is_known,
        "name": result.name,
        "similarity": round(result.similarity, 6),
        "similarity_percent": result.similarity_percent,
        "threshold": round(result.threshold, 6),
        "matches": [
            {
                "rank": match.rank,
                "name": match.name,
                "image": match.image,
                "similarity": round(match.similarity, 6),
            }
            for match in result.matches
        ],
        "objects": [
            {
                "name": score.name,
                "max_similarity": round(score.max_similarity, 6),
                "mean_similarity": round(score.mean_similarity, 6),
                "best_image": score.best_image,
                "count": score.count,
            }
            for score in result.object_scores
        ],
    }


def _format_recognition(result: Recognition, top_k: int) -> str:
    if not result.matches:
        return "特征库为空，无法识别。请先用 add 命令录入物体。"

    lines = [f"识别结果: {result.name}"]
    if result.is_known:
        lines.append(
            f"相似度:   {result.similarity:.4f} ({result.similarity_percent:.2f}%)"
        )
        lines.append(f"判定阈值: {result.threshold:.4f}  ->  已知物体")
    else:
        lines.append(
            f"最高相似度 {result.similarity:.4f} 低于阈值 {result.threshold:.4f}"
            "  ->  未知物体"
        )

    lines.append(f"Top-{min(top_k, len(result.matches))} 近邻:")
    for match in result.matches[:top_k]:
        lines.append(
            f"  {match.rank}. {match.name:<8} {match.similarity:.4f}  {match.image}"
        )
    return "\n".join(lines)


def cmd_recognize(args: argparse.Namespace) -> int:
    """识别单张图片。"""
    image = args.image
    if not image.is_file():
        raise FileNotFoundError(f"图片不存在: {image}")

    library = _load_for_read(args.library)
    if library.total_vectors == 0:
        raise ValueError(f"特征库为空: {args.library}，请先录入物体")

    result = recognize(
        library,
        image,
        threshold=args.threshold,
        top_k=args.top_k,
    )
    payload = _recognition_payload(result)
    payload["image"] = str(image)
    return _emit(payload, args.json, _format_recognition(result, args.top_k))


def cmd_stats(args: argparse.Namespace) -> int:
    """查看特征库内容。"""
    library = _load_for_read(args.library)
    rows = library.stats()

    payload = {
        "library": str(args.library),
        "model": library.model,
        "dim": library.dim,
        "threshold": library.threshold,
        "effective_threshold": resolve_threshold(library),
        "object_count": library.object_count,
        "total_vectors": library.total_vectors,
        "objects": rows,
    }

    header = (
        f"特征库: {args.library}\n"
        f"模型: {library.model}  dim={library.dim}  阈值={_format_threshold(library)}\n"
        f"物体数: {library.object_count}   特征向量: {library.total_vectors}"
    )
    if not rows:
        human = header + "\n（空库，请先用 add 命令录入物体）"
    else:
        lines = [header, "", f"{'名称':<12}{'向量数':>6}   首张图片"]
        for row in rows:
            lines.append(f"{row['name']:<12}{row['count']:>6}   {row['first_image']}")
        human = "\n".join(lines)

    return _emit(payload, args.json, human)


def cmd_eval(args: argparse.Namespace) -> int:
    """按目录结构批量评测 Top-1 准确率。"""
    root = args.dir
    if not root.is_dir():
        raise FileNotFoundError(f"测试目录不存在: {root}")

    library = _load_for_read(args.library)
    if library.total_vectors == 0:
        raise ValueError(f"特征库为空: {args.library}，请先录入物体")

    class_dirs = sorted(item for item in root.iterdir() if item.is_dir() and not item.name.startswith("."))
    if not class_dirs:
        raise ValueError(f"测试目录下没有按物体名分好的子目录: {root}")

    threshold = resolve_threshold(library, args.threshold)
    rows: list[dict[str, Any]] = []

    for class_dir in class_dirs:
        truth = class_dir.name
        for image in _collect_images(class_dir):
            result = recognize(library, image, threshold=threshold, top_k=1)
            correct = bool(result.is_known and result.name == truth)
            rows.append(
                {
                    "image": str(image.relative_to(root)),
                    "truth": truth,
                    "predicted": result.name,
                    "similarity": round(result.similarity, 6),
                    "is_known": result.is_known,
                    "correct": correct,
                }
            )

    if not rows:
        raise ValueError(f"测试目录中没有图片: {root}")

    total = len(rows)
    correct_count = sum(1 for row in rows if row["correct"])
    accuracy = correct_count / total

    payload = {
        "test_dir": str(root),
        "library": str(args.library),
        "threshold": round(threshold, 6),
        "total": total,
        "correct": correct_count,
        "accuracy": round(accuracy, 6),
        "rows": rows,
    }

    lines = [
        f"测试集: {root}",
        f"阈值: {threshold:.4f}   图片数: {total}",
        "",
    ]
    for row in rows:
        mark = "OK  " if row["correct"] else "FAIL"
        lines.append(
            f"[{mark}] {row['image']:<28} 真实={row['truth']:<8} "
            f"预测={row['predicted']:<8} 相似度={row['similarity']:.4f}"
        )
    lines.append("")
    lines.append(f"Top-1 正确: {correct_count}/{total}   准确率: {accuracy:.2%}")
    return _emit(payload, args.json, "\n".join(lines))


def cmd_threshold(args: argparse.Namespace) -> int:
    """查看或写入判未知阈值。"""
    library = _load_for_read(args.library)

    if args.set_value is not None:
        library.set_threshold(args.set_value)
        saved = library.save()
        payload = {
            "library": str(saved),
            "threshold": library.threshold,
            "saved": True,
        }
        human = f"[OK] 阈值已设为 {library.threshold:.4f} 并写入 {saved}"
        return _emit(payload, args.json, human)

    payload = {
        "library": str(args.library),
        "threshold": library.threshold,
        "effective_threshold": resolve_threshold(library),
        "saved": False,
    }
    human = f"当前阈值: {_format_threshold(library)}"
    return _emit(payload, args.json, human)


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(
        prog="demo_cli.py",
        description="ResNet18 特征库命令行验收工具",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="录入物体图片并保存特征")
    _add_common_options(add_parser)
    add_parser.add_argument("--name", required=True, help="物体名称")
    source = add_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", nargs="+", type=Path, help="一张或多张注册图片")
    source.add_argument("--dir", type=Path, help="注册图片所在目录")
    add_parser.set_defaults(handler=cmd_add)

    recognize_parser = subparsers.add_parser("recognize", help="识别单张图片")
    _add_common_options(recognize_parser)
    recognize_parser.add_argument("--image", required=True, type=Path, help="待识别图片")
    recognize_parser.add_argument("--top-k", type=int, default=3, help="展示前 k 个近邻")
    recognize_parser.add_argument(
        "--threshold", type=float, default=None, help="临时覆盖判未知阈值"
    )
    recognize_parser.set_defaults(handler=cmd_recognize)

    stats_parser = subparsers.add_parser("stats", help="查看特征库")
    _add_common_options(stats_parser)
    stats_parser.set_defaults(handler=cmd_stats)

    eval_parser = subparsers.add_parser("eval", help="按 data/test/<物体名>/ 批量评测 Top-1")
    _add_common_options(eval_parser)
    eval_parser.add_argument("--dir", required=True, type=Path, help="测试集根目录")
    eval_parser.add_argument(
        "--threshold", type=float, default=None, help="临时覆盖判未知阈值"
    )
    eval_parser.set_defaults(handler=cmd_eval)

    threshold_parser = subparsers.add_parser("threshold", help="查看或设置判未知阈值")
    _add_common_options(threshold_parser)
    threshold_parser.add_argument("--set", dest="set_value", type=float, default=None, help="写入新阈值")
    threshold_parser.set_defaults(handler=cmd_threshold)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口，返回进程退出码（0 成功，2 业务错误）。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
