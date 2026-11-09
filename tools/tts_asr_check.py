"""用 Edge TTS 合成物体名 -> 百度短语音识别回读，产出可复现的正确率表。

这张表用于 RN18-5（P3 语音输入）判据 1 的证据：
"用 TTS 合成音频作为输入，不少于 10 条，正确不少于 8 条"。

用法::

    python tools/tts_asr_check.py                      # 10 条默认词表，打印表格
    python tools/tts_asr_check.py --out docs/tts_asr_evidence.md
    python tools/tts_asr_check.py --audio-dir data/tts_evidence

说明（写进报告时要照抄这三条口径）：

1. **这是合成音频验证，不是真人录音验证**。它能证明"音频链路 +
   百度接口 + 文本长度"没问题，不能替代真人现场的识别率结论。
2. 判定规则：去掉标点与空白后，识别文本与目标词完全相等算 ``严格``；
   识别文本包含目标词（或多说了几个字）算 ``包含``；其余算 ``否``。
   两种匹配都记为正确，但表格里分开标注，避免把宽松匹配包装成严格正确。
3. 合成文件默认落在 ``data/tts_evidence/``，可直接作为提交材料里的样例音频。
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from voice import AsrError, TtsError, asr, synthesize  # noqa: E402

DEFAULT_WORDS = (
    "充电宝",
    "水杯",
    "耳机",
    "键盘",
    "鼠标",
    "台灯",
    "苹果",
    "香蕉",
    "书本",
    "眼镜",
)

# 中英文标点 + 空白；百度经常在结果末尾补一个句号或逗号。
_PUNCTUATION = re.compile(r"[\s，。、！？,.!?；;：:\"'“”‘’（）()【】\[\]…—-]+")


def normalize(text: str) -> str:
    """去掉标点与空白，只留下用于比较的字符。

    Args:
        text: 原始文本（可能是空字符串）。

    Returns:
        归一化后的文本。
    """
    return _PUNCTUATION.sub("", text or "")


def match_kind(target: str, recognized: str) -> str:
    """判断识别结果与目标词的匹配程度。

    Args:
        target: 本轮要说出的物体名。
        recognized: 百度返回的文本。

    Returns:
        ``"严格"`` / ``"包含"`` / ``"否"``。
    """
    want = normalize(target)
    got = normalize(recognized)
    if not want or not got:
        return "否"
    if want == got:
        return "严格"
    if want in got:
        return "包含"
    return "否"


def render_table(rows: list[tuple[int, str, str, str, str]]) -> str:
    """把结果行渲染成 Markdown 表格。

    Args:
        rows: ``(序号, 目标词, 识别文本, 匹配程度, 音频文件名)`` 列表。

    Returns:
        Markdown 文本（不含标题）。
    """
    lines = [
        "| # | 合成文本 | ASR 识别结果 | 匹配 | 正确 | 音频 |",
        "|---|---|---|---|---|---|",
    ]
    for index, target, recognized, kind, audio in rows:
        correct = "✅" if kind in {"严格", "包含"} else "❌"
        lines.append(
            f"| {index} | {target} | {recognized or '（空）'} | {kind} | "
            f"{correct} | `{audio}` |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """跑完整流程：逐词合成 -> asr 回读 -> 打印/落盘表格。

    Args:
        argv: 命令行参数，``None`` 表示使用 ``sys.argv``。

    Returns:
        进程退出码：全部正确返回 0，否则返回 1（便于 CI 用）。
    """
    parser = argparse.ArgumentParser(description="Edge TTS -> 百度 ASR 回读自检")
    parser.add_argument(
        "--words",
        nargs="*",
        default=list(DEFAULT_WORDS),
        help="要合成的物体名，默认 10 个",
    )
    parser.add_argument(
        "--audio-dir",
        default=str(Path(tempfile.gettempdir()) / "rn18-tts-evidence"),
        help="合成音频落盘目录（默认系统临时目录，不污染仓库）",
    )
    parser.add_argument(
        "--out",
        default="",
        help="把 Markdown 表格追加写入该文件；留空则只打印",
    )
    args = parser.parse_args(argv)

    audio_dir = Path(args.audio_dir).expanduser()
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[int, str, str, str, str]] = []
    failures = 0
    for index, word in enumerate(args.words, start=1):
        try:
            path = synthesize(
                word,
                out_dir=audio_dir,
                # 复用默认音色；这里不节流，一条词一条音频。
            )
        except TtsError as exc:
            print(f"[{index}] 合成失败：{word} -> {exc}")
            failures += 1
            rows.append((index, word, f"合成失败：{exc}", "否", "-"))
            continue

        try:
            recognized = asr(path)
        except AsrError as exc:
            print(f"[{index}] 识别失败：{word} -> {exc}")
            failures += 1
            rows.append((index, word, f"识别失败：{exc}", "否", path.name))
            continue

        kind = match_kind(word, recognized)
        if kind not in {"严格", "包含"}:
            failures += 1
        rows.append((index, word, recognized, kind, path.name))
        print(f"[{index}] {word} -> {recognized!r} ({kind})")

    table = render_table(rows)
    correct = sum(1 for row in rows if row[3] in {"严格", "包含"})
    summary = f"正确 {correct}/{len(rows)}（失败 {failures}）"
    print()
    print(table)
    print()
    print(summary)

    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            "# Edge TTS -> 百度 ASR 回读结果\n\n"
            "口径：合成音频验证，非真人录音；标点与空白已归一化。\n\n"
            f"{table}\n\n**{summary}**\n",
            encoding="utf-8",
        )
        print(f"已写入 {out_path}")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
