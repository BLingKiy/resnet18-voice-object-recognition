"""P4 验收脚本：真实 Edge TTS 的播报证据。

跑三段真实检查，全部用仓库里的真实 ResNet18 与课程数据，不 mock 识别：

1. **静态链路**：识别已知测试图与一张未录入图，分别合成并记录
   播报文本、合成耗时、音频时长（与 soundfile 独立读数对照）、音频路径；
2. **实时链路节流**：假帧源每 0.2 秒喂一次结果（苹果 -> 鼠标 -> 苹果），
   证明同一结果不重复、结果变化到点才播，并统计帧循环耗时证明不阻塞；
3. **断网/合成失败**：把 edge-tts 的 WebSocket 端点指到本机不可达端口，
   实测界面文案是否为「语音播报不可用，请检查网络」，且识别结果照常返回。

用法（在仓库根目录）::

    python tools/verify_tts.py                 # 三段全跑
    python tools/verify_tts.py --skip-frames   # 只跑静态与失败段

输出为 JSON，便于整段贴进任务评论。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image

import app
from core import load_library
from voice.tts import (
    TTS_COOLDOWN_SECONDS,
    TTS_UNAVAILABLE_MESSAGE,
    TtsThrottler,
    TtsResult,
)
from voice.tts import tts as tts_speak

DEFAULT_LIBRARY = PROJECT_ROOT / "data" / "features.json"
DEFAULT_KNOWN_IMAGE = PROJECT_ROOT / "data" / "test" / "water_cup" / "01.jpg"
UNKNOWN_IMAGE = PROJECT_ROOT / ".validation" / "unknown_noise.png"


def _record_synthesis() -> tuple[list[TtsResult], object]:
    """包一层 app.synthesize，收集每次真实合成结果。

    Returns:
        ``(记录列表, 原函数)``；调用方用完必须把 ``app.synthesize`` 还原，
        否则后面的段落会把结果混进同一个列表。
    """
    recorded: list[TtsResult] = []
    original = app.synthesize

    def spy(text: str, **kwargs: object) -> TtsResult:
        result = original(text, **kwargs)  # type: ignore[misc]
        recorded.append(result)
        return result

    app.synthesize = spy  # type: ignore[assignment]
    return recorded, original


def _audio_duration_ms(path: Path) -> int | None:
    """用 soundfile 独立读一次 mp3 时长，用来核对 TtsResult 的估算值。"""
    try:
        import soundfile as sf
    except Exception:
        return None
    try:
        info = sf.info(str(path))
        return int(round(info.frames / info.samplerate * 1000))
    except Exception:
        return None


def _make_unknown_image() -> Path:
    """生成一张未录入的噪声图，用于触发「未知物体」播报分支。

    这不是课程要求的未知物体样本（P6b 由人类提供真实未录入物体），
    只用来证明播报的未知分支确实会说「未知物体」。
    """
    UNKNOWN_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261109)
    noise = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    Image.fromarray(noise, "RGB").save(UNKNOWN_IMAGE)
    return UNKNOWN_IMAGE


def static_link(library_path: Path, known_image: Path) -> dict[str, object]:
    """第一段：静态链路逐次播报，已知与未知两种文本各跑一次。"""
    library = load_library(library_path)

    # 先把 ResNet18 权重读进内存：首次识别要 8~10 秒，混进"点击到返回"
    # 的耗时里会让人误以为播报很慢。
    warmup_started = time.perf_counter()
    app.recognize_image(known_image, library=library)
    warmup_ms = int(round((time.perf_counter() - warmup_started) * 1000))

    case_results: list[dict[str, object]] = []

    for label, image in (
        ("已知物体", known_image),
        ("未知物体", _make_unknown_image()),
    ):
        recorded, original = _record_synthesis()
        started = time.perf_counter()
        try:
            headline, annotated, rows, status, audio = app.recognize_and_speak(
                image, library=library
            )
        finally:
            app.synthesize = original  # type: ignore[assignment]
        wall_ms = int(round((time.perf_counter() - started) * 1000))

        assert recorded, f"{label}：没有调用合成"
        spoken = recorded[0]
        payload = {
            "场景": label,
            "输入图片": str(image),
            "识别结论首行": headline.splitlines()[0] if headline else "",
            "是否已知": "未知物体" not in headline.splitlines()[0],
            "播报文本": spoken.text,
            "合成耗时_ms": spoken.elapsed_ms,
            "音频时长_ms": spoken.duration_ms,
            "soundfile时长_ms": _audio_duration_ms(spoken.audio_path),
            "音频字节": spoken.bytes_written,
            "音频路径": str(spoken.audio_path),
            "音频文件存在": spoken.audio_path.is_file(),
            "界面状态行": status,
            "播放器返回值": audio,
            "从点击到返回_ms": wall_ms,
            "标注图生成": annotated is not None,
            "近邻行数": len(rows),
        }
        # 关键断言：播报文本必须来自本次识别，而不是固定话术。
        if label == "已知物体":
            assert spoken.text.startswith("识别结果："), spoken.text
            assert payload["是否已知"] is True, headline
        else:
            assert spoken.text == "未知物体", spoken.text
            assert payload["是否已知"] is False, headline
        case_results.append(payload)

    return {
        "特征库": {
            "路径": str(library_path),
            "物体数": library.object_count,
            "特征条数": library.total_vectors,
            "阈值": library.threshold,
        },
        "首次识别预热_ms": warmup_ms,
        "用例": case_results,
    }


def throttled_stream(
    frames: int = 60, interval: float = 0.2, cooldown: float = TTS_COOLDOWN_SECONDS
) -> dict[str, object]:
    """第二段：假帧源 + 真实合成，验证节流规则与"不阻塞帧循环"。"""

    def timeline(index: int) -> str:
        """三段稳定结果 + 一次"冷却窗口内换结果"的抖动。"""
        if index < 15:  # 0.0 ~ 2.8 s
            return "识别结果：水杯"
        if index < 20:  # 3.0 ~ 3.8 s：冷却窗口内换结果，应当被拦下
            return "识别结果：鼠标"
        if index < 35:  # 4.0 ~ 6.8 s：窗口打开后的新结果，应当立刻播
            return "识别结果：键盘"
        return "识别结果：水杯"  # 7.0 s 起：再换回来

    plays: list[dict[str, object]] = []
    frame_costs: list[float] = []
    origin = time.monotonic()

    def speak(text: str) -> str:
        plays.append({"秒": round(time.monotonic() - origin, 2), "文本": text})
        return tts_speak(text)

    throttler = TtsThrottler(cooldown_seconds=cooldown, speak=speak)

    for index in range(frames):
        begin = time.perf_counter()
        throttler.announce(timeline(index))
        frame_costs.append((time.perf_counter() - begin) * 1000)
        time.sleep(interval)

    finished = throttler.wait_idle(timeout=30)
    stats = throttler.stats()

    gaps = [
        round(plays[i + 1]["秒"] - plays[i]["秒"], 2)  # type: ignore[operator]
        for i in range(len(plays) - 1)
    ]
    assert len(plays) >= 2, "应当至少播报两次（结果变化）"
    assert all(gap >= cooldown - 0.3 for gap in gaps), f"冷却被击穿: {gaps}"

    return {
        "帧数": frames,
        "帧间隔_秒": interval,
        "冷却_秒": cooldown,
        "播报时间线": plays,
        "相邻播报间隔_秒": gaps,
        "帧循环耗时_ms": {
            "平均": round(statistics.fmean(frame_costs), 3),
            "中位": round(statistics.median(frame_costs), 3),
            "最大": round(max(frame_costs), 3),
        },
        "节流计数": {
            "尝试合成": stats["attempts"],
            "完成播报": stats["spoken"],
            "同文本拦截": stats["skipped_same"],
            "冷却拦截": stats["skipped_cooldown"],
            "忙碌拦截": stats["skipped_busy"],
        },
        "后台线程已收尾": finished,
    }


def offline_failure(library_path: Path, known_image: Path) -> dict[str, object]:
    """第三段：把 WebSocket 端点指到不可达端口，验证失败文案与识别不受影响。"""
    import edge_tts.communicate as communicate

    original_url = communicate.WSS_URL
    # 127.0.0.1:9 (discard) 通常直接拒绝连接，比断网更好复现。
    unreachable = (
        "wss://127.0.0.1:9/consumer/speech/synthesize/readaloud/edge/v1"
        "?TrustedClientToken=0000"
    )
    communicate.WSS_URL = unreachable
    try:
        library = load_library(library_path)
        started = time.perf_counter()
        headline, annotated, rows, status, audio = app.recognize_and_speak(
            known_image, library=library
        )
        wall_ms = int(round((time.perf_counter() - started) * 1000))
    finally:
        communicate.WSS_URL = original_url

    assert TTS_UNAVAILABLE_MESSAGE in status, status
    assert "识别结果：" in headline, "播报失败不应影响识别结果"
    assert annotated is not None and rows, "播报失败不应影响标注图与近邻表"

    return {
        "不可达端点": unreachable,
        "原端点": original_url,
        "界面提示文案": status,
        "文案是否逐字匹配": status == f"⚠️ {TTS_UNAVAILABLE_MESSAGE}",
        "识别结论首行": headline.splitlines()[0],
        "识别结果仍返回": True,
        "标注图生成": annotated is not None,
        "近邻行数": len(rows),
        "播放器返回值": audio,
        "从点击到返回_ms": wall_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="P4 Edge TTS 播报验收")
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--known-image", type=Path, default=DEFAULT_KNOWN_IMAGE)
    parser.add_argument("--skip-static", action="store_true")
    parser.add_argument("--skip-frames", action="store_true")
    parser.add_argument("--skip-offline", action="store_true")
    args = parser.parse_args()

    report: dict[str, object] = {}
    if not args.skip_static:
        report["一_静态链路"] = static_link(args.library, args.known_image)
    if not args.skip_frames:
        report["二_实时节流"] = throttled_stream()
    if not args.skip_offline:
        report["三_断网失败"] = offline_failure(args.library, args.known_image)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
