"""P5 证据脚本：在没有摄像头/麦克风的机器上，把实时识别与教学闭环跑通并留下证据。

它**不伪造硬件**：摄像头帧用真实测试图片组成的"预置帧序列"模拟，
教学抓帧用 5 张不同图片按 0.3 s 间隔喂入，ASR 走真实百度接口
（输入是 Edge TTS 合成的预录音频，报告中必须注明"合成音频、非真人录音"），
TTS 走真实 Edge TTS 并记录产出的 mp3 大小。

脚本做的事：

1. 复制一份真实特征库到输出目录，移除 ``--teach-object``（默认 ``mouse``），
   得到"4 个物体 / 20 条特征"的教学前状态，绝不改动原始库；
2. 30 帧连续识别计时（首帧冷启动 + 均值/最大值/P95），保存已知/未知标注图；
3. 用真实 ``SpeechGate`` 跑一段"同结果不播 / 冷却内换结果不播 / 冷却后换结果播"；
4. 教学闭环：抓 5 帧（0.3 s 间隔）-> 真实百度 ASR -> 回填 -> 确认 -> 库 +5
   -> 用**未参与教学的测试图**再识别同一物体并触发播报；
5. 取消教学、键盘兜底、坏帧隔离各跑一遍，记录库条数与磁盘哈希不变；
6. 输出 ``evidence.json`` + 标注图 PNG + 播报 mp3 路径。

用法::

    python tools/p5_evidence.py --data-dir ~/.rn18/data --out p5-evidence

依赖：真实 ResNet18 权重已在 ``~/.cache/torch/hub/checkpoints``，百度 Key
按 ``voice/asr.py`` 的四级优先级读取（本机为 ``~/.rn18/.env``）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app  # noqa: E402
from core.features import get_default_encoder  # noqa: E402
from core.library import FeatureLibrary, load_library  # noqa: E402
from voice import AsrError, TtsError, asr, synthesize  # noqa: E402

# 中文标点 + 空白：百度常在结果末尾补句号，比较名称时忽略。
_PUNCTUATION = re.compile(r"[\s，。、！？,.!?；;：:\"'“”‘’（）()【】\[\]…—-]+")


def normalize(text: str) -> str:
    """去掉标点与空白，只保留用于比较的字符。"""
    return _PUNCTUATION.sub("", text or "")


def sha256_file(path: Path) -> str:
    """返回文件 SHA-256，用于证明取消教学没有改动磁盘。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rgb(path: Path) -> np.ndarray:
    """读取图片为 RGB numpy 数组（模拟一帧画面）。"""
    with Image.open(path) as opened:
        return np.asarray(opened.convert("RGB"))


def image_paths(data_dir: Path, split: str, name: str) -> list[Path]:
    """按文件名排序列出某物体在 known/test 下的全部图片。"""
    directory = data_dir / split / name
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )


def install_recorder() -> list[dict[str, Any]]:
    """把后台播报替换成"只记录文本"，避免计时循环发起几十次网络合成。"""
    spoken: list[dict[str, Any]] = []

    def recorder(text: str, *, on_success: Any = None, on_error: Any = None, **_: Any):
        spoken.append({"text": text, "at": time.perf_counter()})

        class _NoopThread:
            def join(self, timeout: float | None = None) -> None:  # noqa: D401
                return None

        return _NoopThread()

    app.speak_in_background = recorder  # type: ignore[assignment]
    return spoken


def synthesize_for_teaching(text: str, out_dir: Path) -> dict[str, Any]:
    """真实 Edge TTS 合成一句物体名，作为 ASR 的预录音频输入。"""
    started = time.perf_counter()
    try:
        path = synthesize(text, out_dir=out_dir)
    except TtsError as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "ok": True,
        "text": text,
        "path": str(path),
        "bytes": path.stat().st_size,
        "elapsed_s": round(time.perf_counter() - started, 3),
    }


def transcribe_audio(audio_path: Path, out_dir: Path) -> dict[str, Any]:
    """真实百度 ASR；mp3 解码失败时用 ffmpeg 转 16k 单声道 wav 再试一次。"""
    started = time.perf_counter()
    used = audio_path
    try:
        text = asr(audio_path)
    except AsrError as first_error:
        if audio_path.suffix.lower() == ".wav":
            return {"ok": False, "error": f"{type(first_error).__name__}: {first_error}"}
        wav = out_dir / f"{audio_path.stem}-16k.wav"
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(audio_path),
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    str(wav),
                ],
                check=True,
            )
        except Exception as exc:  # noqa: BLE001 - 现场兜底失败要说明原因
            return {
                "ok": False,
                "error": (
                    f"mp3 解码失败（{type(first_error).__name__}: {first_error}），"
                    f"ffmpeg 转 wav 也失败（{type(exc).__name__}: {exc}）"
                ),
            }
        used = wav
        try:
            text = asr(wav)
        except AsrError as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    return {
        "ok": True,
        "text": text,
        "normalized": normalize(text),
        "engine": "百度短语音识别（真实接口）",
        "audio_path": str(used),
        "audio_note": "Edge TTS 合成音频（预录音频，非真人录音）",
        "elapsed_s": round(time.perf_counter() - started, 3),
    }


def copy_library(source: Path, target: Path, *, drop: str | None) -> FeatureLibrary:
    """复制一份特征库，可选移除一个物体，返回已加载的内存库。"""
    shutil.copyfile(source, target)
    library = load_library(target)
    if drop and drop in library:
        library._objects.pop(drop, None)  # noqa: SLF001 - 证据脚本刻意构造教学前状态
        library.save()
    return library


def summarize_library(library: FeatureLibrary) -> dict[str, Any]:
    """库的对象数、条数与每个物体的条数。"""
    return {
        "objects": library.object_count,
        "vectors": library.total_vectors,
        "per_object": {row["name"]: row["count"] for row in library.stats()},
    }


def main() -> int:
    """运行全部证据步骤并写出 ``evidence.json``。"""
    # Windows 控制台默认 GBK，状态文案里有 ✅/📸 等字符会直接抛
    # UnicodeEncodeError；证据 JSON 始终写 UTF-8，这里只调整标准输出。
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(description="P5 实时识别 + 教学闭环证据")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / ".rn18" / "data",
        help="含 known/ test/ 与 features.json 的数据目录",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.cwd() / "p5-evidence",
        help="证据输出目录（不能是仓库内的受检目录，避免污染 git）",
    )
    parser.add_argument("--teach-object", default="mouse", help="教学前从副本库移除的物体")
    parser.add_argument("--teach-name", default="鼠标", help="预期说出的中文物体名")
    parser.add_argument("--frames", type=int, default=30, help="连续识别计时帧数")
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    features_json = data_dir / "features.json"
    if not features_json.is_file():
        print(f"[P5] 找不到特征库: {features_json}", file=sys.stderr)
        return 2

    print("[P5] 加载 ResNet18 编码器（权重应已在本地缓存）……")
    encoder = get_default_encoder()

    library = copy_library(
        features_json, out_dir / "library.json", drop=args.teach_object
    )
    before = summarize_library(library)
    print(f"[P5] 教学前库: {before['objects']} 个物体 / {before['vectors']} 条")

    spoken = install_recorder()
    controller = app.RealtimeController(
        library,
        encoder=encoder,
        gate=app.SpeechGate(app.TTS_COOLDOWN_SECONDS),
        # 证据要测"每帧处理耗时"，不是测 0.3 s 节流，所以关闭服务端跳帧。
        min_interval=0.0,
    )

    # ------------------------------------------------------------------ 判据 1
    known_names = [name for name in before["per_object"] if name != args.teach_object]
    known_frames: list[tuple[str, Path]] = []
    for name in known_names:
        for path in image_paths(data_dir, "test", name):
            known_frames.append((name, path))
    if not known_frames:
        print("[P5] 测试图片为空，无法计时", file=sys.stderr)
        return 2

    timing_rows: list[dict[str, Any]] = []
    saved_known_png: str | None = None
    cold_start_ms: float | None = None

    for index in range(args.frames):
        expected, path = known_frames[index % len(known_frames)]
        frame = load_rgb(path)
        started = time.perf_counter()
        outcome = controller.process(frame)
        wall_ms = (time.perf_counter() - started) * 1000.0
        if cold_start_ms is None:
            cold_start_ms = round(wall_ms, 1)
        result = controller.last_result
        timing_rows.append(
            {
                "frame": index + 1,
                "source": str(path.relative_to(data_dir)),
                "expected": expected,
                "name": result.name if result else None,
                "known": bool(result and result.is_known),
                "similarity": round(result.similarity, 4) if result else 0.0,
                "elapsed_ms": round(controller.last_elapsed_ms or 0.0, 1),
                "wall_ms": round(wall_ms, 1),
            }
        )
        if saved_known_png is None and outcome.frame is not None:
            saved_known_png = str(out_dir / "frame_known.png")
            Image.fromarray(outcome.frame).save(saved_known_png)

    warm = [row["elapsed_ms"] for row in timing_rows[1:]] or [timing_rows[0]["elapsed_ms"]]
    warm_sorted = sorted(warm)
    p95 = warm_sorted[min(len(warm_sorted) - 1, int(round(0.95 * (len(warm_sorted) - 1))))]
    timing = {
        "frames": len(timing_rows),
        "cold_start_ms": cold_start_ms,
        "mean_ms": round(float(np.mean(warm)), 1),
        "max_ms": round(float(np.max(warm)), 1),
        "p95_ms": round(float(p95), 1),
        "first_result_after_cold_start_ms": cold_start_ms,
        "within_3s": bool(cold_start_ms is not None and cold_start_ms < 3000),
        "rows": timing_rows[:8],
        "note": "elapsed_ms 为 recognize_stream 内部（特征提取 + 检索 + 标注）；"
        "wall_ms 含 Gradio 回调前的 Python 开销。摄像头 stream_every=0.3 s。",
    }

    # 未知物体：优先用"教学前被移除的真实物体测试图"；若占位阈值误判为已知，
    # 再补一张合成噪声图作为"必然低相似度"的未知画面，并如实记录两者分数。
    unknown_path = image_paths(data_dir, "test", args.teach_object)
    unknown_info: dict[str, Any] = {}
    unknown_png = out_dir / "frame_unknown.png"
    if unknown_path:
        unknown_frame = load_rgb(unknown_path[0])
        unknown_outcome = controller.process(unknown_frame)
        result = controller.last_result
        assert result is not None
        unknown_info = {
            "source": str(unknown_path[0].relative_to(data_dir)),
            "known": result.is_known,
            "name": result.name,
            "similarity": round(result.similarity, 4),
        }
        if unknown_outcome.frame is not None:
            Image.fromarray(unknown_outcome.frame).save(unknown_png)
    if unknown_info.get("known", True):
        rng = np.random.default_rng(20261109)
        noise = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
        noise_outcome = controller.process(noise)
        result = controller.last_result
        assert result is not None
        unknown_info["noise_fallback"] = {
            "known": result.is_known,
            "name": result.name,
            "similarity": round(result.similarity, 4),
        }
        if noise_outcome.frame is not None:
            Image.fromarray(noise_outcome.frame).save(unknown_png)
    unknown_info["screenshot"] = str(unknown_png)

    # TTS 冷却：同结果不播 -> 冷却内换结果不播 -> 冷却后换结果播。
    # 先 reset，保证这一段从"没有播报历史"开始，结论可复现。
    controller.gate.reset()
    by_name: dict[str, list[Path]] = {}
    for name, path in known_frames:
        by_name.setdefault(name, []).append(path)

    gate_trace: list[dict[str, Any]] = []

    def trace(name: str, started: float) -> None:
        outcome = controller.process(load_rgb(by_name[name][0]))
        gate_trace.append(
            {
                "object": name,
                "at_s": round(time.perf_counter() - started, 2),
                "spoken": outcome.gate_spoken,
            }
        )

    trace_started = time.perf_counter()
    trace("water_cup", trace_started)  # 首次：结果变化 + 无冷却 -> 播
    trace("water_cup", trace_started)  # 同结果 -> 不播
    trace("earphone", trace_started)  # 冷却内换结果 -> 不播
    time.sleep(4.2)
    trace("keyboard", trace_started)  # 冷却已过 + 换结果 -> 播
    trace("keyboard", trace_started)  # 同结果 -> 不播

    # 真实 Edge TTS 一条（证明"播报的是当次识别文本"，而不是占位音）。
    real_tts = synthesize_for_teaching("识别结果：键盘", audio_dir)

    # ------------------------------------------------------------------ 判据 2
    teach_frames = image_paths(data_dir, "known", args.teach_object)
    if len(teach_frames) < app.TEACHING_FRAME_COUNT:
        print(
            f"[P5] {args.teach_object} 的注册图不足 {app.TEACHING_FRAME_COUNT} 张",
            file=sys.stderr,
        )
        return 2

    before_teaching = summarize_library(library)
    spoken_before_teaching = len(spoken)
    begin_ok, begin_message = controller.begin_teaching(load_rgb(teach_frames[0]))
    paused_state = {
        "teaching": controller.teaching,
        "recognition_paused": controller.recognition_paused,
        "tts_paused": controller.tts_paused,
        "audio_cleared": controller.audio_path is None,
    }
    for path in teach_frames[1:]:
        time.sleep(0.3)
        controller.offer_frame(load_rgb(path))
    capture = {
        "requested": app.TEACHING_FRAME_COUNT,
        "captured": controller.captured_count,
        "intervals_s": controller.capture_intervals,
        "begin_ok": begin_ok,
        "begin_message": begin_message,
    }

    asr_result = {"ok": False, "error": "未执行"}
    tts_for_asr = synthesize_for_teaching(args.teach_name, audio_dir)
    if tts_for_asr.get("ok"):
        asr_result = transcribe_audio(Path(tts_for_asr["path"]), audio_dir)
    else:
        asr_result = {"ok": False, "error": f"TTS 合成失败: {tts_for_asr.get('error')}"}

    teaching_status = "未执行（ASR 失败）"
    teaching_retrieval: dict[str, Any] = {}
    after_teaching = summarize_library(library)
    gate_after_exit: dict[str, Any] = {}
    spoken_during_teaching: list[str] = []
    if asr_result.get("ok") and asr_result.get("text"):
        teaching_status, _ = controller.commit_teaching(asr_result["text"])
        after_teaching = summarize_library(library)
        gate_after_exit = {
            "last_text": controller.gate._last_text,  # noqa: SLF001
            "last_at": controller.gate._last_at,  # noqa: SLF001
        }
        # 只统计"教学窗口内"的播报调用；后面的测试图复识别不属于教学态。
        spoken_during_teaching = [item["text"] for item in spoken[spoken_before_teaching:]]
        # 用未参与教学的测试图再识别同一物体（严格分开注册/测试）。
        verify_paths = image_paths(data_dir, "test", args.teach_object)
        if verify_paths:
            verify_outcome = controller.process(load_rgb(verify_paths[0]))
            result = controller.last_result
            assert result is not None
            teaching_retrieval = {
                "query": str(verify_paths[0].relative_to(data_dir)),
                "name": result.name,
                "similarity": round(result.similarity, 4),
                "known": result.is_known,
                "speech_text": app.speech_text(result),
                "gate_spoken": verify_outcome.gate_spoken,
                "normalized_match": normalize(result.name)
                == normalize(asr_result["text"]),
                "screenshot": None,
            }
            if verify_outcome.frame is not None:
                screenshot = out_dir / "frame_after_teaching.png"
                Image.fromarray(verify_outcome.frame).save(screenshot)
                teaching_retrieval["screenshot"] = str(screenshot)
            if result.is_known:
                tts_after = synthesize_for_teaching(
                    f"识别结果：{result.name}", audio_dir
                )
                teaching_retrieval["tts_after_teaching"] = tts_after
    else:
        controller.cancel_teaching()

    teaching = {
        "object": args.teach_object,
        "expected_name": args.teach_name,
        "before": before_teaching,
        "after": after_teaching,
        "delta_vectors": after_teaching["vectors"] - before_teaching["vectors"],
        "capture": capture,
        "paused_state": paused_state,
        "asr": asr_result,
        "tts_for_asr": tts_for_asr,
        "commit_status": teaching_status,
        "retrieval": teaching_retrieval,
    }

    # ------------------------------------------------------------------ 判据 3
    # 恢复后冷却清零：commit_teaching / cancel 都会 SpeechGate.reset()。
    echo = {
        "during_teaching": paused_state,
        "gate_reset": {
            "cooldown_seconds": controller.gate.cooldown_seconds,
            "last_text_after_exit": gate_after_exit.get("last_text"),
            "last_at_after_exit": gate_after_exit.get("last_at"),
            "reset_means_last_at_is_negative_infinity": (
                gate_after_exit.get("last_at") == float("-inf")
            ),
        },
        "tts_calls_during_teaching": spoken_during_teaching,
        "note": "教学态 recognition_paused 与 tts_paused 同时为真，且 audio_path 被清空；"
        "退出时 SpeechGate.reset() 把 last_at 置为 -inf。",
    }

    # ------------------------------------------------------------------ 判据 4
    keyboard_library = copy_library(
        features_json, out_dir / "library_keyboard.json", drop=args.teach_object
    )
    keyboard_controller = app.RealtimeController(
        keyboard_library,
        encoder=encoder,
        gate=app.SpeechGate(app.TTS_COOLDOWN_SECONDS),
    )
    keyboard_before = summarize_library(keyboard_library)
    keyboard_controller.begin_teaching(load_rgb(teach_frames[0]))
    for path in teach_frames[1:]:
        time.sleep(0.3)
        keyboard_controller.offer_frame(load_rgb(path))
    asr_calls: list[object] = []
    original_asr = app.asr

    def counting_asr(path: object) -> str:
        asr_calls.append(path)
        return original_asr(path)

    app.asr = counting_asr  # type: ignore[assignment]
    try:
        keyboard_status, _ = keyboard_controller.commit_teaching("键盘兜底鼠标")
    finally:
        app.asr = original_asr  # type: ignore[assignment]
    keyboard_after = summarize_library(keyboard_library)
    keyboard_reloaded = load_library(keyboard_library.path)
    keyboard = {
        "before": keyboard_before,
        "after": keyboard_after,
        "delta_vectors": keyboard_after["vectors"] - keyboard_before["vectors"],
        "status": keyboard_status,
        "persisted_entries": len(
            keyboard_reloaded.get("键盘兜底鼠标").entries
            if keyboard_reloaded.get("键盘兜底鼠标")
            else []
        ),
        "asr_called": bool(asr_calls),
    }

    # ------------------------------------------------------------------ 判据 5
    cancel_library = copy_library(
        features_json, out_dir / "library_cancel.json", drop=args.teach_object
    )
    cancel_controller = app.RealtimeController(
        cancel_library,
        encoder=encoder,
        gate=app.SpeechGate(app.TTS_COOLDOWN_SECONDS),
    )
    cancel_before = summarize_library(cancel_library)
    cancel_hash_before = sha256_file(cancel_library.path)
    cancel_controller.begin_teaching(load_rgb(teach_frames[0]))
    for path in teach_frames[1:3]:
        time.sleep(0.3)
        cancel_controller.offer_frame(load_rgb(path))
    cancel_status, _, _ = cancel_controller.cancel_teaching()
    cancel_after = summarize_library(cancel_library)
    cancel_hash_after = sha256_file(cancel_library.path)
    cancel = {
        "status": cancel_status,
        "captured_before_cancel": 3,
        "before": cancel_before,
        "after": cancel_after,
        "hash_unchanged": cancel_hash_before == cancel_hash_after,
        "clean": cancel_before == cancel_after and cancel_hash_before == cancel_hash_after,
    }

    isolation_controller = app.RealtimeController(
        library, encoder=encoder, gate=app.SpeechGate(app.TTS_COOLDOWN_SECONDS)
    )
    bad = isolation_controller.process(out_dir / "missing-frame.jpg")
    tab2_headline, tab2_image, _ = app.recognize_image(
        image_paths(data_dir, "test", "water_cup")[0],
        library=library,
        encoder=encoder,
    )
    tab1_library = copy_library(
        features_json, out_dir / "library_isolation.json", drop=None
    )
    tab1_status, _, _ = app.enroll_images(
        "隔离测试",
        [str(teach_frames[0])],
        library=tab1_library,
        encoder=encoder,
    )
    isolation = {
        "bad_frame_headline": bad.headline,
        "bad_frame_did_not_raise": bad.frame is None,
        "tab2_still_recognizes": "识别结果" in tab2_headline,
        "tab1_still_enrolls": "已录入" in tab1_status,
        "tab1_entry_count": len(
            tab1_library.get("隔离测试").entries
            if tab1_library.get("隔离测试")
            else []
        ),
    }

    evidence = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": __import__("torch").__version__,
            "gradio": __import__("gradio").__version__,
            "numpy": np.__version__,
            "device": "cpu",
        },
        "data": {
            "data_dir": str(data_dir),
            "library_source": str(features_json),
            "teaching_object_removed": args.teach_object,
            "before": before,
        },
        "criterion_1_continuous": {
            "timing": timing,
            "unknown": unknown_info,
            "gate_trace": gate_trace,
            "real_tts": real_tts,
            "gate_rule": "结果变化 + 4 s 冷却（SpeechGate）",
        },
        "criterion_2_teaching": teaching,
        "criterion_3_echo": echo,
        "criterion_4_keyboard": keyboard,
        "criterion_5_isolation": isolation,
        "limitations": [
            "本机无摄像头与麦克风：帧源为真实测试图片组成的预置帧序列，"
            "教学抓帧为 5 张不同注册图按 0.3 s 间隔喂入。",
            "语音识别输入为 Edge TTS 合成音频（预录音频），不是真人录音；"
            "真人录音与现场麦克风需在小组电脑上复验。",
            "判定阈值仍是 P6 校准前的占位值 0.60，未知判断只证明链路可跑，"
            "不能当作阈值已校准的结论。",
        ],
    }

    evidence_path = out_dir / "evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[P5] 证据已写入: {evidence_path}")
    print(
        f"[P5] 连续识别 {timing['frames']} 帧：mean={timing['mean_ms']} ms, "
        f"max={timing['max_ms']} ms, cold={timing['cold_start_ms']} ms"
    )
    print(
        f"[P5] 教学：{before_teaching['objects']}/{before_teaching['vectors']} -> "
        f"{after_teaching['objects']}/{after_teaching['vectors']}"
    )
    print(f"[P5] ASR: {asr_result.get('text')!r}; commit: {teaching_status[:60]}")
    print(f"[P5] 取消干净: {cancel['clean']}; 键盘 +{keyboard['delta_vectors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
