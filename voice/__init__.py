"""语音交互层（百度短语音识别 + Edge TTS 播报）。

对外契约（界面层只需依赖这些名字）::

    from voice import AsrNotConfigured, asr, is_asr_configured
    from voice import TtsError, TtsThrottler, announce_stream, tts

    if is_asr_configured():
        text = asr("record.wav")        # 麦克风录音与预录音频走同一入口
        name_box.value = text           # 回填名称框，用户可编辑后入库
    else:
        # 置灰语音按钮并提示「未配置语音识别密钥」；键盘录入照常
        ...

    path = tts("识别结果：苹果")          # 静态链路：每次识别后同步合成
    announce_stream("识别结果：鼠标")     # 实时链路：结果变化 + 冷却才播

注意：包属性 ``voice.asr`` / ``voice.tts`` 是函数（本模块导入时覆盖了同名
子模块），需要访问子模块内部辅助函数时请用
``importlib.import_module("voice.asr")`` / ``importlib.import_module("voice.tts")``。
"""

from .asr import (
    DEFAULT_DEV_PID,
    AsrAudioError,
    AsrConfig,
    AsrDependencyError,
    AsrError,
    AsrNotConfigured,
    BaiduAsrError,
    asr,
    is_asr_configured,
    load_asr_config,
    load_audio_16k_mono,
    to_16k_mono,
)
from .tts import (
    DEFAULT_VOICE,
    TTS_COOLDOWN_SECONDS,
    TTS_UNAVAILABLE_MESSAGE,
    TtsAudioError,
    TtsDependencyError,
    TtsError,
    TtsResult,
    TtsSynthesisError,
    TtsThrottler,
    announce_stream,
    get_throttler,
    synthesize,
    tts,
)

__all__ = [
    "DEFAULT_DEV_PID",
    "DEFAULT_VOICE",
    "TTS_COOLDOWN_SECONDS",
    "TTS_UNAVAILABLE_MESSAGE",
    "AsrAudioError",
    "AsrConfig",
    "AsrDependencyError",
    "AsrError",
    "AsrNotConfigured",
    "BaiduAsrError",
    "TtsAudioError",
    "TtsDependencyError",
    "TtsError",
    "TtsResult",
    "TtsSynthesisError",
    "TtsThrottler",
    "announce_stream",
    "asr",
    "get_throttler",
    "is_asr_configured",
    "load_asr_config",
    "load_audio_16k_mono",
    "synthesize",
    "to_16k_mono",
    "tts",
]
