"""语音交互层：百度短语音识别（输入）+ Edge TTS（播报）。

对外契约（界面层只需依赖这些名字）::

    from voice import AsrNotConfigured, asr, is_asr_configured

    if is_asr_configured():
        text = asr("record.wav")        # 麦克风录音与预录音频走同一入口
        name_box.value = text           # 回填名称框，用户可编辑后入库
    else:
        # 置灰语音按钮并提示「未配置语音识别密钥」；键盘录入照常
        ...

    from voice import TtsError, synthesize, tts

    path = tts("识别结果：苹果")        # 每次调用生成一个新 mp3，返回路径字符串

注意：包属性 ``voice.asr`` 与 ``voice.tts`` 都是函数（本模块导入时覆盖了
同名子模块），需要访问子模块内部辅助函数时请用
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
    DEFAULT_RATE,
    DEFAULT_VOICE,
    DEFAULT_VOLUME,
    SpeechGate,
    TtsConfig,
    TtsDependencyError,
    TtsError,
    TtsSynthesisError,
    speak_in_background,
    synthesize,
    tts,
)

__all__ = [
    "DEFAULT_DEV_PID",
    "DEFAULT_RATE",
    "DEFAULT_VOICE",
    "DEFAULT_VOLUME",
    "AsrAudioError",
    "AsrConfig",
    "AsrDependencyError",
    "AsrError",
    "AsrNotConfigured",
    "BaiduAsrError",
    "SpeechGate",
    "TtsConfig",
    "TtsDependencyError",
    "TtsError",
    "TtsSynthesisError",
    "asr",
    "is_asr_configured",
    "load_asr_config",
    "load_audio_16k_mono",
    "speak_in_background",
    "synthesize",
    "to_16k_mono",
    "tts",
]
