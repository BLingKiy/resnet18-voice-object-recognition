"""百度短语音识别（ASR）封装。

依赖（全部延迟导入，只有真正调用 ``asr()`` 时才需要）::

    pip install soundfile numpy httpx

``soundfile`` 负责解码 wav/flac/ogg/mp3；重采样用 numpy 手写线性插值，
不额外引入 scipy。界面层只调用 ``is_asr_configured()`` 时不会触发这些依赖，
所以即使音频库没装，Tab1 也能正常启动、键盘录入不受影响。

设计要点：

1. **密钥不写死、不进日志**：只从进程环境变量或 ``.env`` 读；
   ``AsrConfig`` 的 repr 屏蔽两把 Key；异常信息只带错误码/异常类型，
   绝不拼接含 ``client_secret`` 的请求 URL。
2. **四级配置优先级（高 → 低）**：
   进程环境变量 → ``RN18_ENV_FILE`` 指向的文件 → ``~/.rn18/.env`` →
   仓库根目录 ``.env``。每个键独立按这个顺序取第一个非空值，
   因此“进程只覆盖 API Key、Secret 仍从文件读”是允许的。
3. **缺 Key 不崩溃**：``asr()`` 抛 ``AsrNotConfigured``（自定义异常），
   调用方捕获后置灰语音按钮并提示「未配置语音识别密钥」；键盘录入照常。
4. **token 缓存**：百度 access_token 约 30 天有效，这里记录过期时间并
   提前 60 秒刷新；遇到 HTTP 401 或百度 ``err_no`` 110/111（token 失效）
   时强制重取并重试一次。

接口选择：百度「短语音识别」标准版端点 ``server_api`` 配 ``dev_pid=1537``，
「极速版」端点 ``pro_api`` 只能用 ``dev_pid=80001``（两者不能混用）。
本项目契约固定了默认 ``dev_pid=1537``，所以这里按 dev_pid 选端点：
1537 → 标准版，80001 → 极速版。现场若要演示极速版，只改
``BAIDU_ASR_DEV_PID=80001`` 即可，不用改代码。
"""

from __future__ import annotations

import base64
import hashlib
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

# 与 feature 层的位置约定保持一致：voice/ 在仓库根目录下。
_REPO_ROOT = Path(__file__).resolve().parents[1]

ENV_API_KEY = "BAIDU_ASR_API_KEY"
ENV_SECRET_KEY = "BAIDU_ASR_SECRET_KEY"
ENV_DEV_PID = "BAIDU_ASR_DEV_PID"
ENV_FILE_VAR = "RN18_ENV_FILE"

DEFAULT_DEV_PID = 1537
PRO_DEV_PID = 80001
_TARGET_SAMPLE_RATE = 16000

TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
ASR_ENDPOINT_STANDARD = "https://vop.baidu.com/server_api"
ASR_ENDPOINT_PRO = "https://vop.baidu.com/pro_api"

_CUID = "rn18-demo"
_TOKEN_TIMEOUT_SECONDS = 10.0
_ASR_TIMEOUT_SECONDS = 30.0
_TOKEN_REFRESH_MARGIN_SECONDS = 60.0
_DEFAULT_TOKEN_TTL_SECONDS = 30 * 24 * 3600.0

# 百度这两个 err_no 表示 token 失效/过期，与 HTTP 401 一样需要重取。
_TOKEN_ERROR_CODES = frozenset({110, 111})


class AsrError(Exception):
    """语音识别相关错误的基类。"""


class AsrNotConfigured(AsrError):
    """缺少百度语音识别密钥。

    这是**可预期的状态**，不是崩溃：调用方据此置灰语音入口，
    键盘录入等其它功能保持可用。
    """


class AsrDependencyError(AsrError):
    """缺少 ASR 运行所需的第三方依赖（soundfile / numpy / httpx）。"""


class AsrAudioError(AsrError):
    """音频文件不存在、为空、无法解码或格式不支持。"""


class BaiduAsrError(AsrError):
    """百度接口层面的失败（网络、鉴权、识别错误码）。"""


@dataclass(frozen=True, slots=True)
class AsrConfig:
    """一次 ASR 调用所需的完整配置。

    ``api_key`` / ``secret_key`` 关闭了 repr 输出，避免 config 被日志打印时
    把密钥带出去（``repr()`` 只会显示 dev_pid）。
    """

    api_key: str = field(repr=False)
    secret_key: str = field(repr=False)
    dev_pid: int = DEFAULT_DEV_PID


def _strip_quotes(value: str) -> str:
    """去掉 .env 值两侧成对的单/双引号（内部辅助函数）。"""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _read_env_file(path: Path) -> dict[str, str]:
    """读取一个 ``KEY=VALUE`` 格式的 .env 文件。

    支持空行、``#`` 注释、``export`` 前缀和成对引号；同一文件重复键以最后
    一行为准（与 python-dotenv 一致）。读取失败（不存在/无权限/编码错误）
    时返回空字典而不抛异常——"缺 Key"统一由 :func:`load_asr_config` 返回
    ``None`` 来表达，保证界面启动路径永远不会因为一个坏文件崩掉。

    Args:
        path: .env 文件路径。

    Returns:
        键值字典；文件不可读时为空字典。
    """
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return {}

    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        if key:
            values[key] = _strip_quotes(value.strip())
    return values


def _default_env_paths() -> tuple[Path, Path]:
    """返回 ``(用户级 .env, 仓库根 .env)``。

    单独抽成函数是为了测试能把它指向临时目录，避免读到开发机上
    已经填了真实 Key 的 ``~/.rn18/.env``。
    """
    return Path.home() / ".rn18" / ".env", _REPO_ROOT / ".env"


def _parse_dev_pid(raw: str) -> int:
    """解析 dev_pid；空值、非数字或非正数一律回退到默认 1537。

    dev_pid 只影响识别模型选择，配置写错不值得让整个界面启动失败。
    """
    if not raw:
        return DEFAULT_DEV_PID
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_DEV_PID
    return value if value > 0 else DEFAULT_DEV_PID


def load_asr_config(
    *,
    environ: Mapping[str, str] | None = None,
    home: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> AsrConfig | None:
    """按四级优先级加载百度 ASR 配置。

    优先级从高到低：

    1. 进程环境变量 ``BAIDU_ASR_API_KEY`` / ``BAIDU_ASR_SECRET_KEY`` /
       ``BAIDU_ASR_DEV_PID``；
    2. ``RN18_ENV_FILE`` 指向的文件；
    3. ``~/.rn18/.env``；
    4. 仓库根目录 ``.env``。

    每个键独立取第一个非空值（逐键覆盖），而不是整层覆盖。

    Args:
        environ: 环境变量映射，默认 ``os.environ``；测试可注入。
        home: 用户主目录，默认 ``Path.home()``；测试可注入。
        repo_root: 仓库根目录，默认由本文件位置推导；测试可注入。

    Returns:
        两把 Key 都齐全时返回 :class:`AsrConfig`，否则返回 ``None``。
        只缺其中一把也返回 ``None``，因为百度鉴权必须成对使用。
    """
    env = os.environ if environ is None else environ

    layers: list[Mapping[str, str]] = [env]

    explicit_env_file = str(env.get(ENV_FILE_VAR, "")).strip()
    if explicit_env_file:
        layers.append(_read_env_file(Path(explicit_env_file).expanduser()))

    default_user_env, default_repo_env = _default_env_paths()
    if home is None:
        user_env = default_user_env
    else:
        user_env = Path(home).expanduser() / ".rn18" / ".env"
    if repo_root is None:
        repo_env = default_repo_env
    else:
        repo_env = Path(repo_root).expanduser() / ".env"
    layers.append(_read_env_file(user_env))
    layers.append(_read_env_file(repo_env))

    def first_non_empty(key: str) -> str:
        for layer in layers:
            value = layer.get(key)
            if value is not None:
                cleaned = str(value).strip()
                if cleaned:
                    return cleaned
        return ""

    api_key = first_non_empty(ENV_API_KEY)
    secret_key = first_non_empty(ENV_SECRET_KEY)
    if not api_key or not secret_key:
        return None

    return AsrConfig(
        api_key=api_key,
        secret_key=secret_key,
        dev_pid=_parse_dev_pid(first_non_empty(ENV_DEV_PID)),
    )


def is_asr_configured() -> bool:
    """是否已配置完整的百度 ASR 密钥。

    Tab1 用它决定语音按钮是否可点：False 时置灰并提示
    「未配置语音识别密钥」，不要调用 :func:`asr`。

    Returns:
        Key/Secret 都非空时为 True。
    """
    return load_asr_config() is not None


def _require_numpy() -> Any:
    """延迟导入 numpy；缺失时给出可执行的安装提示。"""
    try:
        import numpy as np
    except ImportError as exc:
        raise AsrDependencyError(
            "缺少依赖 numpy。请运行 `pip install numpy soundfile httpx`。"
        ) from exc
    return np


def to_16k_mono(
    samples: NDArray[np.floating] | list[float],
    sample_rate: int,
) -> NDArray[np.float32]:
    """把任意采样率/声道的波形转成 16 kHz 单声道 float32。

    多声道直接取平均（语音是各声道共有的主体，平均比只取左声道稳）。
    重采样用线性插值：实现简单、无额外依赖；语音能量集中在 4 kHz 以下，
    对百度 ASR 足够。若现场发现高频辅音识别率偏低，可换
    ``scipy.signal.resample_poly`` 或先让录音端固定 16 kHz。

    Args:
        samples: 形状 ``(N,)`` 或 ``(N, C)`` 的波形，建议浮点范围 [-1, 1]。
        sample_rate: 原始采样率（Hz）。

    Returns:
        float32 一维数组，长度约为 ``N * 16000 / sample_rate``。

    Raises:
        AsrAudioError: 波形为空、维度不支持或采样率非法。
        AsrDependencyError: 未安装 numpy。
    """
    np = _require_numpy()
    wave = np.asarray(samples, dtype=np.float32)
    if wave.ndim == 1:
        wave = wave[:, None]
    if wave.ndim != 2:
        raise AsrAudioError(f"音频波形应为 (N,) 或 (N, C)，收到 shape={wave.shape}")
    if wave.shape[0] == 0:
        raise AsrAudioError("音频波形为空")
    if sample_rate <= 0:
        raise AsrAudioError(f"采样率必须为正数，收到 {sample_rate}")

    mono = wave.mean(axis=1, dtype=np.float32)
    if sample_rate == _TARGET_SAMPLE_RATE:
        return mono

    duration = mono.shape[0] / float(sample_rate)
    target_length = max(1, round(duration * _TARGET_SAMPLE_RATE))
    source_positions = np.arange(target_length, dtype=np.float64) * (
        sample_rate / float(_TARGET_SAMPLE_RATE)
    )
    # 端点保护：浮点误差可能让最后一个索引略微越界。
    source_positions = np.clip(source_positions, 0.0, mono.shape[0] - 1.0)
    resampled = np.interp(
        source_positions,
        np.arange(mono.shape[0], dtype=np.float64),
        mono,
    )
    return resampled.astype(np.float32, copy=False)


def load_audio_16k_mono(audio_path: str | Path) -> NDArray[np.float32]:
    """读取本地音频文件并转成 16 kHz 单声道 float32。

    Args:
        audio_path: 本地音频路径；麦克风录音和预录音频文件走同一入口。

    Returns:
        float32 一维波形，采样率固定 16 kHz。

    Raises:
        AsrAudioError: 文件不存在、为空或 soundfile 无法解码。
        AsrDependencyError: 未安装 soundfile（或其动态库缺失）。
    """
    path = Path(audio_path).expanduser()
    if not path.is_file():
        raise AsrAudioError(f"音频文件不存在: {path}")

    try:
        import soundfile as sf
    except Exception as exc:  # ImportError，或 libsndfile 动态库缺失
        raise AsrDependencyError(
            "缺少音频解码依赖 soundfile（含 libsndfile）。请运行 "
            "`pip install soundfile numpy`；Windows 上若提示找不到 libsndfile，"
            "可用 `pip install --force-reinstall soundfile` 修复。"
        ) from exc

    try:
        data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:
        raise AsrAudioError(
            f"无法解码音频文件 {path.name}（{type(exc).__name__}）。"
            "请使用 wav/flac/ogg/mp3，或先用 ffmpeg 转成 16 kHz 单声道 wav。"
        ) from exc

    if data.size == 0:
        raise AsrAudioError(f"音频文件是空的: {path.name}")
    return to_16k_mono(data, int(sample_rate))


def _to_pcm16(mono: NDArray[np.float32]) -> bytes:
    """float32 [-1, 1] 波形转 16-bit 小端 PCM 字节流（百度要求的格式）。"""
    np = _require_numpy()
    clipped = np.clip(mono, -1.0, 1.0)
    # 显式小端：接口文档要求 16bit little-endian PCM，不依赖本机字节序。
    pcm = (clipped * 32767.0).astype(np.dtype("<i2"))
    return pcm.tobytes()


def _config_fingerprint(config: AsrConfig) -> str:
    """Key/Secret 的指纹，用于判断缓存 token 是否属于当前配置。"""
    digest = hashlib.sha256()
    digest.update(config.api_key.encode("utf-8"))
    digest.update(b"\0")
    digest.update(config.secret_key.encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _CachedToken:
    """进程内缓存的 access_token（token 本身不出现在 repr 里）。"""

    fingerprint: str
    token: str = field(repr=False)
    expires_at: float


_TOKEN_LOCK = threading.Lock()
_TOKEN_CACHE: _CachedToken | None = None


def _post_json(
    url: str,
    *,
    params: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
    timeout: float,
) -> Any:
    """POST 请求并返回 httpx.Response。

    网络异常统一转成 :class:`BaiduAsrError`，且**不拼接原始异常文本**：
    token 接口的 URL 查询串里带 ``client_secret``，一旦进入日志就是泄露。
    这里只保留异常类型名，足够定位是超时还是连接失败。

    Raises:
        AsrDependencyError: 未安装 httpx。
        BaiduAsrError: 网络层异常。
    """
    try:
        import httpx
    except ImportError as exc:
        raise AsrDependencyError(
            "缺少依赖 httpx。请运行 `pip install httpx`（Gradio 环境通常已自带）。"
        ) from exc

    try:
        return httpx.post(url, params=params, json=body, timeout=timeout)
    except httpx.HTTPError as exc:
        raise BaiduAsrError(
            f"请求百度语音服务失败（{type(exc).__name__}），请检查网络或代理设置。"
        ) from None


def _json_object(response: Any) -> dict[str, Any] | None:
    """尽力把响应解析成 JSON 对象；不是 JSON 时返回 None。"""
    try:
        payload = response.json()
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _fetch_access_token(config: AsrConfig) -> tuple[str, float]:
    """向百度 OAuth 接口申请新 token。

    Args:
        config: 当前配置。

    Returns:
        ``(access_token, expires_in_seconds)``。

    Raises:
        BaiduAsrError: HTTP 非 200、响应不是 JSON 或没有 access_token。
    """
    response = _post_json(
        TOKEN_URL,
        params={
            "grant_type": "client_credentials",
            "client_id": config.api_key,
            "client_secret": config.secret_key,
        },
        timeout=_TOKEN_TIMEOUT_SECONDS,
    )
    if response.status_code != 200:
        raise BaiduAsrError(
            f"获取百度 access_token 失败（HTTP {response.status_code}）。"
        )

    payload = _json_object(response)
    if payload is None:
        raise BaiduAsrError("获取百度 access_token 失败：响应不是合法 JSON。")

    token = payload.get("access_token")
    if not token:
        # error / error_description 是百度返回的错误码与描述，不含密钥。
        error = str(payload.get("error", "unknown")).strip()
        description = str(payload.get("error_description", "")).strip()
        raise BaiduAsrError(
            f"获取百度 access_token 失败：{error} {description}".strip()
        )

    ttl = payload.get("expires_in", _DEFAULT_TOKEN_TTL_SECONDS)
    try:
        expires_in = float(ttl)
    except (TypeError, ValueError):
        expires_in = _DEFAULT_TOKEN_TTL_SECONDS
    if expires_in <= 0:
        expires_in = _DEFAULT_TOKEN_TTL_SECONDS
    return str(token), expires_in


def _get_access_token(config: AsrConfig, *, force_refresh: bool = False) -> str:
    """取 access_token：命中未过期缓存就直接返回，否则重新申请。

    用 ``time.monotonic()`` 记过期时间，系统时钟被调整也不受影响；
    提前 60 秒过期，避免“请求发出时还有效、到达百度时刚过期”。
    ``force_refresh=True`` 用于 401 / err_no 110/111 后的重试。

    Args:
        config: 当前配置。
        force_refresh: 跳过缓存，强制重新申请。

    Returns:
        可用的 access_token 字符串。
    """
    # 函数内会重新赋值模块级缓存，必须显式声明 global，
    # 否则 Python 把它当局部变量，读取时直接 UnboundLocalError。
    global _TOKEN_CACHE

    fingerprint = _config_fingerprint(config)
    now = time.monotonic()

    with _TOKEN_LOCK:
        cached = _TOKEN_CACHE
        if (
            not force_refresh
            and cached is not None
            and cached.fingerprint == fingerprint
            and now < cached.expires_at
        ):
            return cached.token

    token, expires_in = _fetch_access_token(config)
    # 下限 30 秒：即使百度返回了异常小的 expires_in，也不会存一个立刻过期的值。
    expires_at = time.monotonic() + max(
        expires_in - _TOKEN_REFRESH_MARGIN_SECONDS,
        30.0,
    )
    with _TOKEN_LOCK:
        _TOKEN_CACHE = _CachedToken(
            fingerprint=fingerprint,
            token=token,
            expires_at=expires_at,
        )
    return token


def _asr_endpoint(dev_pid: int) -> str:
    """按 dev_pid 选择识别端点（极速版 80001 用 pro_api，其余用标准版）。"""
    return ASR_ENDPOINT_PRO if dev_pid == PRO_DEV_PID else ASR_ENDPOINT_STANDARD


def _recognize_pcm(pcm: bytes, config: AsrConfig) -> str:
    """把 16 kHz 单声道 PCM 发给百度，返回拼接后的识别文本。

    token 失效（HTTP 401 或 err_no 110/111）时强制刷新并重试一次；
    其它错误直接抛 :class:`BaiduAsrError`。

    Args:
        pcm: 16-bit 小端 PCM 字节流。
        config: 当前配置。

    Returns:
        识别文本（已 strip，可能为空字符串，例如静音）。
    """
    endpoint = _asr_endpoint(config.dev_pid)
    speech = base64.b64encode(pcm).decode("ascii")

    for attempt in (0, 1):
        token = _get_access_token(config, force_refresh=attempt > 0)
        response = _post_json(
            endpoint,
            body={
                "format": "pcm",
                "rate": _TARGET_SAMPLE_RATE,
                "channel": 1,
                "cuid": _CUID,
                "token": token,
                "dev_pid": config.dev_pid,
                "speech": speech,
                "len": len(pcm),
            },
            timeout=_ASR_TIMEOUT_SECONDS,
        )
        payload = _json_object(response)
        err_no = payload.get("err_no") if payload is not None else None

        if response.status_code == 401 or err_no in _TOKEN_ERROR_CODES:
            if attempt == 0:
                continue  # 缓存 token 失效：下一轮 force_refresh=True
            raise BaiduAsrError(
                "百度语音识别鉴权失败：token 刷新后仍被拒绝，"
                "请确认应用已开通语音识别权限。"
            )

        if response.status_code != 200:
            raise BaiduAsrError(
                f"百度语音识别请求失败（HTTP {response.status_code}）。"
            )
        if payload is None:
            raise BaiduAsrError("百度语音识别返回了无法解析的响应。")

        if err_no != 0:
            err_msg = str(payload.get("err_msg", "")).strip()
            raise BaiduAsrError(f"百度语音识别失败（err_no={err_no}: {err_msg}）")

        results = payload.get("result") or []
        return "".join(str(item) for item in results).strip()

    # 循环内已覆盖所有分支，这行只为让类型检查器看到明确的返回路径。
    raise BaiduAsrError("百度语音识别失败：token 重试流程异常结束。")


def asr(audio_path: str | Path) -> str:
    """识别一个本地音频文件中的语音。

    麦克风录音和预录音频文件走同一入口：调用方把录音落盘成 wav 后传路径
    即可，不需要额外的“文件识别”函数。

    Args:
        audio_path: 任意本地音频路径（wav/flac/ogg/mp3）。

    Returns:
        识别文本（已 ``strip()``）；静音时可能为空字符串。

    Raises:
        AsrNotConfigured: 未配置百度 Key。调用方应提示「未配置语音识别密钥」，
            不要把它当成程序崩溃。
        AsrDependencyError: 缺少 soundfile/numpy/httpx。
        AsrAudioError: 文件不存在、为空或无法解码。
        BaiduAsrError: 网络错误、鉴权失败或百度返回错误码。
    """
    config = load_asr_config()
    if config is None:
        raise AsrNotConfigured(
            "未配置语音识别密钥：请设置 BAIDU_ASR_API_KEY 与 BAIDU_ASR_SECRET_KEY"
            "（可写入 RN18_ENV_FILE 指向的文件、~/.rn18/.env 或仓库根目录 .env，"
            "参考 .env.example）。键盘录入不受影响。"
        )

    mono = load_audio_16k_mono(audio_path)
    return _recognize_pcm(_to_pcm16(mono), config)
