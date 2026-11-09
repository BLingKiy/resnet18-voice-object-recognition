"""ASR 配置链路离线自测：全部使用假值，绝不请求百度接口。

覆盖需求里的四条证据：

1. ``RN18_ENV_FILE`` 指向的文件会被读取；
2. 进程环境变量覆盖文件值（逐键覆盖）；
3. 都没配置时 ``is_asr_configured()`` 为 False，``asr()`` 抛
   ``AsrNotConfigured`` 而不是崩溃；
4. ``git check-ignore .env`` 有输出（``.env`` 不会入库）。

另附：四级链路的层级顺序、空值处理、``.env.example`` 不含真值、
以及 44.1 kHz 立体声重采样到 16 kHz 单声道。
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from voice import AsrNotConfigured, asr, is_asr_configured, load_asr_config

# 注意：voice/__init__.py 把包属性 asr 绑定成了函数，必须用 importlib
# 才能拿到子模块本身（monkeypatch 内部函数时要用）。
asr_module = importlib.import_module("voice.asr")

REPO_ROOT = Path(__file__).resolve().parents[1]

# 明显的假值：既能证明链路生效，又不会被误认为真实 Key。
FAKE_FILE_KEY = "fake-file-api-key"
FAKE_FILE_SECRET = "fake-file-secret-key"
FAKE_PROCESS_KEY = "fake-process-api-key"


@pytest.fixture()
def work_dir() -> Iterator[Path]:
    """工作区内的临时目录。

    不用 pytest 的 ``tmp_path``：它在 Windows 上以 0o700 建目录，
    受限令牌（沙箱/AppContainer）拿不到访问权，测试会直接 error。
    这里用默认权限 ``mkdir()``，跑完清理。
    """
    base = REPO_ROOT / ".pytest-tmp"
    base.mkdir(exist_ok=True)
    path = base / f"asr-selftest-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(autouse=True)
def _isolate_asr_layers(monkeypatch: pytest.MonkeyPatch, work_dir: Path) -> None:
    """隔离真实环境，保证测试不依赖开发机上的 Key。

    真实机器上 ``~/.rn18/.env`` 可能已经填了值，必须把它指向临时空路径，
    否则“未配置”用例会被环境带偏，甚至把真实 Key 读进测试进程。
    """
    for name in (
        "BAIDU_ASR_API_KEY",
        "BAIDU_ASR_SECRET_KEY",
        "BAIDU_ASR_DEV_PID",
        "RN18_ENV_FILE",
    ):
        monkeypatch.delenv(name, raising=False)

    empty_user_env = work_dir / "isolated-home" / ".rn18" / ".env"
    empty_repo_env = work_dir / "isolated-repo" / ".env"
    monkeypatch.setattr(
        asr_module,
        "_default_env_paths",
        lambda: (empty_user_env, empty_repo_env),
    )


def _write_env(path: Path, **values: str) -> Path:
    """把键值写成 .env 格式，返回路径。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{key}={value}\n" for key, value in values.items()),
        encoding="utf-8",
    )
    return path


def test_rn18_env_file_is_loaded(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RN18_ENV_FILE 指向的文件应被读取（优先级 2）。"""
    env_file = _write_env(
        work_dir / "custom.env",
        BAIDU_ASR_API_KEY=FAKE_FILE_KEY,
        BAIDU_ASR_SECRET_KEY=FAKE_FILE_SECRET,
    )
    monkeypatch.setenv("RN18_ENV_FILE", str(env_file))

    config = load_asr_config()

    assert config is not None
    assert config.api_key == FAKE_FILE_KEY
    assert config.secret_key == FAKE_FILE_SECRET
    assert config.dev_pid == 1537
    assert is_asr_configured() is True


def test_process_env_overrides_file(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """进程环境变量优先于文件；未覆盖的 Secret 仍从文件补齐（优先级 1）。"""
    env_file = _write_env(
        work_dir / "custom.env",
        BAIDU_ASR_API_KEY=FAKE_FILE_KEY,
        BAIDU_ASR_SECRET_KEY=FAKE_FILE_SECRET,
    )
    monkeypatch.setenv("RN18_ENV_FILE", str(env_file))
    monkeypatch.setenv("BAIDU_ASR_API_KEY", FAKE_PROCESS_KEY)

    config = load_asr_config()

    assert config is not None
    assert config.api_key == FAKE_PROCESS_KEY
    assert config.api_key != FAKE_FILE_KEY
    assert config.secret_key == FAKE_FILE_SECRET


def test_env_file_precedence_and_home_vs_repo(work_dir: Path) -> None:
    """RN18_ENV_FILE > ~/.rn18/.env > 仓库 .env。"""
    home = work_dir / "home"
    repo = work_dir / "repo"
    repo.mkdir()
    _write_env(
        home / ".rn18" / ".env",
        BAIDU_ASR_API_KEY="fake-home-key",
        BAIDU_ASR_SECRET_KEY="fake-home-secret",
    )
    _write_env(
        repo / ".env",
        BAIDU_ASR_API_KEY="fake-repo-key",
        BAIDU_ASR_SECRET_KEY="fake-repo-secret",
    )

    config = load_asr_config(home=home, repo_root=repo)
    assert config is not None
    assert (config.api_key, config.secret_key) == ("fake-home-key", "fake-home-secret")

    explicit = _write_env(
        work_dir / "explicit.env",
        BAIDU_ASR_API_KEY="fake-explicit-key",
        BAIDU_ASR_SECRET_KEY="fake-explicit-secret",
    )
    config = load_asr_config(
        environ={"RN18_ENV_FILE": str(explicit)},
        home=home,
        repo_root=repo,
    )
    assert config is not None
    assert (config.api_key, config.secret_key) == (
        "fake-explicit-key",
        "fake-explicit-secret",
    )


def test_blank_values_are_treated_as_missing(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """空值等于没填（.env.example 的原样状态不能算已配置）。"""
    env_file = _write_env(
        work_dir / "empty.env",
        BAIDU_ASR_API_KEY="",
        BAIDU_ASR_SECRET_KEY="",
    )
    monkeypatch.setenv("RN18_ENV_FILE", str(env_file))

    assert load_asr_config() is None
    assert is_asr_configured() is False


def test_unconfigured_asr_raises_custom_exception(work_dir: Path) -> None:
    """未配置时 asr() 抛 AsrNotConfigured，且在检查音频之前就抛出。"""
    assert is_asr_configured() is False

    with pytest.raises(AsrNotConfigured, match="未配置语音识别密钥"):
        asr(work_dir / "not-exist.wav")


def test_config_repr_does_not_leak_keys(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """配置对象的 repr 不得带出 Key/Secret，防止日志泄露。"""
    env_file = _write_env(
        work_dir / "custom.env",
        BAIDU_ASR_API_KEY=FAKE_FILE_KEY,
        BAIDU_ASR_SECRET_KEY=FAKE_FILE_SECRET,
    )
    monkeypatch.setenv("RN18_ENV_FILE", str(env_file))

    config = load_asr_config()

    assert config is not None
    rendered = repr(config)
    assert FAKE_FILE_KEY not in rendered
    assert FAKE_FILE_SECRET not in rendered
    assert "dev_pid=1537" in rendered


def test_dot_env_is_git_ignored() -> None:
    """git check-ignore .env 必须有输出（退出码 0）。"""
    git = shutil.which("git")
    if git is None:
        pytest.skip("本机没有 git，无法验证 .gitignore")

    inside = subprocess.run(
        [git, "rev-parse", "--is-inside-work-tree"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        pytest.skip("当前目录不在 git 工作区内（例如解压出来的发布包），无法验证 .gitignore")

    result = subprocess.run(
        [git, "check-ignore", ".env"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert result.returncode == 0, f"git check-ignore 未命中: {result.stderr}"
    assert ".env" in result.stdout.splitlines()


def test_env_example_has_only_placeholders() -> None:
    """.env.example 只能有键名与空值/默认 dev_pid，不能有真实 Key。"""
    text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")

    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        assert separator, f".env.example 出现非法行: {line!r}"
        values[key.strip()] = value.strip()

    assert values["BAIDU_ASR_API_KEY"] == ""
    assert values["BAIDU_ASR_SECRET_KEY"] == ""
    assert values["BAIDU_ASR_DEV_PID"] == "1537"


def test_to_16k_mono_resamples_stereo() -> None:
    """44.1 kHz 立体声应转成 16000 个点的 16 kHz 单声道。"""
    np = pytest.importorskip("numpy")
    source_rate = 44100
    timeline = np.arange(source_rate, dtype=np.float32) / source_rate
    tone = np.sin(2 * np.pi * 440.0 * timeline).astype(np.float32)
    stereo = np.stack([tone, tone], axis=1)

    mono = asr_module.to_16k_mono(stereo, source_rate)

    assert mono.ndim == 1
    assert mono.dtype == np.float32
    assert mono.shape == (16000,)
    # 两个声道相同，取平均后仍是幅度 1 的正弦波。
    assert float(np.max(np.abs(mono))) == pytest.approx(1.0, abs=0.02)


def test_load_audio_file_resamples_and_packs_pcm(work_dir: Path) -> None:
    """真实文件路径：soundfile 解码 -> 16k 单声道 -> 16bit PCM 字节流。"""
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")

    source_rate = 44100
    timeline = np.arange(source_rate, dtype=np.float32) / source_rate
    tone = np.sin(2 * np.pi * 440.0 * timeline).astype(np.float32)
    stereo = np.stack([tone, tone * 0.5], axis=1)
    audio_path = work_dir / "probe-44k-stereo.wav"
    sf.write(str(audio_path), stereo, source_rate, subtype="PCM_16")

    mono = asr_module.load_audio_16k_mono(audio_path)
    pcm = asr_module._to_pcm16(mono)

    assert mono.shape == (16000,)
    assert mono.dtype == np.float32
    assert len(pcm) == 16000 * 2  # 16bit = 2 字节/采样点
    assert float(np.max(np.abs(mono))) > 0.1


class _FakeResponse:
    """httpx.Response 的离线替身（只用到 status_code 与 json()）。"""

    def __init__(self, status_code: int, payload: dict[str, object]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, object]:
        return self._payload


def test_access_token_cache_and_force_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一配置命中缓存；force_refresh=True 才重新申请（不发网络请求）。"""
    config = asr_module.AsrConfig(api_key="fake-api", secret_key="fake-secret")
    issued: list[str] = []

    def fake_fetch(cfg: asr_module.AsrConfig) -> tuple[str, float]:
        issued.append(cfg.api_key)
        return f"fake-token-{len(issued)}", 3600.0

    monkeypatch.setattr(asr_module, "_fetch_access_token", fake_fetch)
    monkeypatch.setattr(asr_module, "_TOKEN_CACHE", None)

    first = asr_module._get_access_token(config)
    second = asr_module._get_access_token(config)
    refreshed = asr_module._get_access_token(config, force_refresh=True)

    assert first == second == "fake-token-1"
    assert refreshed == "fake-token-2"
    assert issued == ["fake-api", "fake-api"]


def test_401_triggers_token_refresh_and_retries_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """第一次 401 时刷新 token 并重试，第二次成功（不请求百度）。"""
    config = asr_module.AsrConfig(api_key="fake-api", secret_key="fake-secret")
    token_calls: list[bool] = []

    def fake_token(cfg: asr_module.AsrConfig, *, force_refresh: bool = False) -> str:
        token_calls.append(force_refresh)
        return f"fake-token-{len(token_calls)}"

    responses = [
        _FakeResponse(401, {}),
        _FakeResponse(200, {"err_no": 0, "result": ["苹果"]}),
    ]

    def fake_post(
        url: str,
        *,
        params: dict[str, str] | None = None,
        body: dict[str, object] | None = None,
        timeout: float,
    ) -> _FakeResponse:
        return responses.pop(0)

    monkeypatch.setattr(asr_module, "_get_access_token", fake_token)
    monkeypatch.setattr(asr_module, "_post_json", fake_post)

    text = asr_module._recognize_pcm(b"\x00\x00", config)

    assert text == "苹果"
    assert token_calls == [False, True]  # 第二次是强制刷新
    assert responses == []
