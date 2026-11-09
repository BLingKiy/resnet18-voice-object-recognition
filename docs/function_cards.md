# 关键函数讲解卡（教师随机抽问用）

> 每个函数给出：**文件路径 + 实际函数签名（参数名、默认值、返回类型）**、输入输出、设计理由、被抽问时怎么答。签名逐字对照 `core/features.py`、`core/library.py`、`core/recognize.py`、`core/pipeline.py`、`voice/asr.py`、`voice/tts.py`。
> 阈值口径：已校准值 **`0.8124`** 存在特征库 JSON 的 `threshold` 字段，程序优先用它；`core/recognize.py` 里的 `DEFAULT_THRESHOLD = 0.60` 只是"库未校准"时的兜底，不是本项目结论。

---

## 1. `core.features.extract_feature`

**文件**：`core/features.py`

```python
def extract_feature(
    image: ImageInput,
    *,
    encoder: Encoder | None = None,
) -> NDArray[np.float32]:
```

- **输入**：单张图片；`ImageInput = str | Path | PIL.Image | np.ndarray`（路径 / PIL / numpy）。**输出**：`(512,)` 的 float32 向量，L2 范数 = 1。
- **为什么这样设计**：去掉 ResNet18 的 `fc` 层，只取 512 维全局平均池化特征；输出统一 L2 归一化，使余弦相似度退化为点积，检索变成一次矩阵乘法。torch/torchvision 延迟导入，纯逻辑测试不用加载 1100 万参数模型。
- **被抽问怎么答**：为什么是 512 维？——ResNet18 去掉分类头后的池化输出就是 512 维。为什么要归一化？——归一化后相似度 = 点积，且模长不再影响阈值，阈值能跨图片复用。

---

## 2. `core.library.add_to_library`

**文件**：`core/library.py`

```python
def add_to_library(
    library: FeatureLibrary,
    name: str,
    images: Iterable[ImageInput],
    *,
    labels: Sequence[str] | None = None,
    encoder: Encoder | None = None,
) -> AddResult:
```

- **输入**：特征库实例、物体名称、图片可迭代对象（路径 / PIL / numpy），可选标签与编码器。**输出**：`AddResult(object_name, added, skipped, object_vectors, total_vectors)`。
- **为什么这样设计**：录入与落盘分离——"录入 5 张图只写一次盘"，也避免测试/误操作污染真实特征库。同一物体内按**解码后 RGB 像素的 SHA-256** 去重，保证"向量条数 = 有效注册图数"。
- **被抽问怎么答**：为什么不自动保存？——显式 `save_library` 让多次录入合并为一次写盘，原子替换保证断电不损坏 JSON。重复图怎么处理？——像素摘要相同就跳过，不虚增条数。Tab3 教学抓帧为什么也调它？——教学态复用同一个入库函数，不另写一套写库逻辑。

---

## 3. `core.recognize.retrieve`

**文件**：`core/recognize.py`

```python
def retrieve(
    library: FeatureLibrary,
    image: ImageInput | NDArray[np.floating],
    *,
    top_k: int = 5,
    encoder: Encoder | None = None,
) -> list[Match]:
```

- **输入**：特征库、查询图片或已提取的一维向量、`top_k`。**输出**：按相似度降序的 `Match` 列表（`name`、`image`、`similarity`、`rank`）；空库返回空列表。
- **为什么这样设计**：`matrix @ query` 一次算出全库相似度，`argsort` 取前 k；空库直接返回空，不抛异常。供界面展示前 3 个近邻。
- **被抽问怎么答**：相似度怎么算？——`library.matrix()` 拼出 `(N,512)` 矩阵，与查询向量点积（已 L2 归一化，点积即余弦）。返回图片级还是物体级？——图片级；物体级聚合在 `recognize` 里做。

---

## 4. `core.recognize.judge_unknown`

**文件**：`core/recognize.py`

```python
def judge_unknown(similarity: float, threshold: float = DEFAULT_THRESHOLD) -> bool:
# DEFAULT_THRESHOLD = 0.60 只是兜底；实际生效值由 resolve_threshold 从库里的 0.8124 取出
```

- **输入**：最高余弦相似度、阈值（默认兜底 `DEFAULT_THRESHOLD = 0.60`）。**输出**：`True` 表示应判为未知物体。
- **为什么这样设计**：判定边界偏向"认出来"——`similarity < threshold` 才判未知，**等于阈值算已知**，避免边界抖动时结果来回跳。非有限值（NaN/Inf）一律判未知。
- **被抽问怎么答**：阈值现在多少？——**0.8124**，写在 `features.json` 里，是先跑 `calibrate.py` 得到两条分布后取的中点；代码里的 `0.60` 只是库没校准时才用的兜底。为什么用 `<` 不用 `<=`？——等于阈值算已知，边界更稳。0.8124 怎么来的？——已知类测试图最低 0.8828，未知样本最高 0.7421，取中点，两侧各留 0.07 余量。

---

## 5. `core.library.load_library`

**文件**：`core/library.py`

```python
def load_library(
    path: str | Path | None = None,
    *,
    create: bool = True,
) -> FeatureLibrary:
```

- **输入**：JSON 路径（默认仓库内 `data/features.json`，可用 `RN18_LIBRARY_PATH` 覆盖）、`create` 开关。**输出**：校验通过的 `FeatureLibrary`，已绑定文件绝对路径。
- **为什么这样设计**：启动时加载 → 这就是"重启后特征库仍在"的实现。严格校验 schema / 版本 / 维度 / 向量范数；`create=False` 时文件不存在抛 `FileNotFoundError`，加载失败抛 `ValueError`，且不静默覆盖磁盘文件。
- **被抽问怎么答**：重启后数据还在吗？——在，`features.json` 落盘，启动时 `load_library` 读回来。文件坏了会怎样？——抛 `ValueError`，不覆盖原文件。阈值也从这里读吗？——是，`library.threshold` 就是 0.8124。

---

## 6. `core.library.save_library`

**文件**：`core/library.py`

```python
def save_library(
    library: FeatureLibrary,
    path: str | Path | None = None,
) -> Path:
```

- **输入**：特征库实例、可选路径（省略时用加载/构造时绑定的路径）。**输出**：实际写入的绝对路径。
- **为什么这样设计**：先写同目录临时文件、`fsync`、再 `os.replace` 原子替换目标——演示现场程序被强杀也不会留下半个 JSON 把特征库彻底弄坏。向量保留 8 位小数，文件约为全精度 1/3，余弦误差 < 1e-7，加载时再归一化修正舍入。
- **被抽问怎么答**：为什么不直接写目标文件？——原子替换保证要么旧文件完整、要么新文件完整。向量保留几位？——8 位小数，加载时重新归一化。阈值也跟着存吗？——存，`threshold` 字段和向量在同一个 JSON 里，一次原子写。

---

## 7. `core.pipeline.recognize_static`（静态图片识别链路入口）

**文件**：`core/pipeline.py`

```python
def recognize_static(
    image: ImageInput,
    library: FeatureLibrary,
    *,
    threshold: float | None = None,
    top_k: int = 3,
    encoder: Encoder | None = None,
) -> Recognition:
```

- **输入**：图片（路径 / PIL / numpy）、特征库、可选阈值与 `top_k`（界面默认 3）。**输出**：`Recognition`（`is_known`、`name`、`similarity`、`threshold`、`matches`、`object_scores`、`similarity_percent`）；未知时 `name="未知物体"`。
- **为什么这样设计**：静态链路（Tab2 / 命令行）与实时链路（Tab3）共用同一个核心，`recognize_static` 只做"一次特征提取 + 全库检索 + 判未知"，把界面关注的东西（标注、计时、播报文本）留在 `core/pipeline.py` 的其他函数里，避免复制一套检索逻辑。
- **被抽问怎么答**：`recognize_static` 和 `core.recognize.recognize` 什么关系？——它就是 `recognize` 的一层薄封装，固定了界面用的参数语义；真正做检索和判未知的是 `recognize`。实时链路怎么复用？——`recognize_stream` 内部同样调 `recognize`，只是 `top_k=1` 并额外画标注、计时。

---

## 8. `voice.asr.asr`

**文件**：`voice/asr.py`

```python
def asr(audio_path: str | Path) -> str:
```

- **输入**：本地音频路径（wav / flac / ogg / mp3）。**输出**：识别文本（已 `strip()`）；静音时可能为空字符串。
- **为什么这样设计**：麦克风录音与预录音频走同一入口（录音落盘成 wav 后传路径）。密钥缺失时抛 `AsrNotConfigured`（**可预期的状态，不是崩溃**），调用方据此处理；Tab1 与 Tab3 教学态已接线：录音 → 回填名称；未配 Key 时按钮置灰。token 缓存约 30 天、提前 60 秒刷新；HTTP 401 或 `err_no` 110/111 强制重取并重试一次。依赖延迟导入，未装不影响界面启动。
- **被抽问怎么答**：缺 Key 会崩吗？——不会，`asr()` 抛 `AsrNotConfigured`，可捕获；键盘录入不受影响。密钥会进日志吗？——不会，`AsrConfig` 的 repr 屏蔽两把 Key，异常只带错误码。极速版怎么切？——`BAIDU_ASR_DEV_PID=80001`，端点随 `dev_pid` 自动切换。

---

## 9. `voice.tts.tts`（Edge TTS 语音播报）

**文件**：`voice/tts.py`

```python
def tts(
    text: str,
    *,
    config: TtsConfig | None = None,
    out_dir: str | Path | None = None,
) -> str:
```

- **输入**：待播报文本、可选音色配置（默认 `zh-CN-XiaoxiaoNeural`）、可选输出目录。**输出**：合成好的 mp3 **绝对路径字符串**，可直接塞给 `gr.Audio`。
- **配套函数**：`speak_in_background(text, *, on_success=None, on_error=None, ...) -> threading.Thread` 在后台线程合成（一次合成要 1–2 秒，实时链路不能同步等）；`SpeechGate(cooldown_seconds=4.0)` 是实时链路的节流器，规则是「结果变化 **且** 冷却已过」才播。
- **为什么这样设计**：合成与播放放后台线程，避免网络请求阻塞 Gradio 事件；失败抛 `TtsError` 子类（`TtsDependencyError` / `TtsSynthesisError`），界面捕获后显示「语音播报不可用，请检查网络」，识别结果与特征库不受影响。播报文本由 `core.pipeline.speech_text` 统一生成（已知报「识别结果：X」，未知报「未知物体」），保证界面显示与语音说法一致。
- **被抽问怎么答**：为什么播报要放后台线程？——合成是网络请求，1–2 秒，放主线程会把实时画面卡住。实时模式会不会一直播同一句？——不会，`SpeechGate` 要求文本变化 + 4 秒冷却都满足。断网会怎样？——抛 `TtsError`，界面显示不可用提示，不崩、不影响识别。

---

## 抽问高频「为什么」速查

| 问题 | 一句话答案 |
|------|------------|
| 为什么用 ResNet18 去掉 fc？ | 取 512 维通用视觉特征，才能"以图搜图"自定义物体 |
| 为什么 L2 归一化？ | 余弦相似度变点积，检索一次矩阵乘，阈值不受模长影响 |
| 为什么 JSON 不用数据库？ | 重启不丢、零依赖、约 300KB，SQLite/FAISS 属扩张不加分 |
| 为什么按物体取最大相似度？ | 多视角注册时"有一张很像就算像"，对多视角更友好 |
| 阈值为什么是 0.8124？ | 已知测试图最低 0.8828、未知样本最高 0.7421，取中点，两侧各留 0.07 |
| 那代码里的 0.60 是什么？ | 只是库未校准时的兜底，本项目的结论是 0.8124 |
| 校准集用了注册图吗？ | 没有，只用 test/ 与 unknown/，三类 md5 两两零重叠 |
| 充电头为什么没被误判？ | 最高 0.7421，低于阈值 0.8124，比已知最低分还低 0.14 |
| 缺语音 Key 会崩吗？ | 不会：按钮置灰且不调用 `asr()`，键盘录入照常 |
| 播报失败会影响识别吗？ | 不会：TTS 只降级提示，识别结果与特征库不受影响 |
