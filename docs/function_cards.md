# 关键函数讲解卡（教师随机抽问用）

> 每个函数都给出：**文件路径 + 实际函数签名（参数名、默认值、返回类型）**、输入输出、设计理由、被抽问时怎么答。签名逐字对照 `core/features.py`、`core/library.py`、`core/recognize.py`、`voice/asr.py`；未实现的接口明确标注。
> 阈值相关：当前占位 `0.60`（`core.recognize.DEFAULT_THRESHOLD`），最终值由 `tools/calibrate.py` 用未参与录入的测试图校准。

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
- **被抽问怎么答**：为什么不自动保存？——显式 `save_library` 让多次录入合并为一次写盘，原子替换保证断电不损坏 JSON。重复图怎么处理？——像素摘要相同就跳过，不虚增条数。

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
# DEFAULT_THRESHOLD = 0.60
```

- **输入**：最高余弦相似度、阈值（默认 `DEFAULT_THRESHOLD = 0.60`）。**输出**：`True` 表示应判为未知物体。
- **为什么这样设计**：判定边界偏向"认出来"——`similarity < threshold` 才判未知，**等于阈值算已知**，避免边界抖动时结果来回跳。非有限值（NaN/Inf）一律判未知。
- **被抽问怎么答**：阈值现在多少？——占位 `0.60`，必须用未参与录入的测试图校准后写回，不能当最终结论。为什么用 `<` 不用 `<=`？——等于阈值算已知，边界更稳。

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

- **输入**：JSON 路径（默认 `data/features.json`）、`create` 开关。**输出**：校验通过的 `FeatureLibrary`，已绑定文件绝对路径。
- **为什么这样设计**：启动时加载 → 这就是"重启后特征库仍在"的实现。严格校验 schema / 版本 / 维度 / 向量范数；`create=False` 时文件不存在抛 `FileNotFoundError`，加载失败抛 `ValueError`，且不静默覆盖磁盘文件。
- **被抽问怎么答**：重启后数据还在吗？——在，`features.json` 落盘，启动时 `load_library` 读回来。文件坏了会怎样？——抛 `ValueError`，不覆盖原文件。

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
- **被抽问怎么答**：为什么不直接写目标文件？——原子替换保证要么旧文件完整、要么新文件完整。向量保留几位？——8 位小数，加载时重新归一化。

---

## 7. `core.recognize.recognize`（静态图片识别入口）

**文件**：`core/recognize.py`

```python
def recognize(
    library: FeatureLibrary,
    image: ImageInput | NDArray[np.floating],
    *,
    threshold: float | None = None,
    top_k: int = 5,
    encoder: Encoder | None = None,
) -> Recognition:
```

- **输入**：特征库、查询图片或一维向量、可选阈值与 `top_k`。**输出**：`Recognition`（`is_known`、`name`、`similarity`、`threshold`、`matches`、`object_scores`、`similarity_percent`）；未知时 `name="未知物体"`。
- **为什么这样设计**：Tab2 与命令行 `recognize` 共用的核心入口。先按物体聚合取**最大相似度**（`object_scores`），再与阈值比较判未知；空库直接返回"未知物体"。
- **被抽问怎么答**：`recognize` 和 `retrieve` 什么关系？——`recognize` 在 `retrieve` 之上做物体级聚合 + 阈值判定，产出最终结论。有 `recognize_static` 这个函数吗？——**代码里没有这个符号**；静态图片识别就是 `core.recognize.recognize`，界面上的 `app.recognize_image` 只是它的事件包装。

---

## 8. `voice.asr.asr`

**文件**：`voice/asr.py`

```python
def asr(audio_path: str | Path) -> str:
```

- **输入**：本地音频路径（wav / flac / ogg / mp3）。**输出**：识别文本（已 `strip()`）；静音时可能为空字符串。
- **为什么这样设计**：麦克风录音与预录音频走同一入口（录音落盘成 wav 后传路径）。密钥缺失时抛 `AsrNotConfigured`（**可预期的状态，不是崩溃**），调用方据此处理；Tab1 已接线：录音 → 回填名称；未配 Key 时按钮置灰。token 缓存约 30 天、提前 60 秒刷新；HTTP 401 或 `err_no` 110/111 强制重取并重试一次。依赖延迟导入，未装不影响界面启动。
- **被抽问怎么答**：缺 Key 会崩吗？——不会，`asr()` 抛 `AsrNotConfigured`，可捕获；键盘录入不受影响。密钥会进日志吗？——不会，`AsrConfig` 的 repr 屏蔽两把 Key，异常只带错误码。极速版怎么切？——`BAIDU_ASR_DEV_PID=80001`，端点随 `dev_pid` 自动切换。

---

## 9. `voice.tts`（Edge TTS 语音播报）—— **P4 未实现，接口待定**

**文件**：`voice/tts.py`（**尚不存在**；`voice/` 目前只有 `asr.py`）

> ⚠️ 当前代码库中**没有** Edge TTS 封装，`app.py` 也没有播报按钮。以下只是接口草案，不是已实现行为。

- **预期签名（草案，未定稿）**：`def tts(text: str, *, voice: str = "zh-CN-XiaoxiaoNeural") -> Path:`，返回合成音频路径供界面播放。
- **输入 / 输出（草案）**：输入待播报文本与音色名；输出本地音频文件路径（或播放完成后返回 `None`，接口未定）。
- **预期职责**：把识别结果文本用 Edge TTS 合成语音并播放；依赖 `edge-tts`，缺失时降级为文本显示，与 ASR 的"缺依赖不崩溃"原则一致。
- **为什么这样设计（草案）**：合成与播放放到后台线程，避免网络请求阻塞 Gradio 事件；TTS 失败只降级不抛到界面，与 ASR 的错误处理保持一致。
- **被抽问怎么答**：语音播报现在能跑吗？——不能，这是 P4 下一步，Edge TTS 合成放后台线程，避免阻塞界面。为什么还没做？——P6a 只交付已知部分的文档与校准，TTS 属未完成项，不写成已完成。

---

## 抽问高频「为什么」速查

| 问题 | 一句话答案 |
|------|------------|
| 为什么用 ResNet18 去掉 fc？ | 取 512 维通用视觉特征，才能"以图搜图"自定义物体 |
| 为什么 L2 归一化？ | 余弦相似度变点积，检索一次矩阵乘，阈值不受模长影响 |
| 为什么 JSON 不用数据库？ | 重启不丢、零依赖、约 300KB，SQLite/FAISS 属扩张不加分 |
| 为什么按物体取最大相似度？ | 多视角注册时"有一张很像就算像"，对多视角更友好 |
| 阈值为什么是 0.60？ | 占位值，必须用未参与录入的测试图校准，不能当结论 |
| 缺语音 Key 会崩吗？ | 不会：按钮置灰且不调用 `asr()`，键盘录入照常 |
