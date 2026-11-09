# 关键函数讲解卡（教师随机抽问用）

> 每个函数都给出：函数签名（输入输出）、设计理由、被抽问时怎么答。所有签名来自 `core/` 与 `voice/` 的实际代码，未实现的接口明确标注。
> 阈值相关：当前占位 `0.60`，最终值由 `tools/calibrate.py` 用未参与录入的测试图校准。

---

## 1. `core.features.extract_feature`

```python
def extract_feature(image: ImageInput, *, encoder: Encoder | None = None) -> NDArray[np.float32]:
    """返回 (512,) 已 L2 归一化特征向量"""
```

- **输入**：单张图片（路径 / `PIL.Image` / numpy 数组）。**输出**：`(512,)` 的 float32 向量，L2 范数 = 1。
- **设计理由**：去掉 ResNet18 的 `fc` 层，只取 512 维全局平均池化特征；输出统一 L2 归一化，使余弦相似度退化为点积，检索变成一次矩阵乘法。延迟导入 torch，纯逻辑测试不需加载 1100 万参数模型。
- **被抽问**：为什么是 512 维？——ResNet18 去掉分类头后的池化输出就是 512 维（ImageNet 的特征维度）。为什么要归一化？——归一化后相似度 = 点积，且向量模长不再影响阈值，阈值能跨图片复用。

---

## 2. `core.library.add_to_library`

```python
def add_to_library(library, name, images, *, labels=None, encoder=None) -> AddResult:
    """向特征库添加一个物体的多张图片（仅改内存，需再 save_library 落盘）"""
```

- **输入**：特征库实例、物体名称、图片序列（路径 / PIL / numpy）。**输出**：`AddResult`（新增条数、重复跳过条数、该物体向量数、全库总数）。
- **设计理由**：录入与落盘分离——"录入 5 张图只写一次盘"，也避免测试/误操作污染真实特征库。同一物体内按**像素 SHA-256 摘要**去重，保证"向量条数 = 有效注册图数"，不会因重复上传虚高。
- **被抽问**：为什么 `add` 不自动保存？——显式 `save_library` 让多次录入合并为一次写盘，原子替换保证断电不损坏 JSON。重复图怎么处理？——解码后 RGB 像素的 SHA-256 比对，相同则跳过。

---

## 3. `core.recognize.retrieve`

```python
def retrieve(library, image, *, top_k: int = 5, encoder=None) -> list[Match]:
    """图片级 top-k 检索，返回按相似度降序的 Match 列表"""
```

- **输入**：特征库、查询图片（或已提取的一维向量）、`top_k`。**输出**：`Match` 列表（每条含 `name`、`image`、`similarity`、`rank`）；空库返回空列表。
- **设计理由**：`matrix @ query` 一次算出全库相似度，`argsort` 取前 k。空库直接返回空，不抛异常。供界面展示前 3 个近邻。
- **被抽问**：相似度怎么算的？——`library.matrix()` 拼出 `(N,512)` 检索矩阵，与查询向量做点积（因已 L2 归一化，点积即余弦相似度）。返回的是图片级还是物体级？——图片级，但置信度用物体级聚合（见 `recognize`）。

---

## 4. `core.recognize.judge_unknown`

```python
def judge_unknown(similarity: float, threshold: float = 0.60) -> bool:
    """similarity < threshold 返回 True（判未知）；等于阈值算已知"""
```

- **输入**：最高余弦相似度、阈值。**输出**：`True` 表示应判为未知物体。
- **设计理由**：判定边界偏向"认出来"——相似度恰好等于阈值时算已知，避免边界抖动时结果来回跳。非有限值（NaN/Inf）一律判未知，绝不输出垃圾名称。
- **被抽问**：阈值现在是多少？——占位 `0.60`，必须用未参与录入的测试图校准后写回，绝不当作最终结论。为什么用 `<` 而不是 `<=`？——等于阈值算已知，边界更稳。

---

## 5. `core.library.load_library`

```python
def load_library(path=None, *, create: bool = True) -> FeatureLibrary:
    """从 JSON 加载特征库，默认 data/features.json；文件不存在时 create=True 返回空库"""
```

- **输入**：JSON 路径（默认 `data/features.json`）、`create` 开关。**输出**：校验通过的 `FeatureLibrary`，已绑定文件路径。
- **设计理由**：启动时加载 → 这就是"重启后特征库仍在"的实现。严格校验 schema / 版本 / 维度 / 向量范数；加载失败抛 `ValueError` 且不静默覆盖磁盘文件，防止把用户数据弄丢。
- **被抽问**：重启后数据还在吗？——在。`features.json` 是落盘的文件，启动时 `load_library` 读回来。文件坏了会怎样？——抛 `ValueError`，不覆盖原文件。

---

## 6. `core.library.save_library`

```python
def save_library(library, path=None) -> Path:
    """原子写入 JSON 文件（临时文件 + os.replace）"""
```

- **输入**：特征库实例、可选路径。**输出**：实际写入的绝对路径。
- **设计理由**：先写同目录临时文件、调用 `fsync`、`os.replace` 原子替换目标——演示现场程序被强杀也不会留下半个 JSON 把特征库彻底弄坏。
- **被抽问**：为什么不直接写目标文件？——原子替换保证要么旧文件完整、要么新文件完整，没有中间损坏态。向量保留几位？——8 位小数，文件约 1/3 体积，余弦误差 < 1e-7，远小于图片本身波动，加载时再归一化修正舍入。

---

## 7. `core.recognize.recognize`（即"静态识别 `recognize_static`"）

```python
def recognize(library, image, *, threshold=None, top_k=5, encoder=None) -> Recognition:
    """识别一张图片，返回名称、相似度、是否未知；未知时 name='未知物体'"""
```

- **输入**：特征库、查询图片（或一维向量）、可选阈值与 `top_k`。**输出**：`Recognition`（含 `is_known`、`name`、`similarity`、`threshold`、`matches`、`object_scores`、`similarity_percent`）。
- **设计理由**：这是 Tab2 与命令行 `recognize` 共用的核心入口。先按物体聚合取**最大相似度**（`object_scores`），再与阈值比较判未知；`similarity_percent` 把余弦相似度映射到 0~100 便于界面显示。空库直接返回"未知物体"。
- **被抽问**：`recognize` 和 `retrieve` 什么关系？——`recognize` 在 `retrieve` 之上做物体级聚合 + 阈值判定，产出最终"是什么 / 是不是未知"的结论。`recognize_static` 指的就是这个静态图片识别函数（界面 Tab2 的 `recognize_image` 只是它的事件包装）。

---

## 8. `voice.asr.asr`

```python
def asr(audio_path: str | Path) -> str:
    """识别一个本地音频文件中的语音，返回识别文本（已 strip）"""
```

- **输入**：本地音频路径（wav / flac / ogg / mp3）。**输出**：识别文本；静音时可能为空字符串。
- **设计理由**：麦克风录音与预录音频走同一入口（录音落盘成 wav 后传路径）。密钥缺失时抛 `AsrNotConfigured`（**可预期状态，不是崩溃**），调用方据此置灰语音入口、键盘录入照常。token 缓存约 30 天、提前 60 秒刷新，401 / err_no 110·111 强制重取并重试一次。依赖（soundfile / httpx）延迟导入，未装也不影响界面启动。
- **被抽问**：缺 Key 程序会崩吗？——不会。`asr()` 抛 `AsrNotConfigured`，界面捕获后置灰语音按钮并提示，键盘录入不受影响。密钥会不会进日志？——不会，`AsrConfig` 的 repr 屏蔽两把 Key，异常只带错误码。极速版怎么切？——`BAIDU_ASR_DEV_PID=80001`，端点随 dev_pid 自动切换，不用改代码。

---

## 9. `voice.tts`（Edge TTS 语音播报）—— **待实现**

> ⚠️ 当前代码库中**尚未实现** Edge TTS 封装（`app.py` 仅有 Tab1 录入与 Tab2 识别，无语音播报；仓库内无 `tts` 模块）。以下为**接口契约草案**，供 P2/P3 收口时落地，不引用任何未验证行为。

- **预期签名**（草案）：`def tts(text: str, *, voice: str = "zh-CN-XiaoxiaoNeural") -> None`（或返回音频路径供界面播放）。
- **预期职责**：把识别结果文本用 Edge TTS 合成语音并播放；依赖 `edge-tts`，缺失时降级为文本显示，与 ASR 的"缺依赖不崩溃"原则一致。
- **被抽问**：语音播报现在能跑吗？——TTS 封装尚未落地，目前识别结果以文本 + 标注图展示；缺 TTS 依赖不影响键盘与文本模式。这是已知待办，不是缺陷隐瞒。

---

## 抽问高频「为什么」速查

| 问题 | 一句话答案 |
|------|------------|
| 为什么用 ResNet18 去掉 fc？ | 取 512 维通用视觉特征，才能"以图搜图"自定义物体 |
| 为什么 L2 归一化？ | 余弦相似度变点积，检索一次矩阵乘，阈值不受模长影响 |
| 为什么 JSON 不用数据库？ | 重启不丢、零依赖、约 300KB，SQLite/FAISS 属扩张不加分 |
| 为什么按物体取最大相似度？ | 多视角注册时"有一张很像就算像"，对多视角更友好 |
| 阈值为什么是 0.60？ | 占位值，必须用未参与录入的测试图校准，不能当结论 |
| 缺语音 Key 会崩吗？ | 不会，置灰语音入口，键盘录入照常 |
