# 核心层使用说明（给界面/语音模块对接用）

本目录只负责"图片 -> 特征 -> 特征库 -> 相似度 -> 未知判断"，不依赖 Gradio、语音或网络接口。
界面层（Tab1 录入 / Tab2 识别 / Tab3 摄像头）直接调用这里的函数即可。

## 文件职责

| 文件 | 职责 |
|------|------|
| `core/features.py` | ResNet18 加载、预处理、512 维特征提取、L2 归一化、图片解码与去重摘要 |
| `core/library.py` | JSON 特征库的增、查、保存、加载与完整性校验 |
| `core/recognize.py` | 余弦相似度 top-k 检索、按物体聚合、阈值判未知 |
| `tools/demo_cli.py` | 命令行验收：录入 / 识别 / 统计 / 批量评测 / 写阈值 |
| `tests/` | 假编码器单测（秒级）+ 可选真实 ResNet18 端到端测试 |

## 环境要求

- Python >= 3.10（用到 `X | None` 类型写法与 `dataclass(slots=True)`）
- torch / torchvision >= 0.13（需要 `torchvision.models.ResNet18_Weights`）
- numpy、Pillow、pytest

与 T1 的 `requirements.txt` 保持一致即可，核心层只需要：

```bash
pip install numpy pillow torch torchvision
```

torch 必须是 CPU 版；首次加载 ResNet18 会下载约 45MB 权重，之后离线可用。
如果下载失败，可手动把 `resnet18-f37072fd.pth` 放到
`%USERPROFILE%\.cache\torch\hub\checkpoints`（Linux/macOS 为
`~/.cache/torch/hub/checkpoints`），或在启动前设置 `TORCH_HOME`。

## 最小接入示例

```python
from core import add_to_library, load_library, recognize, save_library

library = load_library("data/features.json")   # 文件不存在时返回空库

# Tab1：录入（Gradio 的 gr.File 给的是临时路径，直接传路径即可）
result = add_to_library(library, "苹果", ["/tmp/a.jpg", "/tmp/b.jpg"])
if result.added:
    save_library(library)                      # 不调用 save 就不会落盘

# Tab2：识别（传路径 / PIL.Image / numpy 数组都行）
recognition = recognize(library, "/tmp/test.jpg")
print(recognition.name)              # 已知物体名，或 "未知物体"
print(recognition.similarity)        # 最高余弦相似度
print(recognition.similarity_percent)  # 0~100，适合直接显示
print(recognition.is_known)          # bool
print(recognition.threshold)         # 本次使用的阈值
for match in recognition.matches:    # 近邻候选，可展示前 3 个
    print(match.rank, match.name, match.image, match.similarity)

# 查看特征库（Tab1 的表格）
for row in library.stats():
    print(row["name"], row["count"])
```

### 为什么 `add_to_library` 不自动保存

显式 `save_library` 让"录入 5 张图只写一次盘"成为可能，也避免测试和误操作污染真实特征库。
如果界面希望"点一下就保存"，在事件函数末尾调用 `save_library(library)` 即可。

## JSON 格式（版本 1）

```json
{
  "schema": "hifox-resnet18-feature-library",
  "version": 1,
  "model": "resnet18-imagenet1k-v1",
  "dim": 512,
  "normalized": true,
  "threshold": 0.62,
  "created_at": "2026-11-09T02:30:00Z",
  "updated_at": "2026-11-09T02:31:00Z",
  "objects": [
    {
      "name": "苹果",
      "created_at": "2026-11-09T02:30:00Z",
      "entries": [
        {
          "image": "apple_01.jpg",
          "digest": "sha256...",
          "added_at": "2026-11-09T02:30:00Z",
          "vector": [0.0123, -0.0456, "...共 512 个数"]
        }
      ]
    }
  ]
}
```

- `entries` 的条数就是该物体的有效注册图数（重复图片会被跳过，不计数）。
- `digest` 是解码后 RGB 像素的 SHA-256，用来识别"同一张图重复录入"。
- 整个文件 UTF-8、`ensure_ascii=False`，中文名称可直接阅读。
- 保存采用临时文件 + `os.replace` 原子替换；写入失败时旧文件保持完好。

## 关键设计决策

1. **去掉 ResNet18 的 `fc`**：输出 512 维池化特征。保留 1000 类分类头只会得到
   ImageNet 类别分数，无法表达自定义物体。
2. **L2 归一化**：余弦相似度 = 点积，检索是一次 `(N,512) @ (512,)` 矩阵乘法；
   同时让阈值不随向量模长漂移。入库时统一归一化，加载时校验范数。
3. **按物体取最大相似度**：多视角注册时，只要有一张注册图与查询图接近，
   就认为该物体接近；平均分会因为"没拍到的角度"被拖低。
4. **阈值优先级**：显式参数 > 特征库 `threshold` > `DEFAULT_THRESHOLD`（0.60）。
   0.60 只是占位值，必须用未参与录入的测试图校准后写回（T6）。
5. **未知判断是 `score < threshold`**：等于阈值算已知，避免边界抖动时来回跳。
6. **重复图去重**：同一物体内按像素摘要去重，防止"向量条数 ≠ 注册图数"，
   也防止同一张图反复入库把该物体的 max 分数抬高。

## 命令行验收

```bash
# 录入（每个物体 5 张）
python tools/demo_cli.py add --name 苹果 --dir data/known/苹果
python tools/demo_cli.py add --name 香蕉 --dir data/known/香蕉

# 查看库
python tools/demo_cli.py stats

# 单图识别
python tools/demo_cli.py recognize --image data/test/苹果/t01.jpg --top-k 3

# 批量 Top-1（目录结构 data/test/<物体名>/*.jpg）
python tools/demo_cli.py eval --dir data/test

# 写入校准阈值
python tools/demo_cli.py threshold --set 0.63
```

所有命令支持 `--json`，便于把结果贴进报告。默认特征库是 `data/features.json`，
可用 `--library` 指定其它文件。

## 测试

```bash
# 秒级逻辑测试，不需要 torch
python -m pytest tests -q

# 真实 ResNet18 端到端（首次下载权重，5 类合成图案）
RUN_RESNET_TESTS=1 python -m pytest tests/test_integration_resnet.py -q
```

Windows PowerShell 设置环境变量：

```powershell
$env:RUN_RESNET_TESTS = "1"
python -m pytest tests/test_integration_resnet.py -q
```

### 本地验证记录（2026-11-09）

- 逻辑测试 61 项 + 真实 ResNet18 端到端 3 项，共 **64 passed**。
- 端到端覆盖：5 类合成图案各 5 张注册、5 张独立测试图，JSON 每物体 5 条向量、
  512 维、L2 范数≈1；保存后重新加载仍可检索；未知噪声图判为未知。
- 额外用 scikit-image 自带的 8 张公开真实照片做工程冒烟（5 已知 × 5 注册 + 2 测试，
  3 未知 × 2 测试）：Top-1 **10/10**，已知最低分 **0.9218**，未知最高分 **0.6182**，
  间隔 **0.3036**，中点阈值约 **0.77**。在 0.60 占位阈值下有一张"书页"被误判为已知，
  这说明**必须用课程数据校准阈值**，不能把 0.60 写进报告当作最终值。
- 上述真实照片只用于工程验证，不是课程数据；最终报告的数字必须来自 T2 采集的
  注册/测试图片。

## 已知边界

- 第一次调用 `get_default_encoder()` 会下载权重并加载模型，约 2~5 秒；之后同进程复用。
- CPU 上单张 224×224 推理约 30~80ms，摄像头链路建议 `stream_every >= 0.3s`。
- JSON 全量加载：25 条 512 维向量约 300KB，内存与耗时都可忽略；不建议再引入数据库。
- 特征库 `model` 字段只做记录，不会自动迁移。换模型（例如 ResNet50）必须重建特征库，
  否则新旧向量不在同一特征空间，相似度没有意义。
- 阈值 0.60 是占位值，未用真实测试集校准前，报告里的准确率不能当作最终结论。
