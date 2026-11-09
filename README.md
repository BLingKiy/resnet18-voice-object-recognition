# ResNet18 语音物体识别 Demo

基于本地 ImageNet 预训练 **ResNet18** 的物体识别 Demo：多图录入并保存特征，输入图片返回最相似物体与相似度，相似度过低判为未知；带简单界面（录入 / 识别 / 查看），重启后特征库仍在。语音链路由百度短语音识别（模块已实现）与 Edge TTS 播报（P4 待实现）组成，当前界面用键盘录入、文本与标注图展示结果。摄像头实时识别为加分项。

> 运行结束重启后，`data/features.json` 仍在，无需重新录入。
> 课程数据在本仓库 `data/` 下：`data/known/<物体名>/`（注册图，每物体 5 张）、`data/test/<物体名>/`（测试图，每物体 2 张，与注册图严格分开）。图片与特征库都不入库，只有 `data/manifest.csv` 清单随仓库提交。

## 为什么有这个项目

作业要求做一个"能认东西"的本地 Demo，但不能依赖在线识别接口、服务器或手机 App。我们选择 ResNet18 取 512 维特征 + 余弦相似度：模型本地跑、特征库就是一个 JSON 文件，**零部署、可离线、重启不丢数据**。语音识别只用百度短语音识别（录入名称），语音播报计划用 Edge TTS（P4 待实现）；缺密钥或依赖时降级为纯键盘操作。

## 快速开始（到"能跑"的最短路径）

```bash
# 1. 准备环境（Python 3.10+，CPU 版 torch）
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt

# 2. 录入物体（每个物体 5 张以上，目录结构 data/known/<物体名>/*.jpg）
python tools/demo_cli.py add --name water_cup --dir data/known/water_cup

# 3. 识别一张测试图（首次运行会联网下载约 45 MB 的 ResNet18 权重，之后离线可用）
python tools/demo_cli.py recognize --image data/test/water_cup/01.jpg --top-k 3
```

启动图形界面：

```bash
python app.py
# 浏览器打开 http://127.0.0.1:7860
```

## 环境安装

前置条件（带版本号）：

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| Python | ≥ 3.10 | 用了 `X | None` 类型写法和 `dataclass(slots=True)` |
| torch / torchvision | ≥ 0.13 | 必须 **CPU 版**；需要 `torchvision.models.ResNet18_Weights` |
| numpy | ≥ 1.23 | 特征计算 |
| Pillow | 最新 | 图片解码与预处理 |
| gradio | ≥ 4 | 图形界面（Tab1 录入 / Tab2 识别） |
| soundfile / httpx | 最新 | **语音识别**才需要（百度 ASR）；缺了也能跑键盘模式 |
| edge-tts | 最新 | **语音播报**才需要（P4 待实现）；缺了也能跑纯文本模式 |

安装（pip，任选其一）：

```bash
pip install numpy pillow torch torchvision gradio
# 语音链路（可选，缺省则降级为键盘录入 + 文本结果显示）
pip install soundfile httpx edge-tts
```

> 离线权重拷贝：若目标机器无法联网，把 `resnet18-f37072fd.pth` 放到
> `~/.cache/torch/hub/checkpoints/`（Linux/macOS）或
> `%USERPROFILE%\.cache\torch\hub\checkpoints\`（Windows）；
> 也可设置环境变量 `TORCH_HOME` 指向可写目录。文件名必须与 torchvision 期望的一致（默认即 `resnet18-f37072fd.pth`）。

## 演示前检查

答辩与现场演示前逐条过一遍，任何一条不过就会当场掉链子：

- [ ] **中文字体**：Windows 自带微软雅黑即可；Linux 需装 Noto CJK，否则标注图上的中文显示成方框。
- [ ] **摄像头**：设备管理器里能看到，且没被别的程序占用（Tab3 实时识别用；加分项，讲解时再开）。
- [ ] **麦克风**：系统录音里能看到输入电平；Tab1「语音说名称」用，录不到就退回键盘录入。
- [ ] **Edge TTS 需联网**：播报走微软在线服务，断网时界面会显示「语音播报不可用，请检查网络」，不是程序坏了。
- [ ] **ResNet18 权重已预热**：`~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`（Windows 为 `%USERPROFILE%\.cache\torch\hub\checkpoints\`）已存在，避免现场第一次运行临时下载约 45 MB。

## 使用

### 命令行（验收与批处理）

```bash
# 逐个物体录入（目录或显式文件列表）
python tools/demo_cli.py add --name power_bank --dir data/known/power_bank
python tools/demo_cli.py add --name earphone   --dir data/known/earphone

# 查看特征库（物体数 / 向量条数 / 阈值）
python tools/demo_cli.py stats

# 单图识别，展示前 3 个近邻
python tools/demo_cli.py recognize --image data/test/power_bank/01.jpg --top-k 3

# 批量 Top-1 评测（目录结构 data/test/<物体名>/*.jpg）
python tools/demo_cli.py eval --dir data/test --json

# 阈值校准（见下方"阈值与未知判断"）
python tools/calibrate.py --known-dir data/test --unknown-dir data/unknown
```

所有命令都支持 `--json`，便于把结果贴进报告或做自动化断言。

### 图形界面

- **Tab1 · 多图录入**：输入物体名称（键盘）→ 上传该物体的多张图片 → 点「录入并保存」。重复图片按像素摘要去重，不会虚增向量条数。
- **Tab2 · 单图识别**：上传一张测试图 → 点「开始识别」。页面显示物体名、相似度（含百分比）、是否未知，以及前 3 个近邻和标注图（绿框=已知，红框=未知）。
- **刷新特征库**：从磁盘重新读取 `data/features.json`，用于验证"重启后特征库仍在"。

语音识别封装在 `voice/asr.py`（百度短语音识别），已接进 Tab1：录音 → 识别 → 回填物体名称，用户可改后再入库。未配置密钥时按钮置灰、提示用键盘输入，`asr()` 不调用，程序不崩溃；键盘录入始终可用。语音播报（Edge TTS）为 P4 待实现。

## 配置（`.env`）

语音识别需要百度智能云的 API Key 与 Secret Key。把仓库里的 `.env.example` 复制为 `.env` 并填入：

```ini
# .env.example 内容（复制为 .env 后填值）
BAIDU_ASR_API_KEY=你的API_KEY
BAIDU_ASR_SECRET_KEY=你的SECRET_KEY
# 选填：1537=标准版（默认），80001=极速版
BAIDU_ASR_DEV_PID=1537
```

密钥优先级（高 → 低，每个键独立取第一个非空值）：

1. 进程环境变量 `BAIDU_ASR_API_KEY` / `BAIDU_ASR_SECRET_KEY` / `BAIDU_ASR_DEV_PID`
2. `RN18_ENV_FILE` 指向的文件
3. `~/.rn18/.env`
4. 仓库根目录 `.env`

密钥**只**从环境变量或 `.env` 读取，绝不写死在代码里，也不进日志（异常信息只会带错误码，不会拼接含 Secret 的请求 URL）。`.env` 已在 `.gitignore` 中，不会被提交。

图形界面的其他可选环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `RN18_LIBRARY_PATH` | `data/features.json` | 特征库 JSON 路径 |
| `RN18_HOST` | `127.0.0.1` | 监听地址（演示只给本机） |
| `RN18_PORT` | `7860` | 监听端口 |

## 阈值与未知判断（重要）

判定规则：**最高相似度 < 阈值** 即判为"未知物体"。阈值当前是核心层里的**占位值 `0.60`**，必须用**未参与录入**的测试图校准后写回，不能把占位值当作最终结论。

校准步骤（`tools/calibrate.py`）：

1. 用 `data/test/<物体名>/` 的已知类测试图统计"已知类最高相似度"分布；
2. 用 `data/unknown/<物体名>/`（≥3 类、每类 2~3 张）统计"未知类最高相似度"分布；
3. 取"已知类最小值"与"未知类最大值"的中点作为建议阈值，写回特征库：
   `python tools/demo_cli.py threshold --set <建议值>`。

> 当前阶段（验收前）未知类样本尚未采集，`calibrate.py` 会显式输出"待补数据"，**不会生成或引用任何未知判断的最终结论**。注册图与测试图必须严格分开，否则阈值会被自己污染。

## 数据与特征库格式

- 注册图：`data/known/<物体名>/*.jpg`（每物体 ≥5 张；本仓库为 `power_bank` / `water_cup` / `earphone` / `keyboard` / `mouse`）
- 测试图：`data/test/<物体名>/*.jpg`（每物体 ≥2 张，与注册图严格分开）
- 未知类：`data/unknown/<物体名>/*.jpg`（≥3 类，验收前补齐）
- 清单：`data/manifest.csv` 是**唯一入库清单**，35 行 = 25 张 known + 10 张 test，按**文件字节 SHA-256** 去重，35 个摘要全不相同（CSV 第 4 列列名沿用 `md5`，实际值为 SHA-256）
- 特征库：`data/features.json`（JSON，版本 1，512 维，L2 归一化；原子写入，重启不丢）

## 文档与交付物

- [项目报告 REPORT.md](REPORT.md) — 系统功能 / 实现方法 / 测试结果 / 成员分工
- [关键函数讲解卡 docs/function_cards.md](docs/function_cards.md) — 抽问高频函数说明
- [现场演示脚本 docs/demo_script.md](docs/demo_script.md) — 验收演示分步清单
- 核心层说明见 [core/README.md](core/README.md)

## 许可证

课程大作业，仅供教学演示使用。
