# 启动说明（运行说明）

> 本页是现场启动速查。完整说明见仓库根 [README.md](../README.md)，报告见 [REPORT.md](../REPORT.md)，演示顺序见 [docs/demo_script.md](demo_script.md)。
> **交付包默认空库**：`data/features.json` 的 `objects` 为空、不含任何向量；程序启动不载入预置向量、不扫描样本目录。第一次打开要先现场录入物体。

## 1. 环境安装（一次）

前置条件：Python ≥ 3.10，CPU 版 torch / torchvision ≥ 0.13，numpy、Pillow、gradio ≥ 4；语音链路另需 soundfile / httpx / edge-tts。

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

离线权重：把 `resnet18-f37072fd.pth` 放到
`~/.cache/torch/hub/checkpoints/`（Linux/macOS）或
`%USERPROFILE%\.cache\torch\hub\checkpoints\`（Windows），也可设置 `TORCH_HOME`。
中文字体：Windows 自带微软雅黑即可；Linux 需装 Noto CJK，否则标注图上的中文显示成方框。

语音识别：把 `.env.example` 复制为 `.env`，填入 `BAIDU_ASR_API_KEY` / `BAIDU_ASR_SECRET_KEY`，极速版设 `BAIDU_ASR_DEV_PID=80001`。缺密钥时降级为键盘录入 + 文本结果，不崩溃。

## 2. 启动

```bash
python app.py
# 浏览器打开 http://127.0.0.1:7860
```

界面三个 Tab：Tab1 多图录入、Tab2 单图识别、Tab3 摄像头实时识别与教学式录入（加分项）。

## 3. 首次使用（三步）

```bash
# 1) 空库启动：预期「物体数: 0   特征向量: 0」
python tools/demo_cli.py stats

# 2) 现场录入一个新物体：≥5 张不同角度的图放在 my_object/
python tools/demo_cli.py add --name 现场新物体 --dir my_object

# 3) 另拍一张测试图识别；未录入的物体应判「未知物体」
python tools/demo_cli.py recognize --image my_object_test.jpg --top-k 3
```

图形界面等价路径：Tab1 输入名称 + 上传多图 + 点「录入并保存」→ Tab2 上传测试图 + 点「开始识别」。重复图片按像素摘要去重，不虚增条数。

阈值 `0.8124` 是课程 5 物体样本上校准的**参考值**，不是通用常数；现场物体差异较大时用 `tools/calibrate.py` 重新校准。`data/realtime.json` 的 0.75 / 0.90 / 0.05 是未在标准流程下校准的占位值。

## 4. 重启后数据仍在

关掉 `python app.py` 再启动，刚录入的物体仍在：`data/features.json` 采用临时文件 + `os.replace` 原子写入，启动时由 `load_library` 读回。Tab1 的「刷新特征库」可从磁盘重读核对。

## 5. 常见问题

- **首次运行卡在下载权重**：现场先按 §1 预热离线权重；首跑默认会联网下载约 45 MB。
- **标注图中文变方框**：Linux 装 Noto CJK；Windows 用微软雅黑。
- **识别提示"特征库为空"**：正常，交付包是空库，先按 §3 录入。
- **播报没声音**：Edge TTS 需联网，断网时界面显示「语音播报不可用，请检查网络」，识别结果照常。检查音量不为 0。
- **命令行换库**：用 `--library` 或环境变量 `RN18_LIBRARY_PATH` 指定其它 JSON；一台机器只用一个库，不要混用。
