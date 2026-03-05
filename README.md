# FireRedASR2S-cli 语音识别系统

FireRedASR2S-cli 是一款基于开源项目 [FireRedASR2S](https://github.com/FireRedTeam/FireRedASR2S) 开发的端到端语音识别系统，支持批量处理音频/视频文件并输出带时间戳的字幕文件或者纯文本。

截止至2026年3月初，据 [FireRedASR2S](https://github.com/FireRedTeam/FireRedASR2S) 官方报告称 FireRedASR2S 已经取得了开源 ASR 模型的 SOTA 。

本项目移除了就转写需求而言的不需要的 LID模型、体验不佳的 punc 模型，并实现了更深层次的算法优化，主要面向家用消费级显卡，欢迎跑官方原版做对比测试。（triton_tensorrt 版犯规喔。）

## 环境要求

- Python 3.8+（python>=3.12可能需要额外解决soudfile问题）
- CUDA 
- ffmpeg (用于音频格式转换)

## 安装步骤

### 1. 克隆项目

```bash
git clone <repository-url>
cd fireredasr
```

### 2. 安装依赖（建议在uv等虚拟环境中操作）

```bash
pip install -r requirements.txt
```
⚠️：pytorch并没有写进依赖中，请安装自己需要的版本。（只需要 torch、torchaudio，不需要torchvision。cuda118及以上即可）

https://pytorch.org/get-started/locally/

例子：安装适合 cuda126 的最新版 pytorch

pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126

### 3. 安装ffmpeg

**Windows:**
```bash
# 使用Chocolatey
choco install ffmpeg

# 或下载预编译二进制文件并添加到PATH
```

### 4. 下载预训练模型

项目需要下载VAD和ASR预训练模型。运行下载脚本：

```bash
python download_models.py
```

模型将下载到 `pretrained_models/` 目录：
- `pretrained_models/FireRedVAD/VAD` - VAD模型
- `pretrained_models/FireRedASR2-AED` - ASR模型

### 5. 开启TF32（可选但推荐）

针对系统而不是本项目：

```bash
python check_tf32.py
```

## 使用说明

### 基本用法
```bash
python cli.py -h
```
然后学习输出的示例。操作对象是目标文件地址或者所在目录，支持混合输入，会自动识别。默认配置已实现最优。


简单解释
- --nfp, --no-fp16     原精度，识别准确率更高，双倍资源消耗，效率更低。

- --ts, --timestamp     返回词级时间戳  ：时间戳更精确，但是主观感受不出来，拖慢5%左右效率。

- --dur SECONDS      单个批次总时长上限。根据GPU实力调整，建议为32的倍数。当前单片段时长为0.5s-16s，每秒为100帧。（见 config.py:"min_speech_frame": 50, "max_speech_frame": 1600，最大为2000否则出错）

（以6G的RTX3060为例，开启游戏模式，RTF在0.03左右）

--BFD  使用BFD算法进行分批。理论上默认的WFD更好，但由于缺乏足够的测试数据，故而保留了旧方法。



### 标点配置 (config.py - PUNCTUATION_CONFIG)

私以为原标点模型虽然效率极高，但实际效果不理想，故而移除，改用自定义AI模型处理。

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `enabled` | True | 是否启用标点恢复功能 |
| `endpoint` | - | 标点恢复API端点，任意兼容openai格式的 |
| `api_key` | - | API密钥 |
| `model` | - | 使用的LLM模型 |
| `prompt` | - | 标点恢复提示词 |
| `max_timeout` | 600 | API超时时间(秒) |

（请自行去config.py填入相关参数，参数配置错误不会影响任务转写，因为标点是任务结束后的额外操作。）


### 分批策略

原理：ASR 模型做批量推理时，为了满足 GPU 张量的维度一致性要求，同一个 batch 内的所有音频片段必须 Padding 到该 batch 中「最长片段的长度」（按帧数 / 采样点数计算，对应时长即 batch 内最长片段的时长）。比如 batch 内有 16s、8s、0.32s 的片段，8s 和 0.32s 的片段会被补零（zero-padding）到 16s 的长度，再和 16s 片段组成统一维度的张量送入模型。WFD与BFD算法是为了最大限度降低无效计算的比重。


## 核心特色

相比于原开源项目，最大的改进就是：

- Beam Search 解码器实现了增量解码 + KV Cache。
- 工业级别的分批策略：默认WFD，备用BFD。
- 若干算法复杂度问题的优化。

## 开源许可
本项目基于以下开源项目：
- FireRedASR2S：https://github.com/FireRedTeam/FireRedASR2S
- Qwen
- WenetSpeech-Yue
- WenetSpeech-Chuan
