# FireRedASR2S 语音识别系统

FireRedASR2S 是一款基于深度学习的端到端语音识别系统，集成了VAD（语音活动检测）和ASR（自动语音识别）功能，支持批量处理音频文件并输出带时间戳的字幕文件。

## 功能特性

- **VAD语音活动检测**：精确检测音频中的语音片段
- **AED端到端识别**：基于Attention Encoder Decoder架构的高精度语音识别
- **批处理优化**：支持批量处理多个音频文件，优化内存使用
- **多种格式支持**：支持WAV、MP3、FLAC、AAC、OGG、M4A等音频格式，以及MP4、AVI、MKV等视频格式
- **时间戳输出**：支持词级和句子级时间戳输出
- **标点恢复**：可选的智能标点恢复功能

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
然后学习输出的示例。操作对象是目标文件地址或者目录，支持混合输入，会自动识别。默认配置已实现最优。


简单解释
- --nfp, --no-fp16     原精度，识别准确率更高，双倍资源消耗，效率更低。

- --ts, --timestamp     返回词级时间戳  ：时间戳更精确，但是主观感受不出来，拖慢5%左右效率。

- --dur SECONDS      根据GPU实力调整，建议为32的倍数。当前单片段时长为0.32s-16s，每秒为100帧。（以6G的RTX3060为例，开启游戏模式，RTF在0.03左右）

- --mvbs                  实验特性，即VAD模型是否同时处理多个音频。除非你音频时长相差不大，否则不建议改为并行（大于1的整数）。


### 标点配置 (config.py - PUNCTUATION_CONFIG)

私以为原标点模型虽然效率极高，但实际效果不理想，故而移除了，改用自定义AI模型处理。

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

`split_vad_to_batches()` 函数使用动态聚类分桶策略：
1. **按音频分组**：将VAD片段按`audio_idx`分组，保留数据局部性
2. **动态分桶**：时长相近的片段放入同桶（同桶比≤3，最大桶数4）
3. **桶内贪心分批**：每个桶内按`max_batch_dur_s`做最佳适应贪心分批
4. **短片段填充**：≤1s的短片段会被填充到有剩余空间的batch中


## 核心特色

相比于原开源项目，最大的改进就是：

- Beam Search 解码器实现了增量解码 + KV Cache
- 工业级别的分批策略：降序排序 + 动态分桶+最佳适应贪心 + 短片段填充
- 若干算法复杂度问题的优化。

## 开源许可
本项目基于以下开源项目：
- FireRedASR2S：https://github.com/FireRedTeam/FireRedASR2S
- Qwen
- WenetSpeech-Yue
- WenetSpeech-Chuan
