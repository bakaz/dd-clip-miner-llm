# dd-clip-miner-llm

基于 Whisper ASR + LLM 的直播内容挖掘工具。支持从直播录像中识别和提取：
- 🎵 歌曲片段
- 💬 有趣对话
- ⭐ 高能时刻
- 😂 搞笑片段

**完全兼容** [dd-song-miner-llm](https://github.com/bakaz/dd-song-miner-llm) 的配置和工作流程。

## 特性

- **可插拔识别器架构**：每种内容类型独立实现，易于扩展
- **多内容类型支持**：歌曲、对话、高能时刻、搞笑片段
- **智能 LLM 调用**：支持 reasoning followup、工具调用、JSON 修复
- **断点续传**：支持复用上次运行结果
- **批量处理**：支持目录批量处理
- **手动重切**：支持从 CSV 重新切割片段
- **下头片段短标题**：下头片段的 `title` 会作为输出文件名，自动限制为少于20个中文字的总结

## 工作流程

1. 用 FFmpeg 从视频提取 16 kHz 单声道 WAV
2. 用 faster-whisper 生成带时间戳的 ASR segment
3. 把完整 transcript 交给 LLM 识别不同类型的内容
4. 根据识别结果切割音频/视频片段
5. 输出可人工校正的报告

## 安装

### 1. 准备 Python

建议使用 Python 3.10 到 3.12。Windows 下推荐在项目目录创建虚拟环境：

```powershell
cd D:\opencode\dd-clip-miner-llm
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

如果使用可编辑安装：

```powershell
pip install -e .
```

### 2. 准备 FFmpeg

程序会优先使用系统 PATH 里的 `ffmpeg` / `ffprobe`。如果没有系统 FFmpeg，会回退到 `imageio-ffmpeg` 自带的 FFmpeg。

建议 Windows 上安装完整 FFmpeg：

```powershell
winget install Gyan.FFmpeg
```

安装后重新打开终端，检查：

```powershell
ffmpeg -version
ffprobe -version
```

### 3. NVIDIA CUDA 可选加速

如果想用 GPU 跑 faster-whisper，安装 CUDA 12 相关 pip 包：

```powershell
pip install -r requirements-cu12.txt
```

如果本机只有 CUDA 13，仍然需要 CUDA 12 的运行时 DLL，因为当前 CTranslate2 / faster-whisper 依赖 `cublas64_12.dll`。程序检测到 CUDA 缺失时会自动回退 CPU int8，但速度会慢很多。

### 4. 准备 LLM Key

生成配置文件：

```powershell
python -m dd_clip_miner_llm init-config --out config.yaml
```

推荐不要把真实 key 写进仓库文件，使用环境变量：

```powershell
$env:LLM_API_KEY="<your-api-key>"
```

或写入用户环境变量后重新打开终端：

```powershell
[Environment]::SetEnvironmentVariable("LLM_API_KEY", "<your-api-key>", "User")
```

## 快速开始

### 单个视频

```powershell
python -m dd_clip_miner_llm run "D:\videos\live.mp4" --config config.yaml
```

### 指定输出目录

```powershell
python -m dd_clip_miner_llm run "D:\videos\live.mp4" --config config.yaml --out "D:\runs\live_001"
```

### 只识别特定类型

```powershell
# 只识别歌曲
python -m dd_clip_miner_llm run "D:\videos\live.mp4" --config config.yaml --content-types song

# 只识别对话
python -m dd_clip_miner_llm run "D:\videos\live.mp4" --config config.yaml --content-types dialogue

# 只识别高能时刻
python -m dd_clip_miner_llm run "D:\videos\live.mp4" --config config.yaml --content-types highlight

# 只识别搞笑片段
python -m dd_clip_miner_llm run "D:\videos\live.mp4" --config config.yaml --content-types funny

# 只生成当天直播结构化总结
python -m dd_clip_miner_llm run "D:\videos\live.mp4" --config config.daily-summary.example.yaml

# 组合识别
python -m dd_clip_miner_llm run "D:\videos\live.mp4" --config config.yaml --content-types song,dialogue
```

### 批量处理

```powershell
python -m dd_clip_miner_llm batch-run "D:\{input_dir}" --config config.yaml --work-root "D:\{output_dir}" --result-root "D:\{output_dir}"
```

### 批量处理（合并目录下的多个视频）

如果一个目录下有多个视频文件，可以使用 `--concat` 参数将它们合并后再处理：

```powershell
python -m dd_clip_miner_llm batch-run "D:\{input_dir}" --config config.yaml --work-root "D:\{output_dir}" --result-root "D:\{output_dir}" --concat
```

合并策略：
1. 先轻量探测可能损坏的 H.264 片段，避免在已知坏包时做无效的全量 copy。
2. 源文件健康时优先音视频流 copy（最快，不重编码）。
3. 如果只有个别片段坏包，优先只修复坏片段；编码器顺序为硬件优先（nv > intel > amd > cpu）。
4. 每次拼接后都会校验输出时长，避免 FFmpeg 退出码为 0 但实际只拼了一部分。
5. pipeline 会把最终合并视频 stage 到 `00_input/input_*.mp4`，后续 `manual-cut` 从 `manifest.json` 使用这份输入；`concat/concat.mp4` 等中间文件会在处理完成后清理。

### 手动重切

先复制或直接编辑某次运行的 `04_reports\songs.csv`，调整 `start` / `end`，然后：

```powershell
python -m dd_clip_miner_llm manual-cut "D:\runs\某次运行" --config config.yaml
```

### 拖拽重命名切片

把导出的视频文件拖到仓库根目录的 `rename_drag_drop.bat` 上，会自动改成：

```text
【StreamerName】歌曲名-歌手名-250101.mp4
```

默认会从 `001 - 歌曲名 - 歌手名.mp4` 这类文件名里解析歌名和歌手。主播名和日期默认写在 `scripts\rename_drag_drop.py` 顶部，也可以用环境变量 `CLIP_RENAMER_STREAMER` / `CLIP_RENAMER_DATE` 覆盖；日期传 `mtime` 时会使用文件修改日期。

## 配置文件

完整配置结构如下。`config.example.yaml` 和 `config.deepseek.example.yaml` 也可以直接复制修改。

```yaml
# 音频预处理
audio:
  sample_rate: 16000
  channels: 1

# ASR 配置
asr:
  model: small               # tiny | base | small | medium | large-v3
  device: auto               # auto | cpu | cuda
  compute_type: default      # default | float16 | int8
  language: null             # null=自动 | "zh" | "ja" | "en"
  beam_size: 5
  vad_filter: true
  initial_prompt: null

# LLM 配置
llm:
  api_key: null
  api_key_env: LLM_API_KEY
  base_url: null
  model: gpt-4o
  temperature: 0.1
  max_tokens: 8192
  max_completion_tokens: null
  retry_empty_with_reasoning: true
  reasoning_followup_rounds: 5
  reasoning_followup_max_tokens: 32768
  batch_size: null
  use_tools: true
  verify_with_search: true
  json_fix_rounds: 3
  fallbacks: []

# 时间 padding
# before_seconds: 歌曲开始前 padding（基于前一个 ASR start + guard）
# after_seconds: 歌曲结束后 padding（不超过下一个 ASR end - guard）
# 歌曲 start 不会早于“前一个 ASR start + after_next_asr_end_guard_seconds”
padding:
  before_seconds: 15.0
  after_seconds: 15.0
  after_next_asr_end_guard_seconds: 2.0
  min_song_seconds: 30.0
  merge_gap_seconds: 35.0

# 要识别的内容类型（true/false 控制启用/禁用）
content_types:
  song: true
  dialogue: true
  highlight: true
  funny: true
  cringe: true
  daily_summary: false

# 歌曲识别配置
song:
  enabled: true
  padding:
    before_seconds: 15.0
    after_seconds: 15.0
    after_next_asr_end_guard_seconds: 2.0
    min_song_seconds: 30.0
    merge_gap_seconds: 35.0
  missed_recheck:
    enabled: true       # 第一轮歌曲识别后，将未覆盖的 ASR 片段再送 LLM 检查遗漏
    batch_size: 500     # 二次检查时每批最多包含的 ASR segment 数
    min_gap_segments: 1 # 小于该长度的未覆盖区间不复查

# 对话识别配置
dialogue:
  enabled: true
  min_duration: 10.0
  max_duration: 300.0
  min_confidence: 0.6
  merge_gap_seconds: 10.0
  tags:
    - 搞笑
    - 吐槽
    - 名场面
    - 金句
    - 互动
    - 高能

# 高能时刻配置
highlight:
  enabled: true
  min_duration: 5.0
  max_duration: 120.0
  min_confidence: 0.6
  merge_gap_seconds: 15.0

# 搞笑片段配置
funny:
  enabled: true
  min_duration: 5.0
  max_duration: 180.0
  min_confidence: 0.6
  merge_gap_seconds: 15.0

# 当天直播结构化总结配置
daily_summary:
  enabled: false
  summary_only: true
  language: zh-CN
  title: 当天直播内容总结
  max_level1_items: 6
  max_level2_per_level1: 5
  max_level3_per_level2: 4
  include_timeline: true
  include_quotes: true
  include_open_questions: true

# 输出配置
output:
  video_clips: true
  audio_segments: true
  audio_extension: mp3           # 默认 mp3，兼容性最好
  audio_bitrate_kbps: 320
  video_extension: mp4
  video_codec: copy              # copy=不重编码; auto=nv > intel > amd > cpu
  match_context_segments: 10
  concat_videos: false           # 合并目录下的多个视频后再处理
```

## 识别器架构

项目采用可插拔的识别器架构，每种内容类型都有独立的识别器模块：

```
dd_clip_miner_llm/
└── recognizers/
    ├── __init__.py      # 注册表和自动发现
    ├── base.py          # 抽象基类
    ├── song.py          # 歌曲识别器
    ├── dialogue.py      # 对话识别器
    ├── highlight.py     # 高能时刻识别器
    ├── funny.py         # 搞笑片段识别器
    └── cringe.py        # 下头对话识别器
```

### 添加自定义识别器

1. 在 `recognizers/` 目录下创建新文件
2. 继承 `BaseRecognizer` 基类
3. 使用 `@register` 装饰器注册

```python
from . import register
from .base import BaseRecognizer

@register
class MyRecognizer(BaseRecognizer):
    @property
    def name(self) -> str:
        return "my_type"
    
    def build_prompt(self, segments, batch_start, config) -> str:
        # 构建 LLM 提示词
        ...
```

## 输出结构

```
runs/xxx/
├── 00_input/
├── 01_audio/
│   └── source.wav
├── 02_asr/
│   ├── transcript.json
│   └── llm/
│       ├── song/
│       │   ├── matches.json
│       │   └── match_context.csv
│       ├── dialogue/
│       │   ├── matches.json
│       │   └── match_context.csv
│       ├── highlight/
│       │   ├── matches.json
│       │   └── match_context.csv
│       └── funny/
│           ├── matches.json
│           └── match_context.csv
├── 03_clips/
│   ├── audio/
│   │   ├── song/
│   │   ├── dialogue/
│   │   ├── highlight/
│   │   └── funny/
│   └── video/
│       ├── song/
│       ├── dialogue/
│       ├── highlight/
│       └── funny/
├── 04_reports/
│   ├── song/
│   │   ├── songs.csv
│   │   └── songs.json
│   ├── dialogue/
│   │   ├── dialogues.csv
│   │   └── dialogues.json
│   ├── highlight/
│   │   ├── highlights.csv
│   │   └── highlights.json
│   └── funny/
│       ├── funnies.csv
│       └── funnies.json
└── manifest.json
```

## FFmpeg 编码策略

视频导出配置：

```yaml
output:
  video_codec: copy
```

`copy` 会直接复制原视频流，通常是最快、且画质不变的剪切方式。`auto` 会先尝试 copy，失败后按以下顺序重编码：

1. NVIDIA NVENC: `h264_nvenc`
2. Intel Quick Sync: `h264_qsv`
3. AMD AMF: `h264_amf`
4. CPU: `libx264`

显式指定：

```powershell
--video-codec nv      # h264_nvenc
--video-codec intel   # h264_qsv
--video-codec amd     # h264_amf
--video-codec cpu     # libx264
--video-codec copy    # 不重编码，直接复制原视频流
```

查看本机 GPU、FFmpeg 硬件加速和可用 H.264 编码器：

```powershell
python -m dd_clip_miner_llm ffmpeg-info
```

## 常见问题

### `Binary not found: ffprobe`

安装系统 FFmpeg，并确认 `ffprobe -version` 可用。没有 ffprobe 时程序会尝试用 `ffmpeg -i` 解析时长，但推荐安装完整 FFmpeg。

### `cublas64_12.dll is not found`

安装 CUDA 12 运行时依赖：

```powershell
pip install -r requirements-cu12.txt
```

如果仍失败，程序会自动回退 CPU int8，只是速度会慢。

### FFmpeg 找不到中文/日文路径

优先使用 PowerShell 7 / Windows Terminal。程序会自动 staging 非 ASCII 路径，但如果参数在进入 Python 前已经乱码，建议改用 `batch-run` 扫描目录。

### LLM 返回 0 首

先看：

```
02_asr\llm\song\llm_batch_000000.json
02_asr\llm\song\matches.json
02_asr\llm\song\match_context.csv
```

如果 `raw_response` 为空或有 `Connection error`，优先检查 API key、网络和 `base_url`。如果有 matches 但最终报告少，检查 `min_song_seconds` 和 `merge_gap_seconds`。

## 与 dd-song-miner-llm 的兼容性

本项目完全兼容 [dd-song-miner-llm](https://github.com/bakaz/dd-song-miner-llm) 的配置文件和工作流程。

| 功能 | dd-song-miner-llm | dd-clip-miner-llm |
|------|-------------------|-------------------|
| `run` 命令 | ✅ | ✅ |
| `batch-run` 命令 | ✅ | ✅ |
| `manual-cut` 命令 | ✅ | ✅ |
| `--padding-before/after` | ✅ | ✅ |
| 顶层 `padding` 配置 | ✅ | ✅ |
| `json_fix_rounds` | ✅ | ✅ |
| Reasoning followup | ✅ | ✅ |
| Tool call 处理 | ✅ | ✅ |
| 多内容类型扩展 | ❌ | ✅ |
| 可插拔识别器架构 | ❌ | ✅ |

## License

MIT
