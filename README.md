# dd-clip-miner-llm

基于 Whisper ASR + LLM 的直播内容挖掘工具。支持从直播录像中识别和提取：

- 歌曲片段
- 有趣对话
- 高能时刻
- 搞笑片段
- 下头对话
- 当天直播结构化总结（仅报告，不切片）

**完全兼容** [dd-song-miner-llm](https://github.com/bakaz/dd-song-miner-llm) 的配置和工作流程。

## 特性

- **可插拔识别器架构**：每种内容类型独立实现，易于扩展
- **多内容类型支持**：歌曲、对话、高能、搞笑、下头、当天总结
- **智能 LLM 调用**：reasoning followup、工具调用、JSON 修复、歌词搜索（DuckDuckGo）
- **歌曲遗漏复查**：首轮识别后对未覆盖 ASR 区间二次送 LLM 检查
- **断点续传**：同一输入视频再次运行时复用 `01_audio`、`02_asr`、各类型 LLM 结果（见 `progress.json`）
- **批量处理**：目录扫描；可选多视频拼接后统一处理
- **切片导出命名**：JSON 主播词典匹配 + 路径解析 YYMMDD → `【主播】歌名-歌手-YYMMDD`
- **手动重切**：编辑 CSV 后重新导出片段
- **下头片段短标题**：`title` 作为文件名，少于 20 个中文字

## 工作流程

1. FFmpeg 提取 16 kHz 单声道 WAV
2. faster-whisper 转写为带时间戳的 segment
3. 各识别器将 transcript 送 LLM 标注片段
4. 按时间切割音频/视频到 `03_clips/`
5. 生成 `04_reports/` 下 CSV/JSON，可人工修改后 `manual-cut`

## 仓库文件

| 文件 | 说明 |
|------|------|
| `config.example.yaml` | 配置模板（复制为 `config.yaml`） |
| `config.deepseek.example.yaml` | DeepSeek 示例配置 |
| `config.daily-summary.example.yaml` | 仅当天总结示例 |
| `streamer_dictionary.example.json` | 主播词典模板（复制为 `streamer_dictionary.json`） |
| `rename_drag_drop.bat` | 切片拖拽重命名（后处理） |

以下文件在 `.gitignore` 中，**勿提交**：`config.yaml`、`streamer_dictionary.json`、`runs/`。

## 安装

### 1. Python

建议 Python 3.10–3.12：

```powershell
cd path\to\dd-clip-miner-llm
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
pip install -e .
```

开发与测试：`pip install -r requirements-dev.txt`

### 2. FFmpeg

优先使用系统 PATH 中的 `ffmpeg` / `ffprobe`；否则回退 `imageio-ffmpeg` 自带二进制。

```powershell
winget install Gyan.FFmpeg
ffmpeg -version
ffprobe -version
```

无 `ffprobe` 时会尝试用 `ffmpeg -i` 解析时长，但建议安装完整 FFmpeg。

### 3. CUDA 可选（faster-whisper GPU）

```powershell
pip install -r requirements-cu12.txt
```

当前 CTranslate2 / faster-whisper 依赖 CUDA 12 运行时（`cublas64_12.dll`）。本机仅 CUDA 13 时仍需 CUDA 12 DLL；缺失时自动回退 CPU int8。

### 4. LLM API Key

```powershell
copy config.example.yaml config.yaml
python -m dd_clip_miner_llm init-config --out config.yaml
```

推荐环境变量，勿在仓库中写明文 key：

```powershell
$env:LLM_API_KEY="<your-api-key>"
```

或写入用户环境变量：

```powershell
[Environment]::SetEnvironmentVariable("LLM_API_KEY", "<your-api-key>", "User")
```

DeepSeek 等可参考 `config.deepseek.example.yaml`，设置对应 `api_key_env`（如 `DEEPSEEK_API_KEY`）。

## 快速开始

### 单个视频

```powershell
python -m dd_clip_miner_llm run "D:\videos\live.mp4" --config config.yaml
python -m dd_clip_miner_llm run "D:\videos\live.mp4" --config config.yaml --out "D:\runs\live_001"
```

### 只识别部分类型

```powershell
python -m dd_clip_miner_llm run "D:\videos\live.mp4" --config config.yaml --content-types song
python -m dd_clip_miner_llm run "D:\videos\live.mp4" --config config.yaml --content-types dialogue
python -m dd_clip_miner_llm run "D:\videos\live.mp4" --config config.yaml --content-types highlight
python -m dd_clip_miner_llm run "D:\videos\live.mp4" --config config.yaml --content-types funny
python -m dd_clip_miner_llm run "D:\videos\live.mp4" --config config.yaml --content-types cringe
python -m dd_clip_miner_llm run "D:\videos\live.mp4" --config config.yaml --content-types song,dialogue
python -m dd_clip_miner_llm run "D:\videos\live.mp4" --config config.daily-summary.example.yaml
```

### 常用 run 参数

| 参数 | 说明 |
|------|------|
| `--content-types` | 逗号分隔，覆盖配置中的 `content_types` |
| `--asr-model` / `--asr-language` | 覆盖 ASR |
| `--llm-model` / `--llm-api-key` / `--llm-base-url` | 覆盖 LLM |
| `--padding-before` / `--padding-after` | 歌曲 padding（秒），兼容 dd-song-miner-llm |
| `--no-video-clips` | 不导出视频 |
| `--export-audio` / `--export-video` | 覆盖扩展名 |
| `--video-codec` | `copy` / `auto` / `nv` / `intel` / `amd` / `cpu` |
| `--audio-bitrate-kbps` | 音频码率 |

### 批量处理

```powershell
python -m dd_clip_miner_llm batch-run "D:\input" --config config.yaml --work-root "D:\work" --result-root "D:\results"
```

每个视频所在文件夹可含 `.dd_clip_miner_done.json` 标记已完成项，失败项下次会重试。

### 批量合并（同目录多段录像）

`config.yaml` 中 `output.concat_videos: true`，或：

```powershell
python -m dd_clip_miner_llm batch-run "D:\input" --config config.yaml --work-root "D:\work" --result-root "D:\results" --concat
```

合并策略：

1. 轻量探测可能损坏的 H.264 片段
2. 源健康时优先流 copy；坏包则定点重编码（nv > intel > amd > cpu）
3. 校验拼接后总时长
4. 合并结果 stage 到 `00_input/input_*.mp4`；`manual-cut` 从 `manifest.json` 读输入；处理完成后清理 `concat/` 中间文件

### 手动重切

编辑报告 CSV 中的 `start` / `end`（时间码 `HH:MM:SS` 或秒数）：

- 推荐路径：`04_reports/song/songs.csv`
- 兼容旧路径：`04_reports/songs.csv`

```powershell
python -m dd_clip_miner_llm manual-cut "D:\runs\某次运行" --config config.yaml
python -m dd_clip_miner_llm manual-cut "D:\runs\某次运行" --config config.yaml --csv "D:\runs\某次运行\04_reports\song\songs.csv"
python -m dd_clip_miner_llm manual-cut "D:\runs\某次运行" --config config.yaml --content-type dialogue
```

输出默认在 `05_manual/`。启用 `clip_naming` 时重切命名规则与 `run` 一致。

## 切片导出命名

面向主播切片发布：**只改导出文件名**，不改报告里的 `title` / `artist`。

### 文件名

| 条件 | 示例 |
|------|------|
| 启用 `clip_naming` 且路径含合法日期（默认 `apply_to: [song]`） | `【主播名】晴天-周杰伦-260603.mp4` |
| 未启用或路径无 YYMMDD | `001-晴天-周杰伦.mp4` |

无歌手：`【主播名】标题-260603.mp4` 或 `001-标题.mp4`。分隔符 `-` **两侧无空格**。

### 主播词典（JSON）

```powershell
copy streamer_dictionary.example.json streamer_dictionary.json
```

```json
{
  "default_streamer": "StreamerName",
  "min_score": 0.65,
  "entries": [
    {
      "streamer": "你的主播名",
      "aliases": ["文件夹关键词", "房间号"]
    }
  ]
}
```

- 词典仅有 `streamer` + `aliases`，**不含日期**
- `dictionary_path` 相对 `config.yaml` 所在目录
- 命中：路径片段与 `aliases` 相似度最高且 `score >= min_score`
- 未命中：使用 `default_streamer`
- 运行后写入 `clip_naming.json`（`streamer`、`date`、`score`、`matched_alias`）

### 日期（YYMMDD）

**仅**从视频路径/父目录名解析，例如：

- `2026_06_03` → `260603`
- 独立六位 `250603`（校验月日）

找不到合法日期时控制台警告，并回退 legacy 命名 `001-歌名-歌手`。

推荐目录：

```text
D:\archive\房间号_主播名\2026_06_03\part1.mp4
```

### score 调参

相似度为规范化文本比较（完全相等、子串、或 `SequenceMatcher` 比例）。误匹配提高 `min_score`；漏匹配降低阈值或增加 `aliases`。

### 拖拽重命名（可选后处理）

未走流水线命名时，将视频拖到 `rename_drag_drop.bat` → `【主播】歌名-歌手-YYMMDD.mp4`。

- 默认识别 `001-歌名-歌手`；仍兼容带空格的旧名 `001 - 歌名 - 歌手`
- 主播/日期：`scripts/rename_drag_drop.py` 默认值，或 `CLIP_RENAMER_STREAMER`、`CLIP_RENAMER_DATE`（`mtime` = 文件修改日期的 YYMMDD）

## 配置文件

复制 `config.example.yaml` 为 `config.yaml` 后修改。结构与示例一致，要点如下。

```yaml
audio:
  sample_rate: 16000
  channels: 1

asr:
  model: small               # tiny | base | small | medium | large-v3
  device: auto               # auto | cpu | cuda
  compute_type: default      # default | float16 | int8
  language: null             # null=自动 | zh | ja | en
  beam_size: 5
  vad_filter: true
  initial_prompt: null

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
  batch_size: null           # null=整段；正整数=按 segment 分批
  use_tools: true
  verify_with_search: true
  json_fix_rounds: 3
  fallbacks: []

# 顶层 padding 会合并到 song.padding（兼容 dd-song-miner-llm）
padding:
  before_seconds: 15.0
  after_seconds: 15.0
  after_next_asr_end_guard_seconds: 2.0
  min_song_seconds: 30.0
  merge_gap_seconds: 35.0

content_types:
  song: true
  dialogue: true
  highlight: true
  funny: true
  cringe: true
  daily_summary: false

song:
  enabled: true
  padding:
    before_seconds: 15.0
    after_seconds: 15.0
    after_next_asr_end_guard_seconds: 2.0
    min_song_seconds: 30.0
    merge_gap_seconds: 35.0
  missed_recheck:
    enabled: true
    batch_size: 500
    min_gap_segments: 1

dialogue:
  enabled: true
  min_duration: 10.0
  max_duration: 300.0
  min_confidence: 0.6
  merge_gap_seconds: 10.0
  tags: [搞笑, 吐槽, 名场面, 金句, 互动, 高能]

highlight:
  enabled: true
  min_duration: 5.0
  max_duration: 120.0
  min_confidence: 0.6
  merge_gap_seconds: 15.0

funny:
  enabled: true
  min_duration: 5.0
  max_duration: 180.0
  min_confidence: 0.6
  merge_gap_seconds: 15.0

cringe:
  enabled: true
  min_duration: 5.0
  max_duration: 120.0
  min_confidence: 0.6
  merge_gap_seconds: 15.0

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

output:
  video_clips: true
  audio_segments: true
  audio_extension: mp3
  audio_bitrate_kbps: 320
  video_extension: mp4
  video_codec: copy              # copy | auto | 见下方 FFmpeg 节
  match_context_segments: 10
  concat_videos: false
  clip_naming:
    enabled: false
    dictionary_path: streamer_dictionary.json
    default_streamer: StreamerName
    min_score: 0.65
    apply_to: [song]
```

歌曲 padding 说明：`before_seconds` / `after_seconds` 在 ASR 段边界外扩展；`after_next_asr_end_guard_seconds` 限制与相邻段重叠；过短片段由 `min_song_seconds` 过滤；相邻同歌名由 `merge_gap_seconds` 合并。

## CLI 命令

| 命令 | 说明 |
|------|------|
| `run` | 单视频流水线 |
| `batch-run` | 批量目录 |
| `manual-cut` | 从 CSV 重切 |
| `init-config` | 生成默认 YAML |
| `ffmpeg-info` | GPU / 硬件编码器探测 |

## 识别器架构

```
dd_clip_miner_llm/
├── clip_naming.py
├── pipeline.py
├── batch.py / manual.py / merger.py / report.py
└── recognizers/
    ├── __init__.py    # @register 自动发现
    ├── base.py
    ├── song.py
    ├── dialogue.py
    ├── highlight.py
    ├── funny.py
    ├── cringe.py
    └── daily_summary.py
```

### 添加自定义识别器

1. 在 `recognizers/` 新建 `my_type.py`
2. 继承 `BaseRecognizer`，实现 `name`、`build_prompt`、`parse_response`（可选覆盖）
3. 使用 `@register` 注册

```python
from . import register
from .base import BaseRecognizer

@register
class MyRecognizer(BaseRecognizer):
    @property
    def name(self) -> str:
        return "my_type"

    def build_prompt(self, segments, batch_start, config) -> str:
        ...
```

在 `content_types` 中启用对应类型后即可被流水线调用。

## 输出结构

```
runs/<run_name>/
├── 00_input/              # staging 后的输入视频
├── 01_audio/source.wav
├── 02_asr/
│   ├── transcript.json
│   └── llm/<content_type>/
│       ├── matches.json
│       ├── match_context.csv
│       ├── match_context.json
│       └── llm_batch_*.json
├── 03_clips/
│   ├── audio/<content_type>/
│   └── video/<content_type>/
├── 04_reports/<content_type>/
│   ├── songs.csv / songs.json      # song
│   ├── dialogues.csv               # dialogue
│   └── ...
├── clip_naming.json       # 启用 clip_naming 时
├── manifest.json
├── progress.json          # 断点续传
└── 05_manual/             # manual-cut 输出（可选）
```

`daily_summary` 只写报告，不生成 `03_clips`。

## FFmpeg 编码

默认 `output.video_codec: copy`（不重编码、最快）。

`auto` 或切割失败时依次尝试：

| 值 | 编码器 |
|----|--------|
| `nv` | `h264_nvenc` |
| `intel` | `h264_qsv` |
| `amd` | `h264_amf` |
| `cpu` | `libx264` |
| `copy` | 流复制 |

```powershell
python -m dd_clip_miner_llm run video.mp4 --config config.yaml --video-codec nv
python -m dd_clip_miner_llm ffmpeg-info
```

## 常见问题

### `Binary not found: ffprobe`

安装完整 FFmpeg，确认 `ffprobe` 在 PATH。

### `cublas64_12.dll is not found`

```powershell
pip install -r requirements-cu12.txt
```

仍失败则使用 CPU 回退（较慢）。

### 中文/日文路径乱码

使用 PowerShell 7 / Windows Terminal；程序会 staging 非 ASCII 路径。若命令行参数已乱码，改用 `batch-run` 扫描目录。

### LLM 返回 0 条

查看 `02_asr/llm/<type>/llm_batch_*.json`、`matches.json`。检查 key、`base_url`、网络。歌曲过少时调 `song.padding.min_song_seconds`、`merge_gap_seconds`。

### `clip_naming` 未生效

确认 `enabled: true`、词典路径、`streamer_dictionary.json` 存在、输入路径含 `2026_06_03` 等形式日期、`apply_to` 含当前类型。

## 与 dd-song-miner-llm 的兼容性

| 功能 | dd-song-miner-llm | dd-clip-miner-llm |
|------|-------------------|-------------------|
| `run` / `batch-run` / `manual-cut` | ✅ | ✅ |
| `--padding-before` / `--padding-after` | ✅ | ✅ |
| 顶层 `padding` | ✅ | ✅ |
| `json_fix_rounds` / reasoning / tools | ✅ | ✅ |
| 多内容类型 | ❌ | ✅ |
| 可插拔识别器 | ❌ | ✅ |
| 歌曲遗漏复查 | ❌ | ✅ |
| JSON 主播词典切片命名 | ❌ | ✅ |

## License

MIT