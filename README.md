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
- **智能 LLM 调用**：支持 reasoning followup、工具调用、JSON 修复、歌词搜索
- **歌曲遗漏复查**：首轮识别后对未覆盖 ASR 区间二次检查
- **断点续传**：支持复用上次运行的音频、ASR、LLM 结果
- **批量处理**：目录扫描、多视频拼接后处理
- **切片导出命名**：可选 JSON 主播词典 + 路径日期，导出 `【主播】歌名-歌手-YYMMDD`
- **手动重切**：从编辑后的 CSV 重新切割片段
- **下头片段短标题**：`title` 用于文件名，限制少于 20 个中文字

## 工作流程

1. 用 FFmpeg 从视频提取 16 kHz 单声道 WAV
2. 用 faster-whisper 生成带时间戳的 ASR segment
3. 通过识别器将 transcript 交给 LLM 识别各类型内容
4. 根据识别结果切割音频/视频片段
5. 输出 CSV/JSON 报告（可人工校正后 `manual-cut`）

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

可编辑安装：

```powershell
pip install -e .
```

开发依赖（测试、lint）：`pip install -r requirements-dev.txt`

### 2. 准备 FFmpeg

程序会优先使用系统 PATH 里的 `ffmpeg` / `ffprobe`。若无系统 FFmpeg，会回退到 `imageio-ffmpeg` 自带二进制。

```powershell
winget install Gyan.FFmpeg
ffmpeg -version
ffprobe -version
```

### 3. NVIDIA CUDA 可选加速

```powershell
pip install -r requirements-cu12.txt
```

若缺少 `cublas64_12.dll`，程序会自动回退 CPU int8（较慢）。

### 4. 准备 LLM Key

```powershell
python -m dd_clip_miner_llm init-config --out config.yaml
```

推荐用环境变量，不要把真实 key 提交到仓库：

```powershell
$env:LLM_API_KEY="<your-api-key>"
```

`config.yaml`、`clip_dictionary.json` 已在 `.gitignore` 中。

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
python -m dd_clip_miner_llm run "D:\videos\live.mp4" --config config.yaml --content-types song
python -m dd_clip_miner_llm run "D:\videos\live.mp4" --config config.yaml --content-types song,dialogue,cringe
python -m dd_clip_miner_llm run "D:\videos\live.mp4" --config config.daily-summary.example.yaml
```

### 批量处理

```powershell
python -m dd_clip_miner_llm batch-run "D:\input" --config config.yaml --work-root "D:\work" --result-root "D:\results"
```

### 批量处理（目录内多视频合并）

配置 `output.concat_videos: true` 或使用：

```powershell
python -m dd_clip_miner_llm batch-run "D:\input" --config config.yaml --work-root "D:\work" --result-root "D:\results" --concat
```

合并策略概要：

1. 轻量探测可能损坏的 H.264 片段
2. 健康源优先流 copy；坏包则定点重编码（nv > intel > amd > cpu）
3. 校验拼接后时长；staging 到 `00_input/input_*.mp4`，完成后清理 `concat/` 中间文件

### 手动重切

编辑 `04_reports/song/songs.csv`（或其它类型对应 CSV）中的 `start` / `end` 后：

```powershell
python -m dd_clip_miner_llm manual-cut "D:\runs\某次运行" --config config.yaml
python -m dd_clip_miner_llm manual-cut "D:\runs\某次运行" --config config.yaml --content-type dialogue
```

输出默认在 `05_manual/`。若启用了 `clip_naming`，重切时同样应用命名规则。

## 切片导出命名

用于主播内容切片发布：通过 **JSON 外挂词典** 匹配主播名，从 **输入路径** 解析日期，生成导出文件名。CSV/JSON 报告中的 `title`、`artist` **不会被改写**。

### 文件名格式

| 模式 | 示例 |
|------|------|
| 启用 `clip_naming`（默认作用于 `song`） | `【主播名】晴天-周杰伦-260603.mp4` |
| 未启用或路径无合法日期 | `001-晴天-周杰伦.mp4` |

无歌手时：`【主播名】标题-260603.mp4` 或 `001-标题.mp4`（`-` 两侧无空格）。

### 准备词典

```powershell
copy clip_dictionary.example.json clip_dictionary.json
```

`clip_dictionary.json` 仅放本地，不要提交仓库。示例结构：

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

词典 **只包含** `streamer` 与 `aliases`（识别词）。日期 **不在词典中配置**。

### 日期解析（YYMMDD）

仅从视频路径、父目录名等文本中严格提取，例如：

- `2026_06_03` → `260603`
- 路径中的独立六位数字 `250603`（校验月日合法）

路径中找不到合法 YYMMDD 时，会警告并回退为 `001-歌名-歌手` _legacy 命名。

### 配置

```yaml
output:
  clip_naming:
    enabled: true
    dictionary_path: clip_dictionary.json   # 相对 config.yaml 所在目录
    default_streamer: StreamerName          # 词典未命中时的主播名
    min_score: 0.65                       # 路径与 aliases 的相似度阈值
    apply_to:
      - song                              # 也可列出 dialogue 等类型
```

运行后在输出目录生成 `clip_naming.json`，记录本次解析的 `streamer`、`date`、`score`、`matched_alias`。

### 相似度（score）说明

将词典里每条 `aliases` 与路径各片段做文本相似度（规范化后比较，支持子串与 `SequenceMatcher`），取全局最高分。`score >= min_score` 时采用该条的 `streamer`，否则用 `default_streamer`。

- 误匹配多：提高 `min_score`（如 `0.75`）
- 经常匹配不上：降低阈值或补充 `aliases`

### 路径建议

便于同时命中主播与日期，推荐目录结构类似：

```text
D:\archive\房间号_主播名\2026_06_03\part1.mp4
```

### 拖拽重命名（后处理）

若未启用流水线命名，可将切片拖到 `rename_drag_drop.bat`，转为 `【主播】歌名-歌手-YYMMDD.mp4`。脚本默认识别 `001-歌名-歌手` 或带空格的旧格式；主播与日期在 `scripts\rename_drag_drop.py` 或环境变量 `CLIP_RENAMER_STREAMER` / `CLIP_RENAMER_DATE` 中配置。

## 配置文件

完整示例见 `config.example.yaml`、`config.deepseek.example.yaml`。核心结构：

```yaml
audio:
  sample_rate: 16000
  channels: 1

asr:
  model: small
  device: auto
  compute_type: default
  language: null
  beam_size: 5
  vad_filter: true

llm:
  api_key: null
  api_key_env: LLM_API_KEY
  base_url: null
  model: gpt-4o
  use_tools: true
  json_fix_rounds: 3

padding:                        # 兼容旧项目顶层 padding，会同步到 song.padding
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
  padding: { ... }
  missed_recheck:
    enabled: true
    batch_size: 500
    min_gap_segments: 1

dialogue:
  enabled: true
  min_duration: 10.0
  max_duration: 300.0
  tags: [搞笑, 吐槽, 名场面, 金句, 互动, 高能]

highlight: { enabled: true, min_duration: 5.0, ... }
funny: { enabled: true, min_duration: 5.0, ... }
cringe: { enabled: true, min_duration: 5.0, max_duration: 120.0, ... }

daily_summary:
  enabled: false
  summary_only: true

output:
  video_clips: true
  audio_segments: true
  audio_extension: mp3
  video_codec: copy
  match_context_segments: 10
  concat_videos: false
  clip_naming:
    enabled: false
    dictionary_path: clip_dictionary.json
    default_streamer: StreamerName
    min_score: 0.65
    apply_to: [song]
```

## CLI 命令

| 命令 | 说明 |
|------|------|
| `run` | 处理单个视频 |
| `batch-run` | 批量扫描目录 |
| `manual-cut` | 从 CSV 重切 |
| `init-config` | 生成默认 `config.yaml` |
| `ffmpeg-info` | 查看 GPU / 编码器 |

## 识别器架构

```
dd_clip_miner_llm/
├── clip_naming.py       # 切片导出命名
├── pipeline.py          # 主流水线
└── recognizers/
    ├── song.py
    ├── dialogue.py
    ├── highlight.py
    ├── funny.py
    ├── cringe.py
    └── daily_summary.py
```

添加自定义识别器：在 `recognizers/` 新建模块，继承 `BaseRecognizer`，使用 `@register` 装饰器注册。

## 输出结构

```
runs/xxx/
├── 00_input/
├── 01_audio/source.wav
├── 02_asr/
│   ├── transcript.json
│   └── llm/{content_type}/
│       ├── matches.json
│       ├── match_context.csv
│       └── llm_batch_*.json
├── 03_clips/
│   ├── audio/{content_type}/
│   └── video/{content_type}/
├── 04_reports/{content_type}/
│   ├── songs.csv / songs.json        # song 类型
│   └── ...
├── clip_naming.json                  # 启用 clip_naming 时
├── manifest.json
└── progress.json
```

`daily_summary` 仅写入报告，不生成 `03_clips` 片段。

## FFmpeg 编码策略

`output.video_codec: copy` 为默认（不重编码、最快）。`auto` 在 copy 失败后依次尝试 `h264_nvenc` → `h264_qsv` → `h264_amf` → `libx264`。

```powershell
python -m dd_clip_miner_llm run video.mp4 --video-codec nv
python -m dd_clip_miner_llm ffmpeg-info
```

## 常见问题

### `Binary not found: ffprobe`

安装完整 FFmpeg 并确认 `ffprobe` 在 PATH 中。

### `cublas64_12.dll is not found`

```powershell
pip install -r requirements-cu12.txt
```

### FFmpeg 中文/日文路径乱码

使用 PowerShell 7 / Windows Terminal；必要时用 `batch-run` 扫描目录，避免命令行参数乱码。

### LLM 返回 0 条结果

检查 `02_asr/llm/{type}/llm_batch_*.json`、`matches.json`；确认 API key、`base_url`、网络。歌曲过少时调整 `min_song_seconds`、`merge_gap_seconds`。

### `clip_naming` 未生效

1. `output.clip_naming.enabled: true`
2. `clip_dictionary.json` 路径正确（相对 `config.yaml`）
3. 输入路径含合法日期（如 `2026_06_03`）
4. `apply_to` 包含当前内容类型

## 与 dd-song-miner-llm 的兼容性

| 功能 | dd-song-miner-llm | dd-clip-miner-llm |
|------|-------------------|-------------------|
| `run` / `batch-run` / `manual-cut` | ✅ | ✅ |
| 顶层 `padding` | ✅ | ✅ |
| `json_fix_rounds` / reasoning / tools | ✅ | ✅ |
| 多内容类型 / 可插拔识别器 | ❌ | ✅ |
| 歌曲遗漏复查 | ❌ | ✅ |
| JSON 主播词典切片命名 | ❌ | ✅ |

## License

MIT