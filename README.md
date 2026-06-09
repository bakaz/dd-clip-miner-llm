# dd-clip-miner-llm

基于 ASR + LLM 的直播内容挖掘工具。支持从直播录像中识别和提取：

- 歌曲片段
- 有趣对话
- 高能时刻
- 搞笑片段
- 下头对话
- 当天直播结构化总结（仅报告，不切片）

**完全兼容** [dd-song-miner-llm](https://github.com/bakaz/dd-song-miner-llm) 的配置和工作流程。

## 特性

- **可插拔识别器**：每种内容类型独立实现（`recognizers/`）
- **多 ASR 后端**：faster-whisper（默认示例配置）、FunASR / Qwen3-ASR、远程 MiMo ASR
- **智能 LLM**：reasoning followup、工具调用、JSON 修复、歌词搜索
- **歌曲遗漏复查**：首轮后对未覆盖 ASR 区间二次检查
- **断点续传**：复用 `01_audio`、`02_asr`、LLM 结果（`progress.json`）
- **批量 + 多段合并**：`ConcatPipeline` 处理直播分段 H.264 损坏（mkvmerge 优先 + 6 策略 fallback）
- **切片命名**：主播词典 + 路径日期 → `【主播】歌名-歌手-YYMMDD`
- **手动重切**：改 CSV 后 `manual-cut`

## 工作流程

1. FFmpeg 提取 16 kHz 单声道 WAV
2. ASR 转写为带时间戳的 segment
3. 各识别器送 LLM 标注片段
4. 切割音频/视频到 `03_clips/`
5. 生成 `04_reports/` 下 CSV/JSON，可 `manual-cut` 重切

## 快速开始

```powershell
cd path\to\dd-clip-miner-llm
python install.py

copy config.example.yaml config.yaml
$env:LLM_API_KEY = "<your-api-key>"

python -m dd_clip_miner_llm run "D:\videos\live.mp4" --config config.yaml
```

也可使用入口脚本：`dd-clip-miner-llm run ...`（`pip install -e .` 后可用）。

## 安装

需要 **Python 3.10–3.12**（`pyproject.toml`）。

### 推荐：`install.py`

```powershell
python install.py
python install.py --config install.yaml --dev    # 含 pytest
python install.py --check                        # 只检测环境
```

检测 FFmpeg / mkvmerge / GPU，执行 `pip install -e .`，可选 `[funasr]`、`requirements-cu12.txt`、`[test]`。

### 手动安装

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .
pip install -e ".[test]"      # 可选
pip install -e ".[funasr]"    # 可选
pip install -r requirements-cu12.txt   # faster-whisper GPU（CUDA 12）
```

### 系统依赖

| 组件 | 用途 | Windows 示例 |
|------|------|----------------|
| FFmpeg + ffprobe | 抽音频、切片、探测 | `winget install Gyan.FFmpeg` |
| mkvmerge | 多段合并（可选，更稳） | `winget install MKVToolNix` |
| libsndfile | soundfile（Linux CI 需 `libsndfile1`） | 一般随环境已有 |

无 mkvmerge 时合并回退纯 FFmpeg。无 CUDA 12 DLL 时 faster-whisper 回退 CPU。

`setup.py` 仅为 setuptools 入口；交互式旧安装：`python setup_env.py`。

## 配置

复制模板并按需修改：

```powershell
copy config.example.yaml config.yaml
python -m dd_clip_miner_llm init-config --out config.yaml
```

| 文件 | 说明 |
|------|------|
| `config.example.yaml` | 主配置模板（含注释） |
| `config.deepseek.example.yaml` | DeepSeek |
| `config.daily-summary.example.yaml` | 仅当天总结 |
| `streamer_dictionary.example.json` | 主播词典 |

**勿提交**（已在 `.gitignore`）：`config.yaml`、`streamer_dictionary.json`、`runs/`。

ASR 支持两种写法（见 `config.example.yaml`）：

- **新格式**：`asr.mode: local | remote`，`local.backend: faster_whisper | funasr`
- **旧格式**：顶层 `asr.backend`（程序自动兼容）

LLM Key 优先环境变量：`LLM_API_KEY`、`DEEPSEEK_API_KEY`、`MIMO_API_KEY` 等，对应 `llm.api_key_env`。

### MiMo ASR 远程配置示例

```yaml
asr:
  mode: remote
  remote:
    provider: mimo
    base_url: https://token-plan-cn.xiaomimimo.com/v1
    api_key: null
    api_key_env: MIMO_API_KEY
    model: mimo-v2.5-asr
    timestamp_chunk_seconds: 5
```

## 用法

### 单视频

```powershell
python -m dd_clip_miner_llm run "D:\videos\live.mp4" --config config.yaml
python -m dd_clip_miner_llm run "D:\videos\live.mp4" --config config.yaml --content-types song,dialogue
python -m dd_clip_miner_llm run "D:\videos\live.mp4" --config config.yaml --video-codec auto --no-video-clips
```

### 批量

```powershell
python -m dd_clip_miner_llm batch-run "D:\input" --config config.yaml --work-root "D:\work" --result-root "D:\results"
python -m dd_clip_miner_llm batch-run "D:\input" --config config.yaml --work-root "D:\work" --result-root "D:\results" --concat
```

`--concat` 或 `output.concat_videos: true` 时，同目录多段录像先合并再处理。合并失败时查看 `runs/<name>_concat/concat/concat_attempts/*.log`。

### 手动重切

编辑 `04_reports/<type>/*.csv` 中的 `start` / `end`，然后：

```powershell
python -m dd_clip_miner_llm manual-cut "D:\runs\某次运行" --config config.yaml
```

### 常用参数

| 参数 | 说明 |
|------|------|
| `--content-types` | `song,dialogue,highlight,funny,cringe,daily_summary` |
| `--asr-model` / `--asr-language` | 覆盖 ASR |
| `--llm-model` / `--llm-api-key` / `--llm-base-url` | 覆盖 LLM |
| `--padding-before` / `--padding-after` | 歌曲 padding（兼容 dd-song-miner-llm） |
| `--video-codec` | `copy` / `auto` / `nv` / `intel` / `amd` / `cpu` |
| `--no-video-clips` | 只导出音频 |

### CLI 命令

| 命令 | 说明 |
|------|------|
| `run` | 单视频流水线 |
| `batch-run` | 批量目录 |
| `manual-cut` | 从 CSV 重切 |
| `init-config` | 生成默认 YAML |
| `ffmpeg-info` | GPU / 硬件编码器探测 |

## 切片命名

在 `output.clip_naming` 启用后，歌曲导出文件名形如 `【主播】晴天-周杰伦-260603.mp4`（日期 **仅** 从路径解析，如 `2026_06_03`）。需配置 `streamer_dictionary.json`；未命中则用 `default_streamer`。详情见 `config.example.yaml` 与 `clip_naming.py`。

后处理可拖拽 `rename_drag_drop.bat`（逻辑在 `scripts/rename_drag_drop.py`）。

## 项目结构

```
dd-clip-miner-llm/
├── pyproject.toml              # 包元数据；可选依赖 [test] [funasr]
├── setup.py                    # setuptools 入口（非安装脚本）
├── install.py / install.yaml   # 推荐安装（install.yaml 为安装配置模板）
├── setup_env.py                # 旧版交互安装
├── requirements*.txt           # 与 pyproject 同步的 pip 清单
├── config*.yaml                # 配置模板
├── streamer_dictionary.example.json
├── rename_drag_drop.bat
├── scripts/
│   ├── rename_drag_drop.py
│   └── probe_concat_strategies.ps1
├── tests/                      # 单元测试
├── .github/workflows/tests.yml
└── dd_clip_miner_llm/
    ├── cli.py / __main__.py    # python -m dd_clip_miner_llm
    ├── pipeline.py             # 主流水线
    ├── batch.py / manual.py
    ├── config.py / models.py / llm.py / report.py / merger.py
    ├── clip_naming.py / search_tools.py / paths.py
    ├── asr.py
    ├── asr_backends/
    │   ├── faster_whisper.py
    │   ├── funasr_backend.py
    │   └── mimo_asr_backend.py
    ├── recognizers/            # song / dialogue / highlight / funny / cringe / daily_summary
    ├── concat/                 # 多段录像合并流水线
    │   ├── models.py           # VideoMeta, ProblemProfile, ConcatContext
    │   ├── probe.py / planner.py
    │   ├── health.py           # 拼接前 H.264/HEVC 健康探测
    │   ├── strategies.py       # DirectCopy / MkvMerge / DiscardCorrupt / TargetedRepair / SelectiveNormalize / FullReencode
    │   ├── runner.py           # ConcatPipeline 编排
    │   ├── helpers.py
    │   └── pipeline.py         # concat_videos_smart() 对外入口
    └── ffmpeg/                 # FFmpeg / mkvmerge 工具层（由原 ffmpeg.py 拆分）
        ├── command.py          # run_command, require_binary
        ├── probe.py            # get_duration, detect_video_encoders, ...
        ├── validation.py       # 时长 / 音频可解码校验
        ├── diagnosis.py        # classify_ffmpeg_output, find_bad_h264_segments
        ├── encode.py           # 编码器候选
        ├── concat_ops.py       # concat demuxer、remux、TS 等底层操作
        ├── mkvmerge.py         # mkvmerge 拼接
        ├── bitstream.py / errors.py / fsutil.py / compat.py
        ├── media.py / single_input.py / legacy.py
        └── __init__.py         # from dd_clip_miner_llm import ffmpeg
```

**模块关系**：业务合并走 `concat.ConcatPipeline`；底层命令与诊断在 `ffmpeg/`。`ffmpeg.concat_videos()` 委托 `concat.pipeline.concat_videos_smart()`，保留旧 import 路径。

扩展识别器：在 `recognizers/` 新建模块，继承 `BaseRecognizer` 并用 `@register` 注册，再在 `content_types` 中启用。

## 运行输出

```
runs/<run_name>/
├── 00_input/                   # 输入（合并后为 concat.mp4）
├── 01_audio/source.wav
├── 02_asr/transcript.json
├── 02_asr/llm/<type>/          # matches.json, llm_batch_*.json, ...
├── 03_clips/audio|video/<type>/
├── 04_reports/<type>/          # songs.csv, dialogues.csv, ...
├── manifest.json / progress.json
├── clip_naming.json            # 可选
└── 05_manual/                  # manual-cut 输出
```

`daily_summary` 只写报告，不生成 `03_clips`。

## 视频编码与拼接

`output.video_codec` 默认 `copy`。`auto` 或需重编码时按 NVENC → QSV → AMF → libx264 选择。探测：`python -m dd_clip_miner_llm ffmpeg-info`。

多段合并（`--concat`）策略顺序：

1. **DirectCopy** — 参数一致时直接 copy
2. **MkvMerge** — mkvmerge 处理 H.264 bitstream 损坏（推荐）
3. **DiscardCorruptCopy** — ffmpeg `+discardcorrupt` 在 demux 层丢弃损坏包
4. **TargetedRepair** — 只重编码坏段，好段 copy
5. **SelectiveNormalize** — 只重编码不匹配的段
6. **FullReencode** — 最后兜底

安装 mkvmerge：`winget install MKVToolNix`。未安装时跳过 MkvMerge 策略。

## 开发与测试

```powershell
pip install -e ".[test]"
$env:DD_CLIP_MINER_LLM_CI = "1"
python -m pytest tests -q --basetemp=.tmp/pytest
```

GitHub Actions（`.github/workflows/tests.yml`）在 Ubuntu + Python 3.10–3.12 跑离线单元测试。

## 常见问题

| 现象 | 处理 |
|------|------|
| `Binary not found: ffprobe` | 安装完整 FFmpeg |
| `Binary not found: mkvmerge` | `winget install MKVToolNix` |
| `cublas64_12.dll is not found` | `pip install -r requirements-cu12.txt`，或接受 CPU 回退 |
| 中文路径乱码 | 用 PowerShell 7；或 `batch-run` 扫目录 |
| LLM 返回 0 条 | 查 `02_asr/llm/<type>/` 下 JSON；检查 key / `base_url`；网络是否可达 |
| `clip_naming` 未生效 | 确认 `enabled`、词典路径、路径含日期、`apply_to` |
| concat 输出时长异常 | 查 `concat_attempts/*.log`；确认输入文件无损坏 |
| MiMo ASR 连接失败 | 检查 `base_url` 和 `api_key`；确认网络可达 |

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

AGPL-3.0