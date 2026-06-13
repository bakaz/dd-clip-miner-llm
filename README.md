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
- **多 ASR 后端**：faster-whisper（批量推理 `BatchedInferencePipeline`）、FunASR / Qwen3-ASR、远程 MiMo ASR
- **智能 LLM**：reasoning followup、工具调用、JSON 修复、歌词搜索
- **KV 缓存优化**：`cache_friendly_prompt_layout` 复用 ASR 前缀，`compact_segment_ranges` 减少输出 token
- **V3 三轮分段流水线**：高精度发现 → 未覆盖召回审计 → 全量时序裁决
- **时序裁决**：全量 ASR 二次审视，修正首轮边界，支持名称保留
- **副歌感知拆分**：40–120 秒间隔根据文本相似度判断是否为副歌重现
- **同名相邻合并**：排序后字面相邻、标题相同且间隔 ≤ 40 秒的候选自动合并
- **搜索验证命名**：对未知歌曲用歌词锚点搜索，需歌词证据才更新名称
- **锚点漏检审计**：可选的 anchor-based 补查，单次 LLM 调用，默认关闭
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
python -m dd_clip_miner_llm run "D:\videos\live.mp4" --config config.yaml --profile accuracy
python -m dd_clip_miner_llm run "D:\videos\live.mp4" --config config.yaml --profile kv_optimized
```

### 批量

```powershell
python -m dd_clip_miner_llm batch-run "D:\input" --config config.yaml --work-root "D:\work" --result-root "D:\results"
python -m dd_clip_miner_llm batch-run "D:\input" --config config.yaml --work-root "D:\work" --result-root "D:\results" --concat
python -m dd_clip_miner_llm batch-run "D:\input" --config config.yaml --profile kv_optimized --work-root "D:\work" --result-root "D:\results"
```

配置包含 `profiles` 时，音频和 ASR 由两个 profile 共享，LLM、切片和报告分别写入
`02_asr/llm/<profile>`、`03_clips/<profile>`、`04_reports/<profile>`。
`accuracy` 保留 task-first 和 `segment_indices`；`kv_optimized` 使用
`risk_routed_v3`、缓存友好布局和三轮对象协议。时长、边界膨胀和重叠只作为复核风险，
不会被当作全局硬过滤条件。两套 profile 共享歌曲 padding，默认
`merge_gap_seconds: 40`。`accuracy` 显式使用本地 review 和 windowed missed-recheck；
`kv_optimized` 固定执行 Precision Discovery、Recall Audit 和 Segmentation Adjudication。
第一轮只保留明确演唱，第二轮只输出未覆盖区间的短证据，第三轮统一修边界并可有限补漏。
三轮都不搜索歌词；分段稳定后才可独立命名。
两个 profile 都完成后会生成
`02_asr/llm/profile_comparison.json` 和 `profile_comparison.md`。

两套 profile 共用 `max_completion_tokens: 32768` 和 `final_tool_max_tokens: 32768`。
任何普通请求、最终工具轮、review 或 missed-recheck 返回 `finish_reason: length` 时，
会在保留原 ASR 请求前缀的情况下续写剩余 JSON，最多 8 轮。仍未完成的结果会标记
`scan_incomplete`，不会作为可复用缓存。

`song.missed_recheck.strategy` 可设为 `windowed` 或 `full_transcript`。
全量审计会写入 `missed_recheck/audit.json`，其中包含输入指纹、目标区间、
结构失败原因、fallback 状态和当前有效的 LLM 调试文件。

离线比较不会调用 API：

```powershell
python scripts/evaluate_song_pipeline_v2.py "results\<date>\<run>" --output ".tmp\song-v2-evaluation.json"
```

报告使用 accuracy 的高置信已命名区间作为弱时间参考，并单独标记当前固定样本中的
150-360 秒外风险结果；该范围不会进入运行时硬约束。加 `--enforce` 可让质量或费用门槛失败时返回非零。

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
| `--profile` | 选择 YAML 中的 `accuracy` 或 `kv_optimized` profile |
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
│   ├── evaluate_song_pipeline_v2.py
│   ├── adaptive_cost_probe.py
│   ├── review_scope_ab.py
│   └── probe_concat_strategies.ps1
├── tests/                      # 单元测试
├── .github/workflows/tests.yml
└── dd_clip_miner_llm/
    ├── cli.py / __main__.py    # python -m dd_clip_miner_llm
    ├── pipeline.py             # 主流水线
    ├── batch.py / manual.py
    ├── config.py               # 配置加载、profile 管理、歌曲 pipeline 选择
    ├── models.py / llm.py / report.py / merger.py
    ├── profile_state.py        # profile 指纹、usage 汇总、对比报告
    ├── song_adaptive.py        # 自适应策略选择（review scope / missed strategy）
    ├── song_adaptive_cost.py   # 自适应成本估算
    ├── song_evaluation.py      # 离线评估工具
    ├── clip_naming.py / search_tools.py / paths.py
    ├── asr.py
    ├── asr_backends/
    │   ├── faster_whisper.py
    │   ├── funasr_backend.py
    │   └── mimo_asr_backend.py
    ├── recognizers/            # song / dialogue / highlight / funny / cringe / daily_summary
    │   ├── base.py             # BaseRecognizer + post_process 钩子
    │   └── song.py             # SongRecognizer（legacy / V3）
    ├── song_postprocess/       # 歌曲后处理流水线
    │   ├── normalize.py        # 同名合并、副歌感知拆分、通用规范化
    │   ├── review.py           # LLM 复核（local / full scope）
    │   ├── recheck.py          # 遗漏复查（windowed / full_transcript / anchor）
    │   ├── temporal.py         # 时序裁决（全量 ASR 边界修正）
    │   ├── risk.py             # 风险评分、边界修复、anchor 扩展
    │   ├── pipeline.py         # 共享流水线组件（BoundaryRiskStage、FinalAdjudicationStage 等）
    │   └── v3.py               # V3 三轮对象协议与降级处理
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
