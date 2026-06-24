# dd-clip-miner-llm

基于 ASR + LLM 的直播内容挖掘工具。支持从直播录像中识别和提取：

- 歌曲片段
- 有趣对话
- 高能时刻
- 搞笑片段
- 下头对话
- 当天直播结构化总结（仅报告，不切片）

默认配置偏向歌曲切片和当天总结：`content_types.song: true`、`daily_summary: true`，对话/高能/搞笑/下头片段默认关闭，需要时可在配置或 `--content-types` 中开启。

**完全兼容** [dd-song-miner-llm](https://github.com/bakaz/dd-song-miner-llm) 的配置和工作流程。

## 特性

- **可插拔识别器**：每种内容类型独立实现（`recognizers/`）
- **多 ASR 后端**：faster-whisper（批量推理 `BatchedInferencePipeline`）、FunASR / Qwen3-ASR、远程 MiMo ASR
- **智能 LLM**：reasoning followup、工具调用、JSON 修复，可选歌词搜索
- **Provider 路由与重试**：`provider_route` 按名称顺序 fallback；`timeout_schedule` 逐步升级超时；传输异常与产物错误分离重试；`_call_llm_raw` 唯一底层请求入口，测试可拦截
- **KV 缓存优化**：`cache_friendly_prompt_layout` 复用 ASR 前缀，`compact_segment_ranges` 减少输出 token
- **kv_v2 优化流水线**：开口哼唱检测、小簇跳过、高置信度已知曲名保护、低阈值未知歌曲保留
- **三轮分段流水线**：高精度发现 → 未覆盖召回审计 → 全量时序裁决（`song_postprocess/song_kv/` 实现）
- **时序裁决**：全量 ASR 二次审视，修正首轮边界，支持名称保留
- **副歌感知拆分**：40–130 秒间隔根据文本相似度判断是否为副歌重现
- **同名相邻合并**：排序后字面相邻、标题相同且间隔 ≤ 40 秒的候选自动合并
- **搜索验证命名**：可对未知歌曲用歌词锚点搜索，需歌词证据才更新名称；默认关闭
- **未知歌曲合并**：相邻未知歌曲按时间间隔（≤ 40 秒）和 ASR 文本相似度（≥ 0.3）自动合并，被合并的原始片段导出到 `sus/` 文件夹供人工审核
- **锚点漏检审计**：可选的 anchor-based 补查，`kv_optimized` 可用作未覆盖区间召回审计
- **断点续传**：复用 `01_audio`、`02_asr`、LLM 结果（`progress.json`）
- **批量 + 多段合并**：`ConcatPipeline` 处理直播分段 H.264 损坏（mkvmerge 优先 + 6 策略 fallback）
- **切片命名**：主播词典 + 路径日期 → `【主播】歌名-歌手-YYMMDD`
- **手动重切**：改 CSV 后 `manual-cut`
- **拖拽重切合并**：歌曲导出目录内的 `merge_mp4.bat` 可把两个已导出片段从原始 input/concat 重新切成一个片段

## 工作流程

1. FFmpeg 提取 16 kHz 单声道 WAV
2. ASR 转写为带时间戳的 segment
3. 各识别器送 LLM 标注片段
4. 切割音频/视频到 `03_clips/`
5. 生成 `04_reports/` 下 CSV/JSON，可 `manual-cut` 重切

## 快速开始

```powershell
cd path\to\dd-clip-miner-llm
python install.py --gpu cuda13 --funasr   # GPU 生产管线；无 GPU 可用 python install.py

copy config.example.yaml config.yaml
$env:OPENCODE_API_KEY = "<your-api-key>"      # 或设置 DEEPSEEK_API_KEY / MIMO_API_KEY

python -m dd_clip_miner_llm run "D:\videos\live.mp4" --config config.yaml
```

也可使用入口脚本：`dd-clip-miner-llm run ...`（`pip install -e .` 后可用）。

## 安装

需要 **Python 3.10-3.12**（`pyproject.toml`）。

### 推荐：`install.py`

```powershell
python install.py                          # 自动检测并安装
python install.py --config install.yaml    # 使用配置文件
python install.py --check                  # 只检测环境，不安装
python install.py --dev                    # 含 pytest
python install.py --gpu cuda13             # 指定 CUDA 13 (Blackwell / RTX 50xx)
python install.py --gpu cuda12             # 指定 CUDA 12 (Ampere / Ada)
```

自动检测 FFmpeg / mkvmerge / GPU，执行 `pip install -e .`，可选安装 `[funasr]`、`requirements-cu13.txt` / `requirements-cu12.txt`、`[test]`。Blackwell GPU (RTX 50xx, CC ≥ 12.0) 默认走 cu130；Ampere/Ada (CC 8.x–9.x) 走 cu121。

### 手动安装

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .                           # 核心依赖
pip install -e ".[test]"                   # 可选：pytest
pip install -e ".[funasr]"                 # 可选：FunASR（会拉 CPU torch）

# GPU 支持（二选一，按显卡架构）：
pip install -r requirements-cu13.txt       # Blackwell / RTX 50xx (CUDA 13)
pip install -r requirements-cu12.txt       # Ampere / Ada (CUDA 12)
```

> **注意**：`[funasr]` 会从 PyPI 拉 CPU-only torch。先装 CUDA torch 再装 funasr 可避免被覆盖（`install.py` 已处理此顺序）。手动安装时建议先 `pip install -r requirements-cu13.txt`，再 `pip install -e ".[funasr]"`。

### 系统依赖

| 组件 | 用途 | Windows 安装 |
|------|------|--------------|
| FFmpeg + ffprobe | 抽音频、切片、探测 | `winget install Gyan.FFmpeg` |
| mkvmerge | 多段合并（可选，更稳） | `winget install MKVToolNix` |
| libsndfile | soundfile（Linux CI 需 `libsndfile1`） | 一般随环境已有 |

无 mkvmerge 时合并回退纯 FFmpeg。无 CUDA DLL 时 faster-whisper 回退 CPU。

`setup.py` 仅为 setuptools 入口（`pip install -e .` 需要）。

## 配置

复制模板并按需修改：

```powershell
copy config.example.yaml config.yaml
python -m dd_clip_miner_llm init-config --out config.yaml
```

| 文件 | 说明 |
|------|------|
| `config.example.yaml` | 主配置模板（含注释） |
| `config.daily-summary.example.yaml` | 仅当天总结 |
| `cut_copy.example.conf` | 录播自动处理工作流（复制为 `cut_copy.conf`） |
| `streamer_dictionary.example.json` | 主播词典 |

**勿提交**（已在 `.gitignore`）：`config.yaml`、`cut_copy.conf`、`streamer_dictionary.json`、`runs/`。

ASR 使用 `asr.mode: local | remote` 新结构（见 `config.example.yaml`）：

- **硬件自动分流**（`device: auto` + `local.gpu` / `local.cpu`）：
  - **GPU（默认 backend: qwen3_asr）**：Qwen3-1.7B + chunk180 + funasr fallback；FW **turbo** 供对比/fallback
  - **CPU（自动切 backend: faster_whisper）**：FW **small** + `int8` batch 主路径 + standard fallback
  - 安装 GPU 生产管线：`python install.py --gpu cuda13 --funasr`
- 二轮 fallback 结果会写回 `02_asr/transcript.json`，并保留 `transcript_primary.json`、`fallback_ranges.json`、`fallback_segments.json` 供审计。

**硬件分流**：`funasr`/`faster_whisper` 设 `device: auto`，并在 `local.gpu` / `local.cpu` 下分别写硬件专用配置（见 `config.example.yaml`）。无 `gpu:`/`cpu:` 节时，FW 在无 CUDA 时自动将 `compute_type` 降为 `int8`。

LLM Key 优先环境变量：默认模板使用 `OPENCODE_API_KEY`、`DEEPSEEK_API_KEY`、`MIMO_API_KEY`，也可在 provider 的 `api_key_env` 中改成自己的变量名。

### LLM Provider 路由与重试

```yaml
llm:
  provider_route: [opencode, deepseek, mimo]   # fallback 顺序
  providers:
    opencode:
      api_key_env: OPENCODE_API_KEY
      base_url: https://opencode.ai/zen/go/v1
      model: deepseek-v4-flash
      timeout_schedule: [60, 120, 180]    # 逐步升级超时（秒）
      retry_backoff_seconds: [2, 5]      # 退避间隔（索引对应传输轮次）
      retry_jitter_ratio: 0.25           # 抖动比例（0~1），防雪崩
      result_retries: 2                  # 产物无效时重放次数
      # proxy: http://127.0.0.1:7890     # HTTP/HTTPS/SOCKS5 代理
```

**行为规则**：

- `timeout_schedule` 长度决定传输尝试次数（`[60, 120, 180]` = 3 次）
- 传输失败（网络/超时/429/5xx）按 `retry_backoff_seconds` + 随机抖动等待后重试
- 产物失败（空响应/无效 JSON/字段缺失/业务校验不通过）消耗 `result_retries`，从头重放
- 401/403/400 等不可重试异常立即结束当前 provider，切换下一个
- `provider_route` 全部耗尽后才报错；未配置时回退到 `active_provider`
- 每个 provider 有独立的超时、退避和产物重放参数
- 服务返回 `Retry-After` 时优先遵循，上限 60 秒
- OpenAI 客户端按 `(base_url, api_key, proxy)` 缓存，相同密钥不同代理不会复用
- 支持 per-provider `proxy` 配置（HTTP/HTTPS/SOCKS5）
- 支持 `stream: true` 流式接收，代理截断时保留部分内容，配合续写机制补全

**向后兼容**：仅配置 `timeout: 300` + `max_retries: 3` 时保持旧行为（固定超时、指数退避）。

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

`config.example.yaml` 默认使用 `kv_optimized` profile，并默认启用 `song` 与 `daily_summary`。其他内容类型可通过配置或 `--content-types song,dialogue` 临时开启。

### 批量

```powershell
python -m dd_clip_miner_llm batch-run "D:\input" --config config.yaml --work-root "D:\work" --result-root "D:\results"
python -m dd_clip_miner_llm batch-run "D:\input" --config config.yaml --work-root "D:\work" --result-root "D:\results" --concat
python -m dd_clip_miner_llm batch-run "D:\input" --config config.yaml --profile kv_optimized --work-root "D:\work" --result-root "D:\results"
```

配置包含 `profiles` 时，音频和 ASR 由两个 profile 共享，LLM、切片和报告分别写入
`02_asr/llm/<profile>`、`03_clips/<profile>`、`04_reports/<profile>`。
默认 `kv_optimized` 使用
`risk_routed_kv`、缓存友好布局和三轮对象协议。时长、边界膨胀和重叠只作为复核风险，
不会被当作全局硬过滤条件。`accuracy` 保留 task-first 和 `segment_indices`，适合做对照。
两套 profile 共享歌曲 padding，默认 `merge_gap_seconds: 40`。`accuracy` 显式使用本地 review 和 windowed missed-recheck；
`kv_optimized` 执行 Precision Discovery、Recall Audit 和 Segmentation Adjudication。
第一轮只保留明确演唱，第二轮只输出未覆盖区间的短证据，第三轮统一修边界并可有限补漏。
默认 `song.search.enabled: false`，不会联网搜索歌词；需要搜索验证命名时可打开 `song.search.enabled` 或 profile 内的 `song.search.enabled`。
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

### 拖拽重切合并

歌曲导出目录会自动写入 `merge_mp4.bat` 和 `merge_recut_context.json`。把同一目录下两个 `.mp4` 拖到 bat 上，会从原始 input/concat 重新切出一个 `.mp4`；把两个 `.mp3` 拖到 bat 上，会重新切出一个 `.mp3`。输出写回拖入文件所在目录并自动避让重名，不会覆盖原片段。

也可以直接调用：

```powershell
python -m dd_clip_miner_llm post-merge --context "...\merge_recut_context.json" "clip1.mp4" "clip2.mp4"
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
| `post-merge` | 从两个已导出歌曲片段反查 ASR 并重新切为一个片段 |
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
│   ├── fix_garbled_clip_names.py
│   ├── evaluate_song_pipeline_v2.py
│   ├── adaptive_cost_probe.py
│   ├── review_scope_ab.py
│   └── probe_concat_strategies.ps1
├── tests/                      # 单元测试
├── .github/workflows/tests.yml
└── dd_clip_miner_llm/
    ├── cli.py / __main__.py    # python -m dd_clip_miner_llm
    ├── pipeline/               # 主流水线
    ├── batch.py / manual.py
    ├── config.py               # 配置加载、profile 管理、歌曲 pipeline 选择
    ├── models.py / report.py / merger.py / post_merge.py
    ├── llm/                    # Provider、transport、prompt、parse、tools
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
    │   └── song/               # SongRecognizer（accuracy / kv / kv_v2）
    ├── song_postprocess/       # 歌曲后处理流水线
    │   ├── normalize.py        # 同名合并、未知歌曲合并、副歌感知拆分、通用规范化
    │   ├── review.py           # LLM 复核（local / full scope）
    │   ├── recheck.py          # 遗漏复查（windowed / full_transcript / anchor）
    │   ├── temporal.py         # 时序裁决（全量 ASR 边界修正）
    │   ├── risk.py             # 风险评分、边界修复、anchor 扩展
    │   ├── pipeline.py         # 共享流水线组件（BoundaryRiskStage、FinalAdjudicationStage 等）
    │   ├── song_kv/            # 三轮对象协议与 KV 缓存优化
    │   └── lyrics_match.py    # 歌词匹配与标题归一化
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
├── 02_asr/llm/<profile>/<type>/ # matches.json, llm_batch_*.json, ...
├── 03_clips/<profile>/
│   ├── audio/song/merge_mp4.bat # 歌曲拖拽重切合并工具
│   ├── video/song/merge_mp4.bat
│   └── video/song/sus/         # 被合并的未知歌曲原始片段（供人工审核）
├── 04_reports/<profile>/<type>/ # songs.csv, dialogues.csv, ...
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
| `cublas64_12.dll is not found` | `pip install -r requirements-cu13.txt`（Blackwell）或 `requirements-cu12.txt`（Ampere/Ada），或接受 CPU 回退 |
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
