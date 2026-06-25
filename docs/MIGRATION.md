# 配置迁移指南

## 为什么改为模块化结构

旧版将所有配置塞进单个例配置文件（主配置 + 各域 + profiles 全在一起），
外加单独的 `cut_copy.example.conf` 和 `streamer_dictionary.json`。随着功能增加，单文件迅速膨胀到 500+ 行，难以维护和对比差异。
（旧格式示例见 `tests/config.example.yaml`。）

新版把每个配置域拆成独立 YAML，通过 `!include` 在主文件中引用。你可以按需修改单个文件，不影响其他域。也方便 `example/` 官方模板与 `local/` 实际配置分开管理。

## 新旧目录对比

**旧结构（不再支持）：**

```
dd-clip-miner-llm/
├── tests/config.example.yaml       # 旧格式示例（500+ 行单文件）
├── tests/cut_copy.example.conf     # 录播自动处理工作流（独立 ini 格式）
└── tests/streamer_dictionary.example.json  # 主播词典示例
```

**新结构：**

```
dd-clip-miner-llm/
└── config/
    ├── example/                # 官方示例模板（只读，不直接编辑）
    │   ├── main.yaml           # 主入口，用 !include 引用各域文件
    │   ├── audio.yaml
    │   ├── asr.yaml
    │   ├── llm.yaml
    │   ├── content_types.yaml
    │   ├── song.yaml
    │   ├── output.yaml
    │   ├── padding.yaml
    │   ├── dialogue.yaml
    │   ├── highlight.yaml
    │   ├── funny.yaml
    │   ├── cringe.yaml
    │   ├── daily_summary.yaml
    │   ├── cut_copy.yaml
    │   ├── profiles/           # 每个 profile 一个文件
    │   │   ├── accuracy.yaml
    │   │   ├── kv_optimized.yaml
    │   │   └── kv_v2.yaml
    │   └── streamer_dictionary.json
    └── local/                  # 你的本地配置（.gitignore 忽略）
        └── main.yaml
```

## 快速迁移（推荐）

一行命令自动将旧单文件转为新模块化格式：

```powershell
cd path\to\dd-clip-miner-llm
python scripts/migrate_config.py tests/config.example.yaml --output config/local/
```

**可选参数：**

| 参数 | 作用 |
|------|------|
| `--dry-run` | 只预览将要生成的文件，不写入磁盘 |
| `--overwrite` | 覆盖 `config/local/` 下已有文件（默认跳过） |

示例：先预览再执行：

```powershell
python scripts/migrate_config.py tests/config.example.yaml --output config/local/ --dry-run
python scripts/migrate_config.py tests/config.example.yaml --output config/local/
```

迁移脚本还会自动复制同目录下的 `streamer_dictionary.json` 和 `cut_copy.conf`（如果存在）。

## 手动迁移（不放心自动脚本时）

1. 复制官方模板（`config/example/main.yaml` 为入口）作为起点：

```powershell
xcopy config\example config\local /E /I
```

2. 将旧 `tests/config.example.yaml` 中的各域配置逐一填入 `config/local/` 下的对应文件。
   对照表：旧 YAML 顶层键 → 新文件：

   | 旧顶层键 | 新文件 |
   |----------|--------|
   | `audio:` | `config/local/audio.yaml` |
   | `asr:` | `config/local/asr.yaml` |
   | `llm:` | `config/local/llm.yaml` |
   | `song:` | `config/local/song.yaml` |
   | `output:` | `config/local/output.yaml` |
   | `padding:` | `config/local/padding.yaml` |
   | `cut_copy:` | `config/local/cut_copy.yaml` |
   | `profiles:` | `config/local/profiles/` 下每个 profile 一个文件 |

3. `profiles` 段拆为 `config/local/profiles/<name>.yaml`，每个文件内容直接是该 profile 的配置字典。

4. 复制 `tests/cut_copy.example.conf` 为 `config/local/cut_copy.yaml`（注意格式从 ini 改为 YAML）。

5. 复制 `streamer_dictionary.json` 到 `config/local/`。

6. 在 `config/local/main.yaml` 中确认 `default_profile` 和你使用的 profile 名称一致。

## 验证迁移是否成功

运行任意命令检查配置是否正确加载：

```powershell
python -m dd_clip_miner_llm run "D:\videos\live.mp4" --config config/local/main.yaml --dry-run
```

没有报错说明配置加载成功。也可以用：

```powershell
python -c "from pathlib import Path; import yaml; c = yaml.safe_load(Path('config/local/main.yaml').read_text()); print('OK:', list(c.keys())[:5])"
```

迁移脚本还会在运行结束时自动执行 round-trip 校验，对比新文件与原始数据是否一致。

## 常见问题

### 我的自定义 profile 怎么办？

迁移脚本会自动将 `profiles` 段拆分为 `config/local/profiles/<name>.yaml`，数据完全保留。

### `cut_copy.conf` 呢？

脚本会从旧配置所在目录寻找 `cut_copy.conf` 并复制到输出目录。新版统一使用 YAML 格式的 `cut_copy.yaml`，内容与旧 `.conf` 不同，建议参考 `config/example/cut_copy.yaml` 重新填写。

### `streamer_dictionary.json` 呢？

脚本会自动复制同目录下的 `streamer_dictionary.json` 到 `config/local/`。如果没有被复制，手动复制即可：

```powershell
copy streamer_dictionary.json config/local/
```

### `config.daily-summary.example.yaml` 这种旧文件呢？

旧版不存在官方模板中的 `daily_summary.yaml`（它在新版才独立为一个域）。迁移脚本只处理你给的那个单文件，不会涉及其他旧杂文件。手动迁移时，从 `config/example/daily_summary.yaml` 复制并按需修改即可。

### 还能继续用单文件吗？

不能。`load_config()` 会检测并拒绝旧格式单文件，提示参考本文档。你必须迁移到模块化结构。

### 如何回滚？

模块化配置不影响运行产物。你的 `runs/` 目录、切片、报告都不受影响。想回退到旧版需要先还原旧文件（但旧文件不再被支持）。建议先在分支或备份目录操作：

```powershell
git stash
```

## 故障排查

| 现象 | 原因与处理 |
|------|-----------|
| `migrate_config.py` 报 `PyYAML is required` | 运行 `pip install PyYAML` |
| 加载配置报 `!include` 未知标签 | 确认用的新版 `load_config()`（`dd_clip_miner_llm/config.py`），旧版 pyyaml 不认识 `!include` |
| `config/local/` 下缺少文件 | 用 `--dry-run` 预览后再加 `--overwrite` 重新运行迁移脚本 |
| 运行时报 `KeyError` | 某个域文件缺失或 YAML 格式错误，对照 `config/example/` 下的同名文件检查 |
