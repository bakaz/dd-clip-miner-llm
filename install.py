"""智能安装脚本 - 自动检测系统环境并安装依赖

核心步骤为 `pip install -e .`（依赖定义见 pyproject.toml）。可选：
  - `.[funasr]`  FunASR / Qwen3-ASR
  - `.[test]`    pytest
  - requirements-cu12.txt  faster-whisper GPU

用法：
    python install.py                    # 自动检测并安装
    python install.py --config install.yaml  # 使用配置文件
    python install.py --check            # 只检查环境，不安装
    python install.py --dev              # 含测试工具
    python install.py --gpu cuda12       # 指定 GPU 类型
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SystemInfo:
    """系统环境信息"""
    os_name: str = ""
    os_version: str = ""
    python_version: str = ""
    has_cuda: bool = False
    cuda_version: str | None = None
    gpu_name: str | None = None
    gpu_memory_mb: int | None = None
    has_ffmpeg: bool = False
    ffmpeg_version: str | None = None
    has_ffprobe: bool = False
    has_mkvmerge: bool = False
    ram_mb: int = 0


@dataclass
class InstallConfig:
    """安装配置"""
    # ASR 后端
    asr_backend: str = "faster_whisper"  # faster_whisper | funasr
    
    # GPU 支持
    gpu_type: str = "auto"  # auto | cuda12 | cpu
    
    # 可选组件
    install_funasr: bool = False
    install_dev_tools: bool = False
    
    # MKVToolNix
    install_mkvmerge: bool = True
    
    # 跳过已安装
    skip_installed: bool = True


def detect_system() -> SystemInfo:
    """检测系统环境"""
    info = SystemInfo()
    
    # OS 信息
    info.os_name = platform.system()
    info.os_version = platform.version()
    info.python_version = platform.python_version()
    
    # GPU 检测
    info.has_cuda, info.cuda_version, info.gpu_name, info.gpu_memory_mb = _detect_gpu()
    
    # FFmpeg 检测
    info.has_ffmpeg, info.ffmpeg_version = _detect_ffmpeg()
    info.has_ffprobe = _detect_ffprobe()
    info.has_mkvmerge = _detect_mkvmerge()
    
    # RAM
    info.ram_mb = _detect_ram()
    
    return info


def _detect_gpu() -> tuple[bool, str | None, str | None, int | None]:
    """检测 GPU 信息"""
    # 尝试 nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 3:
                gpu_name = parts[0].strip()
                memory_str = parts[1].strip().replace(" MiB", "")
                try:
                    memory_mb = int(memory_str)
                except ValueError:
                    memory_mb = None
                
                # 检测 CUDA 版本
                cuda_result = subprocess.run(
                    ["nvcc", "--version"],
                    capture_output=True, text=True, timeout=10
                )
                cuda_version = None
                if cuda_result.returncode == 0:
                    for line in cuda_result.stdout.splitlines():
                        if "release" in line:
                            cuda_version = line.split("release")[-1].strip().split(",")[0]
                            break
                
                return True, cuda_version, gpu_name, memory_mb
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # 尝试通过 PowerShell 检测 Intel/AMD GPU
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                ["powershell", "-Command", "Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM | ConvertTo-Json"],
                capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                if isinstance(data, dict):
                    data = [data]
                # 优先选择有 AdapterRAM 的非虚拟 GPU
                for gpu in data:
                    name = gpu.get("Name", "")
                    ram = gpu.get("AdapterRAM")
                    if not name or "Remote" in name or "Virtual" in name:
                        continue
                    if ram and ram > 0:
                        memory_mb = ram // (1024 * 1024)
                        return False, None, name, memory_mb
                # 如果没有找到有 RAM 的，返回第一个非虚拟 GPU
                for gpu in data:
                    name = gpu.get("Name", "")
                    if name and "Remote" not in name and "Virtual" not in name:
                        return False, None, name, None
        except Exception:
            pass
    
    return False, None, None, None


def _detect_ffmpeg() -> tuple[bool, str | None]:
    """检测 FFmpeg"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            version_line = result.stdout.splitlines()[0] if result.stdout else ""
            return True, version_line
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False, None


def _detect_ffprobe() -> bool:
    """检测 ffprobe"""
    try:
        result = subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _detect_mkvmerge() -> bool:
    """检测 mkvmerge"""
    # 检查 PATH
    try:
        result = subprocess.run(
            ["mkvmerge", "--version"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # 检查常见安装路径
    common_paths = [
        r"C:\Program Files\MKVToolNix\mkvmerge.exe",
        r"C:\Program Files (x86)\MKVToolNix\mkvmerge.exe",
        "/usr/bin/mkvmerge",
        "/usr/local/bin/mkvmerge",
    ]
    for path in common_paths:
        if Path(path).exists():
            return True
    
    return False


def _detect_ram() -> int:
    """检测 RAM (MB)"""
    try:
        import psutil
        return psutil.virtual_memory().total // (1024 * 1024)
    except ImportError:
        pass
    
    # Windows fallback
    if platform.system() == "Windows":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return int(stat.ullTotalPhys) // (1024 * 1024)
        except Exception:
            pass
    
    return 0


def load_install_config(config_path: str | Path) -> InstallConfig:
    """加载安装配置文件"""
    config = InstallConfig()
    
    try:
        import yaml
    except ImportError:
        return config
    
    path = Path(config_path)
    if not path.exists():
        return config
    
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    
    config.asr_backend = data.get("asr_backend", config.asr_backend)
    config.gpu_type = data.get("gpu_type", config.gpu_type)
    config.install_funasr = data.get("install_funasr", config.install_funasr)
    config.install_dev_tools = data.get("install_dev_tools", config.install_dev_tools)
    config.install_mkvmerge = data.get("install_mkvmerge", config.install_mkvmerge)
    config.skip_installed = data.get("skip_installed", config.skip_installed)
    
    return config


def _install_mkvmerge() -> bool:
    """尝试安装 MKVToolNix（视频拼接依赖 mkvmerge）。"""
    if _detect_mkvmerge():
        return True
    if platform.system() == "Windows":
        print("  尝试通过 winget 安装 MKVToolNix...")
        try:
            result = subprocess.run(
                ["winget", "install", "MoritzBunkus.MKVToolNix", "--accept-package-agreements"],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0 and _detect_mkvmerge():
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        print("  请手动安装: winget install MoritzBunkus.MKVToolNix")
        print("  下载: https://mkvtoolnix.download/downloads.html#windows")
    else:
        print("  请通过系统包管理器安装 mkvtoolnix，例如:")
        print("    Debian/Ubuntu: sudo apt install mkvtoolnix")
        print("    macOS: brew install mkvtoolnix")
    return False


def generate_install_plan(info: SystemInfo, config: InstallConfig) -> list[dict[str, Any]]:
    """生成安装计划"""
    steps = []

    if config.install_mkvmerge:
        steps.append({
            "name": "MKVToolNix (mkvmerge)",
            "check": _detect_mkvmerge,
            "custom_install": _install_mkvmerge,
        })
    
    # 1. 核心依赖（editable install，元数据见 pyproject.toml）
    steps.append({
        "name": "核心依赖",
        "command": [sys.executable, "-m", "pip", "install", "-e", "."],
        "check": lambda: _check_package("dd_clip_miner_llm"),
    })
    
    # 2. GPU 支持
    gpu_type = config.gpu_type
    if gpu_type == "auto":
        if info.has_cuda and info.cuda_version and info.cuda_version.startswith("12"):
            gpu_type = "cuda12"
        else:
            gpu_type = "cpu"
    
    if gpu_type == "cuda12":
        steps.append({
            "name": "CUDA 12 支持",
            "command": [sys.executable, "-m", "pip", "install", "-r", "requirements-cu12.txt"],
            "check": lambda: _check_package("ctranslate2"),
        })
    
    # 3. FunASR 后端
    if config.asr_backend == "funasr" or config.install_funasr:
        steps.append({
            "name": "FunASR 后端",
            "command": [sys.executable, "-m", "pip", "install", "-e", ".[funasr]"],
            "check": lambda: _check_package("funasr"),
        })
    
    # 4. 开发/测试工具
    if config.install_dev_tools:
        steps.append({
            "name": "开发/测试工具",
            "command": [sys.executable, "-m", "pip", "install", "-e", ".[test]"],
            "check": lambda: _check_package("pytest"),
        })
    
    return steps


def _check_package(package_name: str) -> bool:
    """检查包是否已安装"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False


def execute_install_plan(steps: list[dict[str, Any]], skip_installed: bool = True) -> bool:
    """执行安装计划"""
    print("\n" + "="*60)
    print("开始安装")
    print("="*60)
    
    for i, step in enumerate(steps, 1):
        name = step["name"]
        command = step.get("command")
        check = step.get("check")
        custom_install = step.get("custom_install")
        
        # 检查是否已安装
        if skip_installed and check and check():
            print(f"\n[{i}/{len(steps)}] {name} - 已安装，跳过")
            continue
        
        print(f"\n[{i}/{len(steps)}] {name}")
        if custom_install:
            if not custom_install():
                print("  ✗ 安装失败")
                return False
            print("  ✓ 安装成功")
            continue
        if not command:
            print("  ✗ 缺少安装命令")
            return False
        print(f"  命令: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                capture_output=False,
                text=True,
                timeout=600,
            )
            if result.returncode == 0:
                print(f"  ✓ 安装成功")
            else:
                print(f"  ✗ 安装失败 (返回码: {result.returncode})")
                return False
        except subprocess.TimeoutExpired:
            print(f"  ✗ 安装超时")
            return False
        except Exception as e:
            print(f"  ✗ 安装异常: {e}")
            return False
    
    print("\n" + "="*60)
    print("安装完成")
    print("="*60)
    return True


def print_system_info(info: SystemInfo) -> None:
    """打印系统信息"""
    print("\n" + "="*60)
    print("系统环境检测")
    print("="*60)
    print(f"操作系统: {info.os_name} {info.os_version}")
    print(f"Python: {info.python_version}")
    print(f"RAM: {info.ram_mb} MB")
    print()
    if info.gpu_name:
        print(f"GPU: {info.gpu_name}")
        if info.gpu_memory_mb:
            print(f"  显存: {info.gpu_memory_mb} MB")
        if info.has_cuda:
            print(f"  CUDA: {info.cuda_version}")
        else:
            print(f"  类型: 非 NVIDIA (无 CUDA)")
    else:
        print("GPU: 未检测到")
    print()
    print(f"FFmpeg: {'[OK]' if info.has_ffmpeg else '[NO]'}")
    if info.has_ffmpeg:
        print(f"  版本: {info.ffmpeg_version}")
    print(f"ffprobe: {'[OK]' if info.has_ffprobe else '[NO]'}")
    print(f"mkvmerge: {'[OK]' if info.has_mkvmerge else '[NO]'}")


def main():
    parser = argparse.ArgumentParser(description="智能安装脚本")
    parser.add_argument("--config", help="安装配置文件路径")
    parser.add_argument("--check", action="store_true", help="只检查环境，不安装")
    parser.add_argument("--gpu", choices=["auto", "cuda12", "cpu"], default="auto", help="GPU 类型")
    parser.add_argument("--asr", choices=["faster_whisper", "funasr"], default="faster_whisper", help="ASR 后端")
    parser.add_argument("--funasr", action="store_true", help="安装 FunASR 支持")
    parser.add_argument("--no-funasr", action="store_true", help="不安装 FunASR 支持")
    parser.add_argument("--dev", action="store_true", help="安装开发工具")
    parser.add_argument("--no-mkvmerge", action="store_true", help="不安装 MKVToolNix")
    parser.add_argument("--force", action="store_true", help="强制重新安装已安装的包")
    
    args = parser.parse_args()
    
    # 检测系统
    print("正在检测系统环境...")
    info = detect_system()
    print_system_info(info)
    
    # 加载配置
    config = InstallConfig()
    if args.config:
        config = load_install_config(args.config)
    
    # 命令行参数覆盖
    if args.gpu != "auto":
        config.gpu_type = args.gpu
    if args.asr:
        config.asr_backend = args.asr
    if args.funasr:
        config.install_funasr = True
    if args.no_funasr:
        config.install_funasr = False
    if args.dev:
        config.install_dev_tools = True
    if args.no_mkvmerge:
        config.install_mkvmerge = False
    if args.force:
        config.skip_installed = False
    
    # 生成安装计划
    steps = generate_install_plan(info, config)
    
    # 打印安装计划
    print("\n" + "="*60)
    print("安装计划")
    print("="*60)
    for i, step in enumerate(steps, 1):
        name = step["name"]
        check = step.get("check")
        status = "已安装" if check and check() else "待安装"
        print(f"{i}. {name} [{status}]")
    
    # 只检查模式
    if args.check:
        print("\n检查完成。")
        return
    
    # 执行安装
    success = execute_install_plan(steps, config.skip_installed)
    
    if success:
        print("\n✓ 所有组件安装成功！")
        print("\n下一步：")
        print("  1. 复制配置文件: copy config.example.yaml config.yaml")
        print("  2. 设置 API key: $env:LLM_API_KEY='your-key'")
        print("  3. 运行: python -m dd_clip_miner_llm run video.mp4 --config config.yaml")
    else:
        print("\n✗ 安装过程中出现错误，请检查日志。")
        sys.exit(1)


if __name__ == "__main__":
    main()
