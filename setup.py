"""傻瓜化安装脚本 - 自动检测系统、下载依赖、配置环境

功能：
1. 创建 Python 3.12 venv
2. 自动下载安装 ffmpeg/ffprobe/mkvtoolnix
3. 检测 GPU（通过 ffmpeg）
4. 根据 GPU 决定是否安装 CUDA
5. 询问是否安装 FunASR + PyTorch
6. 运行验证

用法：
    python setup.py              # 交互式安装
    python setup.py --auto       # 自动模式（使用默认配置）
    python setup.py --check      # 只检查环境
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SystemInfo:
    """系统环境信息"""
    os_name: str = ""
    os_version: str = ""
    python_version: str = ""
    python_path: str = ""
    
    # GPU
    gpu_name: str | None = None
    gpu_vendor: str | None = None  # nvidia | intel | amd | None
    gpu_memory_mb: int | None = None
    cuda_version: str | None = None
    has_cuda: bool = False
    
    # 工具
    has_ffmpeg: bool = False
    ffmpeg_path: str | None = None
    ffmpeg_version: str | None = None
    has_ffprobe: bool = False
    has_mkvmerge: bool = False
    mkvmerge_path: str | None = None
    
    # RAM
    ram_mb: int = 0


@dataclass
class SetupConfig:
    """安装配置"""
    python_version: str = "3.12"
    venv_path: str = ".venv"
    
    # 工具安装
    install_ffmpeg: bool = True
    install_mkvmerge: bool = True
    
    # ASR
    install_funasr: bool = False
    install_cuda: bool = False
    
    # 跳过已安装
    skip_installed: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# 系统检测
# ═══════════════════════════════════════════════════════════════════════════════


def detect_system() -> SystemInfo:
    """检测系统环境"""
    info = SystemInfo()
    
    # OS 信息
    info.os_name = platform.system()
    info.os_version = platform.version()
    info.python_version = platform.python_version()
    info.python_path = sys.executable
    
    # RAM
    info.ram_mb = _detect_ram()
    
    # GPU（通过 ffmpeg 检测）
    info.gpu_name, info.gpu_vendor, info.gpu_memory_mb, info.has_cuda, info.cuda_version = _detect_gpu_via_ffmpeg()
    
    # FFmpeg
    info.has_ffmpeg, info.ffmpeg_path, info.ffmpeg_version = _detect_ffmpeg()
    info.has_ffprobe = _detect_ffprobe()
    
    # mkvmerge
    info.has_mkvmerge, info.mkvmerge_path = _detect_mkvmerge()
    
    return info


def _detect_ram() -> int:
    """检测 RAM (MB)"""
    if platform.system() == "Windows":
        try:
            import ctypes
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
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return int(stat.ullTotalPhys) // (1024 * 1024)
        except Exception:
            pass
    return 0


def _detect_gpu_via_ffmpeg() -> tuple[str | None, str | None, int | None, bool, str | None]:
    """通过 ffmpeg 检测 GPU"""
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return None, None, None, False, None
    
    # 获取 ffmpeg 编码器列表
    try:
        result = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return None, None, None, False, None
        
        encoders_text = result.stdout + result.stderr
        
        # 检测 NVIDIA（需要验证 GPU 实际存在）
        if "h264_nvenc" in encoders_text:
            gpu_name, gpu_mem, cuda_ver = _detect_nvidia_details()
            if gpu_name:  # nvidia-smi 成功才确认
                return gpu_name, "nvidia", gpu_mem, True, cuda_ver
        
        # 检测 Intel QSV（需要验证 GPU 实际存在）
        if "h264_qsv" in encoders_text:
            gpu_name = _detect_intel_gpu()
            if gpu_name:
                return gpu_name, "intel", None, False, None
        
        # 检测 AMD AMF（需要验证 GPU 实际存在）
        if "h264_amf" in encoders_text:
            gpu_name = _detect_amd_gpu()
            if gpu_name:
                return gpu_name, "amd", None, False, None
            
    except Exception:
        pass
    
    return None, None, None, False, None


def _detect_nvidia_details() -> tuple[str | None, int | None, str | None]:
    """检测 NVIDIA GPU 详情"""
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
                
                # CUDA 版本
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
                
                return gpu_name, memory_mb, cuda_version
    except Exception:
        pass
    return None, None, None


def _detect_intel_gpu() -> str | None:
    """检测 Intel GPU"""
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                ["powershell", "-Command", 
                 "Get-CimInstance Win32_VideoController | Where-Object {$_.Name -like '*Intel*'} | Select-Object -First 1 -ExpandProperty Name"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass
    return "Intel GPU"


def _detect_amd_gpu() -> str | None:
    """检测 AMD GPU"""
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                ["powershell", "-Command",
                 "Get-CimInstance Win32_VideoController | Where-Object {$_.Name -like '*AMD*' -or $_.Name -like '*Radeon*'} | Select-Object -First 1 -ExpandProperty Name"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass
    return "AMD GPU"


def _detect_ffmpeg() -> tuple[bool, str | None, str | None]:
    """检测 FFmpeg"""
    path = shutil.which("ffmpeg")
    if path:
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                version_line = result.stdout.splitlines()[0] if result.stdout else ""
                return True, path, version_line
        except Exception:
            pass
    return False, None, None


def _detect_ffprobe() -> bool:
    """检测 ffprobe"""
    path = shutil.which("ffprobe")
    return path is not None


def _detect_mkvmerge() -> tuple[bool, str | None]:
    """检测 mkvmerge"""
    # 检查 PATH
    path = shutil.which("mkvmerge")
    if path:
        return True, path
    
    # 检查常见安装路径
    common_paths = [
        r"C:\Program Files\MKVToolNix\mkvmerge.exe",
        r"C:\Program Files (x86)\MKVToolNix\mkvmerge.exe",
        "/usr/bin/mkvmerge",
        "/usr/local/bin/mkvmerge",
    ]
    for p in common_paths:
        if Path(p).exists():
            return True, p
    
    return False, None


# ═══════════════════════════════════════════════════════════════════════════════
# 工具安装
# ═══════════════════════════════════════════════════════════════════════════════


def add_to_path(directory: str) -> None:
    """添加目录到 PATH"""
    if platform.system() == "Windows":
        # Windows: 设置用户 PATH
        current_path = os.environ.get("PATH", "")
        if directory not in current_path:
            os.environ["PATH"] = directory + ";" + current_path
            print(f"  已添加到 PATH: {directory}")
    else:
        # Unix: 写入 .bashrc/.zshrc
        shell_rc = os.path.expanduser("~/.bashrc")
        if os.path.exists(os.path.expanduser("~/.zshrc")):
            shell_rc = os.path.expanduser("~/.zshrc")
        
        export_line = f'export PATH="{directory}:$PATH"'
        with open(shell_rc, "r") as f:
            content = f.read()
        if export_line not in content:
            with open(shell_rc, "a") as f:
                f.write(f"\n{export_line}\n")
            print(f"  已添加到 {shell_rc}")


def download_file(url: str, dest: Path) -> bool:
    """下载文件"""
    try:
        import urllib.request
        print(f"  下载: {url}")
        urllib.request.urlretrieve(url, str(dest))
        return True
    except Exception as e:
        print(f"  下载失败: {e}")
        return False


def install_ffmpeg_windows(install_dir: Path) -> bool:
    """Windows 安装 FFmpeg"""
    url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    zip_path = install_dir / "ffmpeg.zip"
    
    if not download_file(url, zip_path):
        return False
    
    print("  解压中...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(install_dir)
        
        # 找到 bin 目录
        for d in install_dir.iterdir():
            if d.is_dir() and d.name.startswith("ffmpeg"):
                bin_dir = d / "bin"
                if bin_dir.exists():
                    add_to_path(str(bin_dir))
                    # 清理 zip
                    zip_path.unlink()
                    return True
    except Exception as e:
        print(f"  解压失败: {e}")
    
    return False


def install_mkvmerge_windows(install_dir: Path) -> bool:
    """Windows 安装 MKVToolNix"""
    print("  MKVToolNix 需要手动安装")
    print("  下载地址: https://mkvtoolnix.download/downloads.html#windows")
    print("  或运行: winget install MoritzBunkus.MKVToolNix")
    
    # 尝试 winget
    try:
        result = subprocess.run(
            ["winget", "install", "MoritzBunkus.MKVToolNix", "--accept-package-agreements"],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            # 添加到 PATH
            mkvmerge_path = r"C:\Program Files\MKVToolNix"
            if Path(mkvmerge_path).exists():
                add_to_path(mkvmerge_path)
            return True
    except Exception:
        pass
    
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Venv 管理
# ═══════════════════════════════════════════════════════════════════════════════


def create_venv(venv_path: str, python_version: str = "3.12") -> bool:
    """创建 Python venv"""
    venv = Path(venv_path)
    
    if venv.exists():
        print(f"  venv 已存在: {venv}")
        return True
    
    # 查找 Python 3.12
    python_cmd = _find_python(python_version)
    if not python_cmd:
        print(f"  未找到 Python {python_version}")
        return False
    
    print(f"  创建 venv: {venv}")
    result = subprocess.run(
        [python_cmd, "-m", "venv", str(venv)],
        capture_output=True, text=True
    )
    
    if result.returncode != 0:
        print(f"  创建失败: {result.stderr}")
        return False
    
    print("  venv 创建成功")
    return True


def _find_python(version: str) -> str | None:
    """查找指定版本的 Python"""
    # Windows
    if platform.system() == "Windows":
        # 尝试 py launcher
        try:
            result = subprocess.run(
                ["py", f"-{version}", "--version"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return f"py -{version}"
        except FileNotFoundError:
            pass
        
        # 尝试常见路径
        common_paths = [
            f"C:\\Python{version.replace('.', '')}\\python.exe",
            f"C:\\Users\\{os.getenv('USERNAME')}\\AppData\\Local\\Programs\\Python\\Python{version.replace('.', '')}\\python.exe",
        ]
        for p in common_paths:
            if Path(p).exists():
                return p
    
    # Unix
    for name in [f"python{version}", "python3", "python"]:
        path = shutil.which(name)
        if path:
            try:
                result = subprocess.run(
                    [path, "--version"],
                    capture_output=True, text=True
                )
                if version in result.stdout:
                    return path
            except Exception:
                continue
    
    # 回退到当前 Python
    return sys.executable


def install_packages(venv_path: str, packages: list[str], name: str = "依赖") -> bool:
    """安装包到 venv"""
    pip = _get_pip(venv_path)
    if not pip:
        return False
    
    print(f"  安装 {name}...")
    result = subprocess.run(
        [*pip, "install"] + packages,
        capture_output=False,
        text=True,
        timeout=600
    )
    
    if result.returncode == 0:
        print(f"  {name} 安装成功")
        return True
    else:
        print(f"  {name} 安装失败")
        return False


def _get_pip(venv_path: str) -> list[str] | None:
    """获取 pip 命令"""
    if platform.system() == "Windows":
        pip = Path(venv_path) / "Scripts" / "pip.exe"
    else:
        pip = Path(venv_path) / "bin" / "pip"
    
    if pip.exists():
        return [str(pip)]
    
    # 回退到 python -m pip
    python = _get_python(venv_path)
    if python:
        return [python, "-m", "pip"]
    
    return None


def _get_python(venv_path: str) -> str | None:
    """获取 Python 命令"""
    if platform.system() == "Windows":
        python = Path(venv_path) / "Scripts" / "python.exe"
    else:
        python = Path(venv_path) / "bin" / "python"
    
    if python.exists():
        return str(python)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 打印
# ═══════════════════════════════════════════════════════════════════════════════


def print_header(text: str) -> None:
    """打印标题"""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_section(text: str) -> None:
    """打印小节标题"""
    print(f"\n--- {text} ---")


def print_system_info(info: SystemInfo) -> None:
    """打印系统信息"""
    print_header("系统环境检测")
    print(f"操作系统: {info.os_name} {info.os_version}")
    print(f"Python: {info.python_version}")
    print(f"RAM: {info.ram_mb} MB")
    print()
    
    # GPU
    if info.gpu_name:
        print(f"GPU: {info.gpu_name}")
        if info.gpu_memory_mb:
            print(f"  显存: {info.gpu_memory_mb} MB")
        if info.has_cuda:
            print(f"  CUDA: {info.cuda_version}")
        else:
            print(f"  类型: {info.gpu_vendor or '未知'} (无 CUDA)")
    else:
        print("GPU: 未检测到")
    
    print()
    print(f"FFmpeg: {'[OK]' if info.has_ffmpeg else '[NO]'}")
    if info.has_ffmpeg:
        print(f"  路径: {info.ffmpeg_path}")
    print(f"ffprobe: {'[OK]' if info.has_ffprobe else '[NO]'}")
    print(f"mkvmerge: {'[OK]' if info.has_mkvmerge else '[NO]'}")
    if info.has_mkvmerge:
        print(f"  路径: {info.mkvmerge_path}")


def ask_question(question: str, default: bool = True) -> bool:
    """询问用户"""
    suffix = "[Y/n]" if default else "[y/N]"
    answer = input(f"{question} {suffix}: ").strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes", "是")


# ═══════════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════════


def run_setup(config: SetupConfig, info: SystemInfo) -> bool:
    """执行安装流程"""
    
    # 1. 安装系统工具
    print_header("1. 安装系统工具")
    
    if config.install_ffmpeg and not info.has_ffmpeg:
        print_section("安装 FFmpeg")
        if platform.system() == "Windows":
            install_dir = Path.cwd() / "tools"
            install_dir.mkdir(exist_ok=True)
            if install_ffmpeg_windows(install_dir):
                info.has_ffmpeg = True
            else:
                print("  FFmpeg 安装失败，请手动安装")
        else:
            print("  请运行: sudo apt install ffmpeg 或 brew install ffmpeg")
    
    if config.install_mkvmerge and not info.has_mkvmerge:
        print_section("安装 MKVToolNix")
        if platform.system() == "Windows":
            if install_mkvmerge_windows(Path.cwd()):
                info.has_mkvmerge = True
            else:
                print("  MKVToolNix 安装失败，请手动安装")
        else:
            print("  请运行: sudo apt install mkvtoolnix 或 brew install mkvtoolnix")
    
    # 2. 创建 venv
    print_header("2. 创建 Python 环境")
    if not create_venv(config.venv_path, config.python_version):
        return False
    
    # 3. 安装核心依赖
    print_header("3. 安装核心依赖")
    if not install_packages(config.venv_path, ["-r", "requirements.txt"], "核心依赖"):
        return False
    
    # 4. GPU/CUDA
    print_header("4. GPU 配置")
    if info.has_cuda:
        print(f"检测到 NVIDIA GPU: {info.gpu_name}")
        print(f"CUDA 版本: {info.cuda_version}")
        
        if config.install_cuda or ask_question("是否安装 CUDA 支持?"):
            print_section("安装 CUDA 依赖")
            if info.cuda_version and info.cuda_version.startswith("12"):
                install_packages(config.venv_path, ["-r", "requirements-cu12.txt"], "CUDA 12 支持")
            else:
                print(f"  当前 CUDA 版本 ({info.cuda_version}) 不支持，跳过")
    elif info.gpu_vendor == "intel":
        print(f"检测到 Intel GPU: {info.gpu_name}")
        print("  Intel GPU 不支持 CUDA，将使用 CPU 模式")
    elif info.gpu_vendor == "amd":
        print(f"检测到 AMD GPU: {info.gpu_name}")
        print("  AMD GPU 不支持 CUDA，将使用 CPU 模式")
    else:
        print("未检测到 GPU，将使用 CPU 模式")
    
    # 5. FunASR
    print_header("5. FunASR 支持")
    if config.install_funasr or ask_question("是否安装 FunASR 支持 (SenseVoiceSmall/Paraformer)?", default=False):
        print_section("安装 FunASR")
        funasr_packages = ["funasr", "torch", "torchaudio"]
        
        # 如果有 NVIDIA GPU，安装 CUDA 版 PyTorch
        if info.has_cuda and info.cuda_version:
            if info.cuda_version.startswith("12"):
                funasr_packages = ["funasr", "torch", "torchaudio", "--index-url", 
                                   "https://download.pytorch.org/whl/cu124"]
            elif info.cuda_version.startswith("11"):
                funasr_packages = ["funasr", "torch", "torchaudio", "--index-url",
                                   "https://download.pytorch.org/whl/cu118"]
        
        install_packages(config.venv_path, funasr_packages, "FunASR + PyTorch")
    
    # 6. 安装项目
    print_header("6. 安装项目")
    install_packages(config.venv_path, ["-e", "."], "dd-clip-miner-llm")
    
    # 7. 验证
    print_header("7. 验证安装")
    return verify_installation(config.venv_path)


def verify_installation(venv_path: str) -> bool:
    """验证安装"""
    python = _get_python(venv_path)
    if not python:
        print("  验证失败: 无法找到 Python")
        return False
    
    checks = [
        ("核心包", "import yaml, tqdm, httpx, openai"),
        ("FFmpeg 模块", "from dd_clip_miner_llm import ffmpeg"),
        ("Pipeline", "from dd_clip_miner_llm.pipeline import run_pipeline"),
        ("ASR 后端", "from dd_clip_miner_llm.asr_backends import build_asr_backend"),
    ]
    
    all_ok = True
    for name, import_stmt in checks:
        try:
            result = subprocess.run(
                [python, "-c", import_stmt],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                print(f"  {name}: [OK]")
            else:
                print(f"  {name}: [FAIL] {result.stderr[:100]}")
                all_ok = False
        except Exception as e:
            print(f"  {name}: [ERROR] {e}")
            all_ok = False
    
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="傻瓜化安装脚本")
    parser.add_argument("--auto", action="store_true", help="自动模式（使用默认配置）")
    parser.add_argument("--check", action="store_true", help="只检查环境，不安装")
    parser.add_argument("--no-funasr", action="store_true", help="不安装 FunASR")
    parser.add_argument("--no-cuda", action="store_true", help="不安装 CUDA")
    parser.add_argument("--no-ffmpeg", action="store_true", help="不安装 FFmpeg")
    parser.add_argument("--no-mkvmerge", action="store_true", help="不安装 MKVToolNix")
    parser.add_argument("--python", default="3.12", help="Python 版本")
    parser.add_argument("--venv", default=".venv", help="venv 路径")
    
    args = parser.parse_args()
    
    # 检测系统
    print("正在检测系统环境...")
    info = detect_system()
    print_system_info(info)
    
    # 检查模式
    if args.check:
        print("\n检查完成。")
        return
    
    # 配置
    config = SetupConfig(
        python_version=args.python,
        venv_path=args.venv,
        install_ffmpeg=not args.no_ffmpeg and not info.has_ffmpeg,
        install_mkvmerge=not args.no_mkvmerge and not info.has_mkvmerge,
        install_funasr=args.auto and not args.no_funasr,
        install_cuda=args.auto and not args.no_cuda and info.has_cuda,
    )
    
    # 自动模式或交互模式
    if not args.auto:
        print_header("安装选项")
        
        if info.has_ffmpeg:
            print("FFmpeg: 已安装")
        else:
            config.install_ffmpeg = ask_question("是否安装 FFmpeg?", default=True)
        
        if info.has_mkvmerge:
            print("MKVToolNix: 已安装")
        else:
            config.install_mkvmerge = ask_question("是否安装 MKVToolNix?", default=True)
        
        config.install_funasr = ask_question("是否安装 FunASR 支持?", default=False)
        
        if info.has_cuda:
            config.install_cuda = ask_question("是否安装 CUDA 支持?", default=True)
    
    # 执行安装
    success = run_setup(config, info)
    
    if success:
        print_header("安装完成")
        print("下一步:")
        print("  1. 激活环境: .venv\\Scripts\\activate (Windows) 或 source .venv/bin/activate (Unix)")
        print("  2. 配置: copy config.example.yaml config.yaml")
        print("  3. 设置 API key: set LLM_API_KEY=your-key (Windows) 或 export LLM_API_KEY=your-key (Unix)")
        print("  4. 运行: python -m dd_clip_miner_llm run video.mp4 --config config.yaml")
    else:
        print_header("安装失败")
        print("请检查错误信息并重试。")
        sys.exit(1)


if __name__ == "__main__":
    main()
