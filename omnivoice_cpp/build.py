# -*- coding: utf-8 -*-
"""
OmniVoice C++ Engine Build Script (CUDA / CPU)
Compiles omnivoice-tts.exe and GGML runtime for high-performance GGUF inference.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BUILD_DIR = SCRIPT_DIR / "build"


def log(msg: str):
    print(f"[OmniVoice-Build] {msg}", flush=True)


def ensure_tools():
    # Check if cmake and ninja are available
    venv_scripts = Path(sys.executable).parent
    if str(venv_scripts) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{venv_scripts}{os.pathsep}{os.environ.get('PATH', '')}"

    need_install = []
    if not shutil.which("cmake"):
        need_install.append("cmake")
    if not shutil.which("ninja"):
        need_install.append("ninja")

    if need_install:
        log(f"Installing missing build tools ({', '.join(need_install)}) via pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *need_install])


def find_vcvars64() -> Path:
    # 1. Try vswhere
    candidates = [
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe",
    ]
    for vswhere in candidates:
        if vswhere.exists():
            try:
                out = subprocess.check_output(
                    [str(vswhere), "-latest", "-prerelease", "-property", "installationPath"],
                    text=True,
                    errors="ignore"
                ).strip()
                if out:
                    bat = Path(out) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                    if bat.exists():
                        return bat
            except Exception:
                pass

    # 2. Known static paths
    known_paths = [
        r"C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    ]
    for p in known_paths:
        bat = Path(p)
        if bat.exists():
            return bat

    return None


def setup_msvc_env(vcvars_path: Path):
    log(f"Configuring MSVC environment from: {vcvars_path}")
    cmd = f'call "{vcvars_path}" && set'
    output = subprocess.check_output(cmd, shell=True, text=True, errors="ignore")
    for line in output.splitlines():
        if "=" in line:
            key, val = line.split("=", 1)
            os.environ[key.strip()] = val.strip()


def main():
    print("=" * 60)
    print("       OmniVoice C++ Engine (GGML) Build System")
    print("=" * 60)

    # 1. Ensure cmake & ninja
    ensure_tools()
    cmake_bin = shutil.which("cmake")
    ninja_bin = shutil.which("ninja")
    if not cmake_bin or not ninja_bin:
        log("ERROR: cmake or ninja not found. Build cannot proceed.")
        return 1

    log(f"Found CMake: {cmake_bin}")
    log(f"Found Ninja: {ninja_bin}")

    # 2. Find and setup MSVC
    vcvars = find_vcvars64()
    if not vcvars:
        log("WARNING: Microsoft Visual Studio C++ build environment (vcvars64.bat) was not found.")
        log("If you do not need GGUF quantization, you can run OmniVoice with 'PyTorch FP16' mode directly.")
        return 0

    setup_msvc_env(vcvars)

    # 3. Detect CUDA
    has_cuda = bool(shutil.which("nvcc"))
    if has_cuda:
        log("NVIDIA CUDA Compiler (nvcc) detected. Compiling with CUDA acceleration (GGML_CUDA=ON)...")
        cuda_flag = "-DGGML_CUDA=ON"
    else:
        log("CUDA compiler not found. Compiling in CPU-only mode (GGML_CUDA=OFF)...")
        cuda_flag = "-DGGML_CUDA=OFF"

    # 4. Configure CMake
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    cfg_cmd = [
        cmake_bin,
        "-B", str(BUILD_DIR),
        "-G", "Ninja",
        cuda_flag,
        "-DCMAKE_BUILD_TYPE=Release"
    ]
    log(f"Running CMake configuration: {' '.join(cfg_cmd)}")
    subprocess.check_call(cfg_cmd, cwd=str(SCRIPT_DIR))

    # 5. Build
    cpu_cores = str(os.cpu_count() or 4)
    build_cmd = [
        cmake_bin,
        "--build", str(BUILD_DIR),
        "--config", "Release",
        "-j", cpu_cores
    ]
    log(f"Compiling targets with {cpu_cores} threads...")
    subprocess.check_call(build_cmd, cwd=str(SCRIPT_DIR))

    exe_path = BUILD_DIR / "omnivoice-tts.exe"
    if exe_path.exists():
        print()
        print("=" * 60)
        log("SUCCESS: omnivoice-tts.exe built successfully!")
        log(f"Binary path: {exe_path}")
        print("=" * 60)
        return 0
    else:
        log("ERROR: Build completed but omnivoice-tts.exe was not found.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
