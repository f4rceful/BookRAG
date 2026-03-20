import subprocess
import sys
import re
import os
import platform
from pathlib import Path

# Минимальная рабочая версия PyTorch для проекта
TORCH_MIN_VERSION = "2.5.0"

def run_command(cmd: list):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return result.returncode == 0, result.stdout
    except FileNotFoundError:
        return False, ""

def get_cuda_version():
    ok, out = run_command(["nvidia-smi"])
    if ok:
        match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
        if match: return int(match.group(1)), int(match.group(2))
    return None

def has_amd_gpu_windows():
    """Проверка наличия AMD GPU на Windows через PowerShell."""
    if sys.platform != "win32":
        return False
    ok, out = run_command(["powershell", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"])
    if ok:
        output = out.lower()
        return "amd" in output or "radeon" in output
    return False

def is_apple_silicon():
    return sys.platform == "darwin" and platform.machine() == "arm64"

def install_pytorch():
    print("--- Фаза 1: Установка PyTorch ---")
    cuda = get_cuda_version()
    amd_win = has_amd_gpu_windows()
    
    index_url = None
    label = "CPU"
    
    if cuda:
        major, _ = cuda
        print(f"Найдена NVIDIA CUDA {major}.x")
        index_url = "https://download.pytorch.org/whl/cu124" if major >= 12 else "https://download.pytorch.org/whl/cu118"
        label = "NVIDIA GPU"
    elif amd_win:
        print("Найдена видеокарта AMD (Windows).")
        print("INFO: Ollama будет использовать GPU автоматически.")
        print("Эмбеддинги будут работать на CPU (самый стабильный вариант для AMD Windows).")
        label = "AMD (Ollama GPU + Torch CPU)"
    elif sys.platform == "linux" and Path("/opt/rocm").exists():
        index_url = "https://download.pytorch.org/whl/rocm6.1"
        label = "AMD ROCm (Linux)"
    elif is_apple_silicon():
        label = "Apple Silicon (MPS)"
    
    print(f"Выбрана конфигурация: {label}")
    
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    install_cmd = [sys.executable, "-m", "pip", "install", f"torch>={TORCH_MIN_VERSION}"]
    if index_url:
        install_cmd += ["--index-url", index_url]
    
    subprocess.run(install_cmd)

def install_requirements():
    print("\n--- Фаза 2: Установка зависимостей проекта ---")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def verify_gpu():
    print("\n--- Фаза 3: Проверка GPU ---")
    # Для Ollama мы не можем проверить из Python напрямую легко, но проверим Torch
    script = "import torch; print(f'Torch version: {torch.__version__}'); print(f'Torch CUDA: {torch.cuda.is_available()}')"
    subprocess.run([sys.executable, "-c", script])
    
    if has_amd_gpu_windows():
        print("ПОДСКАЗКА: Твоя AMD видяха будет работать через Ollama.")
        print("Убедись, что Ollama запущена (иконка в трее).")

def main():
    print("=== BookRAG Installer (AMD Optimized) ===")
    install_pytorch()
    install_requirements()
    verify_gpu()
    print("\nГотово! Запускай проект.")

if __name__ == "__main__":
    main()
