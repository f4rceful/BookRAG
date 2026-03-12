import subprocess
import sys
import re


def get_cuda_version():
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode != 0:
            return None
        match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", result.stdout)
        if match:
            return int(match.group(1)), int(match.group(2))
    except FileNotFoundError:
        pass
    return None


def main():
    cuda = get_cuda_version()

    if cuda:
        major, minor = cuda
        print(f"Обнаружена CUDA {major}.{minor}")

        if major >= 12:
            index_url = "https://download.pytorch.org/whl/cu121"
            label = "CUDA 12.1"
        elif major == 11 and minor >= 8:
            index_url = "https://download.pytorch.org/whl/cu118"
            label = "CUDA 11.8"
        elif major == 11:
            index_url = "https://download.pytorch.org/whl/cu117"
            label = "CUDA 11.7"
        else:
            index_url = None
            label = None

        if index_url:
            print(f"Устанавливаем PyTorch с поддержкой GPU ({label})...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "torch",
                "--index-url", index_url
            ])
            if result.returncode != 0:
                print(f"Не удалось установить GPU-версию (возможно, Python {sys.version_info.major}.{sys.version_info.minor} не поддерживается индексом). Устанавливаем CPU-версию...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])
        else:
            print(f"CUDA {major}.{minor} не поддерживается, устанавливаем CPU-версию PyTorch...")
    else:
        print("NVIDIA GPU не обнаружен (AMD GPU не поддерживается), устанавливаем CPU-версию PyTorch...")

    print("Устанавливаем остальные зависимости...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✅ Установка завершена!")


if __name__ == "__main__":
    main()
