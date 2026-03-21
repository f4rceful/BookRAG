import logging

from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


def get_device() -> str:
    # Определяет лучшее доступное устройство: CUDA (NVIDIA/AMD ROCm), MPS (Apple), CPU
    try:
        import torch

        # NVIDIA CUDA или AMD ROCm (ROCm использует тот же torch.cuda API)
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            try:
                cap = torch.cuda.get_device_capability(0)
                # PyTorch 2.6 поддерживает sm_60+ (Pascal и новее, GTX 1050 = sm_61)
                if cap[0] >= 6:
                    logger.info(f"NVIDIA GPU (sm_{cap[0]}{cap[1]}), используем CUDA: {device_name}")
                    return "cuda"
                # ROCm не использует compute capability — пропускаем проверку
                is_amd = "amd" in device_name.lower() or "radeon" in device_name.lower()
                if is_amd:
                    logger.info(f"AMD GPU (ROCm), используем cuda device: {device_name}")
                    return "cuda"
                logger.warning(
                    f"GPU {device_name} (sm_{cap[0]}{cap[1]}) не поддерживается "
                    f"PyTorch 2.6 (нужен sm_60+), используем CPU"
                )
            except Exception:
                # ROCm иногда не поддерживает get_device_capability
                logger.info(f"GPU обнаружен, используем: {device_name}")
                return "cuda"

        # Apple Silicon — MPS (Metal Performance Shaders)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Apple Silicon обнаружен, используем MPS")
            return "mps"

    except Exception:
        pass

    logger.info("Используем CPU для эмбеддингов")
    return "cpu"


def build_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    # Создаёт модель эмбеддингов с автоматическим выбором устройства (CUDA/ROCm/MPS/CPU)
    device = get_device()
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
