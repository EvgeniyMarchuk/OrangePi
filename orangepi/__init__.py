import torch

# Определеяем устройство на котором будем производить вычисления
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_IDS = {
    "road": 255,  # Дорога
}

NUM_CLASSES = len(CLASS_IDS)

__all__ = [
    "DEVICE",
    "NUM_CLASSES",
]