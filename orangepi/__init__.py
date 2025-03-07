import torch

# Определеяем устройство на котором будем производить вычисления
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_IDS = {
    "road": 255,  # Дорога
}

CLASS_COLORS = {
    0: 0,
    1: 250,
    2: 190,
    3: 130,
    4: 70
}

NUM_CLASSES = len(CLASS_IDS)

__all__ = [
    "DEVICE",
    "CLASS_COLORS",
    "NUM_CLASSES",
]