import torch

# Определеяем устройство на котором будем производить вычисления
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_IDS = {
    "road": 255,  # Дорога
}

CLASS_COLORS = {
    # 0: [0, 0, 0],
    # 1: [255, 0, 0],
    # 2: [0, 255, 0],
    # 3: [0, 0, 255],
    # 4: [255, 255, 0]
    0: 0,
    1: 255,
    2: 255,
    3: 255,
    4: 255
}

NUM_CLASSES = len(CLASS_COLORS)

__all__ = [
    "DEVICE",
    "CLASS_COLORS",
    "NUM_CLASSES",
]