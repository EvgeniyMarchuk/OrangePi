import torch

# Определеяем устройство на котором будем производить вычисления
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_IDS = {
    "road": 1,  # Дорога
    "lake": 2,  # Озеро / река
    "bridge": 3,  # Мост
    "tree": 4,  # Деревья
    "background": 0,  # Фон
}

NUM_CLASSES = len(CLASS_IDS)

__all__ = [
    "DEVICE",
    "NUM_CLASSES",
]