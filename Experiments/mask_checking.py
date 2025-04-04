import cv2
import numpy as np

image = cv2.imread('./yolo/data/train/masks/CameraFlight.0074.png')
print(f"Уникальные классы: {np.unique(image)}")