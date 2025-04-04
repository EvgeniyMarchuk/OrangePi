import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

image_paths = sorted(os.listdir("./data/road_simulator/all_new_images"))
for path in image_paths:
    image_path = f"./data/road_simulator/all_new_images/{path}"
    mask_path = f"./data/road_simulator/all_new_masks/{path}"
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)

    # Конвертируем в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Определяем порог для черного цвета (можно подстроить)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Находим контуры
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Находим bounding box, который охватывает все значимые пиксели
    x, y, w, h = cv2.boundingRect(np.vstack(contours))

    # Обрезаем изображение
    cropped = image[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]
    # Сохраняем результат
    cv2.imwrite(f"./data/road_simulator/correct_images/{path}", cv2.resize(cropped, (640, 480)))
    cv2.imwrite(f"./data/road_simulator/correct_masks/{path}", cv2.resize(cropped_mask, (640, 480)))
