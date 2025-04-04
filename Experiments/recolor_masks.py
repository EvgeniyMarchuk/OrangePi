import os
import cv2
import numpy as np

mask_dir = "./data/road_simulator/correct_masks"
output_dir = "./yolo/data/correct_masks"
os.makedirs(output_dir, exist_ok=True)

for mask_name in os.listdir(mask_dir):
    mask_path = os.path.join(mask_dir, mask_name)
    output_path = os.path.join(output_dir, mask_name)
    
    img = cv2.imread(mask_path)

    # Если изображение уже одноканальное (ЧБ)
    if len(img.shape) == 2:
        # Нормализуем: все ненулевые пиксели -> 255
        _, result = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    else:
        # Цветная маска: преобразуем голубые/белые в белые
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Маска для голубого (H=85-115)
        blue_mask = cv2.inRange(hsv, np.array([85, 50, 50]), np.array([115, 255, 255]))
        
        # Маска для белого (S=0-30, V=220-255)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 220]), np.array([180, 30, 255]))
        
        # Объединяем маски: голубые ИЛИ белые -> белые
        combined_mask = cv2.bitwise_or(blue_mask, white_mask)
        
        # Создаем одноканальный результат
        result = np.zeros_like(combined_mask)
        result[combined_mask > 0] = 255

    cv2.imwrite(output_path, result)
    print(f"Обработано: {mask_name}")

print("Все маски преобразованы в одноканальные ЧБ!")