import cv2
import numpy as np

def overlay_images(image1_path, image2_path, alpha=0.5):
    """
    Накладывает второе изображение на первое с заданной прозрачностью.
    :param image1_path: Путь к первому изображению (фоновое).
    :param image2_path: Путь ко второму изображению (накладываемое).
    :param alpha: Коэффициент прозрачности второго изображения (0-1).
    :return: Итоговое изображение с наложением.
    """
    # Загружаем изображения
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    
    if img1 is None or img2 is None:
        raise ValueError("Ошибка загрузки изображений. Проверьте пути к файлам.")
    
    # Изменяем размер второго изображения до размеров первого
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Наложение изображений с прозрачностью
    blended = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
    
    cv2.imwrite("./yolo/predictions/result.png", blended)

overlay_images("./yolo/data/train/images/CameraFlight.0522.png", "./yolo/predictions/mask_0522.png", alpha=0.5)
