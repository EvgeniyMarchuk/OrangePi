import os
import cv2
import numpy as np

def coco_semantic_to_yolo_segmentation(output_dir, masks_dir, include_other=False):
    """
    Преобразует семантические маски COCO в формат YOLO для сегментации.
    """
    # Итерируемся по изображениям в image_dir
    for mask_filename in os.listdir(masks_dir):
        # Получаем имя файла маски (предполагаем, что маски имеют те же имена, что и изображения)
        mask_path = os.path.join(masks_dir, mask_filename)

        # Загружаем маску
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Could not read mask file: {mask_path}")
            continue

        # Получаем размеры изображения
        height, width = mask.shape

        # Создаем файл для YOLO-аннотаций
        txt_filename = mask_filename.replace('.png', '.txt')
        txt_path = os.path.join(output_dir, txt_filename)

        with open(txt_path, 'w') as f:
            # Находим уникальные class_id на маске
            class_ids = np.unique(mask)
            for class_id in class_ids:
                if class_id == 255 and not include_other:
                    continue  # Пропускаем фон (если class_id=255)

                # Создаем бинарную маску для текущего class_id
                binary_mask = (mask == class_id).astype(np.uint8)

                # Находим контуры на бинарной маске
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Записываем контуры в YOLO-формат
                for contour in contours:
                    if len(contour) >= 3:  # Контур должен содержать хотя бы 3 точки
                        yolo_points = []
                        for point in contour.squeeze():
                            x = point[0] / width  # Нормализация координат
                            y = point[1] / height
                            yolo_points.extend([str(x), str(y)])

                        if len(yolo_points) > 0:
                            yolo_line = f"{class_id} {' '.join(yolo_points)}\n"
                            f.write(yolo_line)

# Пример использования
data_root = './data/road_simulator/'
masks = os.path.join(data_root, 'correct_masks')
output = os.path.join(data_root, 'correct_labels')

# Создаем директории для выходных файлов
os.makedirs(output, exist_ok=True)

coco_semantic_to_yolo_segmentation(output, masks, include_other=False)
print("Преобразование для сегментации завершено!")