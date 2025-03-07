import sys
sys.path.insert(0, "/home/user/Projects/OrangePi")
from orangepi.__init__ import *
from orangepi.imports import *
from orangepi.utils import load_model, make_prediction

def segment_video(model_type, video_path, segmented_path):
    """Сегментирует переданное видео и сохраняет видео с сегментацией
    Args:
        model_type: тип модели, которую использовать для сегментации
        video_path: путь до видео, которое необходимо просегментировать
        segmented_path: путь, куда сохранить результат сегментации
    Return:
        nothing
    """
    if model_type == "deeplab":
        model_path = Path("../models/newest_deeplab.pth")
    elif model_type == "segformer":
        model_path = Path("../models/newest_segformer.pth")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = load_model(model_path, model_type, num_classes=NUM_CLASSES, device=DEVICE)
    model.eval()  # Переключение модели в режим inference

    # Открытие видеофайла
    camera = cv2.VideoCapture(video_path)
    if not camera.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")

    # Получение параметров видео
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = camera.get(cv2.CAP_PROP_FPS)

    # Создание VideoWriter для сохранения результата
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(segmented_path, fourcc, fps, (width, height))
    N = 0
    # Обработка каждого кадра
    while True:
        success, frame = camera.read()
        if not success:
            print("Видео закончилось")
            break

        # Изменение размера кадра до нужного разрешения
        frame = cv2.resize(frame, (width, height))

        # Преобразование кадра для модели
        input_tensor = T.ToTensor()(frame).unsqueeze(0).to(DEVICE)
        # Предсказание
        with torch.no_grad():
            output = make_prediction(model, input_tensor, model_type, size=(width, height))
            predicted_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            predicted_mask = predicted_mask.astype(np.uint8)
            predicted_mask = cv2.cvtColor(predicted_mask, cv2.COLOR_GRAY2BGR)

        # # Преобразование маски в цветное изображение
        colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
        for class_id in range(NUM_CLASSES):
            colored_mask[predicted_mask == class_id] = CLASS_COLORS[class_id]  # CLASS_COLORS - словарь цветов для каждого класса

        # Наложение маски на исходный кадр
        alpha = 0.5  # Прозрачность маски
        overlayed_frame = cv2.addWeighted(frame, 1 - alpha, colored_mask, alpha, 0)

        # Сохранение кадра в выходное видео
        video.write(overlayed_frame)
        N += 1
        print(f"Кадр {N} обработан")

    # Освобождение ресурсов
    camera.release()
    video.release()
    cv2.destroyAllWindows()

    print(f"Сегментация завершена. Результат сохранён в: {segmented_path}")


if __name__ == "__main__":
    fire.Fire()