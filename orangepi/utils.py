from orangepi.__init__ import *
from orangepi.imports import *

def load_model(
    model_path,
    model_type,
    num_classes=NUM_CLASSES,
    device=DEVICE,
):
    model_config = {
        "deeplab": {
            "constructor": deeplab_model,
            "modify_layer": lambda model: setattr(
                model.classifier, "4", torch.nn.Conv2d(256, num_classes, kernel_size=1)
            ),
        },
        "segformer": {
            "constructor": SegformerForSemanticSegmentation.from_pretrained,
            "pretrained": "nvidia/segformer-b0-finetuned-ade-512-512",
            "modify_layer": lambda model: setattr(
                model.decode_head,
                "classifier",
                torch.nn.Conv2d(256, num_classes, kernel_size=1),
            ),
            "kwargs": {"num_labels": num_classes, "ignore_mismatched_sizes": True},
        },
        "unet_mobile": {
            "constructor": smp.Unet,
            "kwargs": {
                "encoder_name": "mobilenet_v2",
                "encoder_weights": "imagenet",
                "classes": num_classes,
            },
        },
        "unet_efficient": {
            "constructor": smp.Unet,
            "kwargs": {
                "encoder_name": "efficientnet-b4",
                "encoder_weights": "imagenet",
                "classes": num_classes,
            },
        },
    }

    config = model_config.get(model_type)
    if not config:
        raise ValueError(f"Unknown model type: {model_type}")

    if "pretrained" in config:
        model = config["constructor"](config["pretrained"], **config.get("kwargs", {}))
    else:
        model = config["constructor"](**config.get("kwargs", {}))

    if "modify_layer" in config:
        config["modify_layer"](model)

    # Загружаем веса
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model


def make_prediction(model, frame, model_type):
    """В зависимости от типа модели делаем предсказания для переданных данных"""
    pred = None
    if model_type == "deeplab":
        pred = model(frame)["out"]
    if model_type == "unet_mobile":
        pred = model(frame)
    if model_type == "segformer":
        pred = model(pixel_values=frame).logits
        pred = F.interpolate(
            pred, size=(512, 512), mode="bilinear", align_corners=False
        )
    return pred


def segment_frame(
    model,
    frame,
    model_type,
    transform,
    visualization=False,
    device=DEVICE
):
    """Данная функция визуализирует сегментацию кадра frame
    Args:
        model: модель для сегментации
        frame: изображения для сегментации
        model_type: аргумент для функции make_prediction, описывающий как
        произвести предсказания в зависимости от типа модели
        transform: трансформация входного изображения image
        device: тип девайса на котором производить вычисления
    Return:
        predicted_mask: сегментированное изображение (тип - np.array())
    """
    frame = frame.convert("RGB")
    frame_tensor = transform(frame).unsqueeze(0).to(device)

    # Предсказание
    with torch.no_grad():
        output = make_prediction(model, frame_tensor, model_type)
        predicted_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # === Визуализация ===
    if visualization:
        plt.figure(figsize=(12, 5))

        # Оригинальное изображение
        plt.subplot(1, 3, 1)
        plt.imshow(frame)
        plt.axis("off")
        plt.title("Original Image")

        # Предсказанная маска
        plt.subplot(1, 3, 2)
        plt.imshow(predicted_mask, cmap="jet", alpha=0.7)
        plt.axis("off")
        plt.title("Predicted Segmentation")

        # Наложение маски на изображение
        plt.subplot(1, 3, 3)
        plt.imshow(frame)
        plt.imshow(predicted_mask, cmap="jet", alpha=0.5)
        plt.axis("off")
        plt.title("Overlay")

        plt.show()

    return np.array(predicted_mask)
