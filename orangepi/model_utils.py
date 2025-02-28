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


def train(
    model,
    train_dataloader,
    valid_dataloader,
    criterion,
    lr,
    epochs,
    model_name,
    num_classes=5,
    verbose=False,
    is_scheduler=False,
):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    iou_score = JaccardIndex(
        task="multiclass" if num_classes > 2 else "binary", num_classes=num_classes
    ).to(DEVICE)
    dice_score = Dice(num_classes=num_classes, threshold=0.5, zero_division=1e-8).to(
        DEVICE
    )

    if is_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Получаем первый батч валидации (фиксируем его)
    val_data, val_true_mask = next(iter(valid_dataloader))
    val_data, val_true_mask = val_data.to(DEVICE), val_true_mask.to(DEVICE)

    best_model = None
    best_val_loss = float("inf")  # Инициализируем наихудшее значение лосса
    best_val_iou_score = 0
    best_val_dice_score = 0

    loss_history = {"train": [], "valid": []}
    iou_score_history = {"train": [], "valid": []}
    dice_score_history = {"train": [], "valid": []}

    # Начальная проверка
    model.eval()
    with torch.no_grad():
        val_pred = make_prediction(model, val_data, model_name)
        predicted_classes = torch.argmax(val_pred, dim=1)  # Теперь [B, H, W]
        initial_val_loss = criterion(val_pred, val_true_mask).item()
        initial_val_iou_score = iou_score(predicted_classes, val_true_mask).detach()
        initial_val_dice_score = dice_score(predicted_classes, val_true_mask).detach()

    print(f"Initial validation loss: {initial_val_loss:.4f}")
    print(f"Initial validation IoU score: {initial_val_iou_score:.4f}")
    print(f"Initial validation Dice score: {initial_val_dice_score:.4f}")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Обучение
        model.train()
        total_loss = 0
        total_iou_score, total_dice_score = 0, 0

        for data, true_mask in tqdm(train_dataloader, desc="Training", leave=False):
            data, true_mask = data.to(DEVICE), true_mask.to(DEVICE)
            optimizer.zero_grad()

            prediction = make_prediction(model, data, model_name)
            loss = criterion(prediction, true_mask)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_iou_score += iou_score(prediction, true_mask).item()
            total_dice_score += dice_score(prediction, true_mask).item()

        train_loss = total_loss / len(train_dataloader)
        train_iou_score = total_iou_score / len(train_dataloader)
        train_dice_score = total_dice_score / len(train_dataloader)

        # Шаг learning rate scheduler'а
        if is_scheduler:
            scheduler.step()

        # Оценка на валидации
        torch.cuda.empty_cache()  # Очищаем кеш перед валидацией
        model.eval()
        with torch.no_grad():
            val_pred = make_prediction(model, val_data, model_name)
            predicted_classes = torch.argmax(val_pred, dim=1)  # Теперь [B, H, W]
            val_loss = criterion(val_pred, val_true_mask).item()
            val_iou_score = iou_score(predicted_classes, val_true_mask).detach()
            val_dice_score = dice_score(predicted_classes, val_true_mask).detach()

        # Сохраняем лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_iou_score = val_iou_score
            best_val_dice_score = val_dice_score
            best_model = deepcopy(model)  # Глубокая копия, а не ссылка

        # Логирование
        loss_history["train"].append(train_loss)
        loss_history["valid"].append(val_loss)
        iou_score_history["train"].append(train_iou_score)
        iou_score_history["valid"].append(val_iou_score)
        dice_score_history["train"].append(train_dice_score)
        dice_score_history["valid"].append(val_dice_score)

        if verbose:
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(
                f"Train IoU score: {train_iou_score:.4f}, Val IoU score: {val_iou_score:.4f}"
            )
            print(
                f"Train Dice score: {train_dice_score:.4f}, Val Dice score: {val_dice_score:.4f}"
            )

    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Validation IoU score: {best_val_iou_score:.4f}")
    print(f"Best Validation Dice score: {best_val_dice_score:.4f}")

    return best_model, loss_history, iou_score_history, dice_score_history


def visualize_losses_and_scores(losses, iou_scores, dice_scores):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    epochs = np.arange(len(losses["train"]))

    # Лоссы
    axes[0].plot(epochs, losses["train"], c="r", label="Train")
    axes[0].scatter(epochs, losses["train"], c="r")
    axes[0].plot(epochs, losses["valid"], c="b", label="Validation")
    axes[0].scatter(epochs, losses["valid"], c="b")
    axes[0].set_title("Train and validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid()

    scores = [iou_scores, dice_scores]
    for i, score in enumerate(scores):
        # Scores переводим тензоры в CPU и numpy
        train_scores = [
            s.cpu().item() if isinstance(s, torch.Tensor) else s
            for s in score["train"]
        ]
        valid_scores = [
            s.cpu().item() if isinstance(s, torch.Tensor) else s
            for s in score["valid"]
        ]

        axes[i].plot(epochs, train_scores, c="r", label="Train")
        axes[i].scatter(epochs, train_scores, c="r")
        axes[i].plot(epochs, valid_scores, c="b", label="Validation")
        axes[i].scatter(epochs, valid_scores, c="b")
        axes[i].set_title("Train and validation IoU score")
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel(f"{"IoU" if i == 0 else "Dice"} score")
        axes[i].legend()
        axes[i].grid()

    plt.show()


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


__all__ = [
    "load_model",
    "make_prediction",
    "train",
    "visualize_losses_and_scores",
    "segment_frame",
]
