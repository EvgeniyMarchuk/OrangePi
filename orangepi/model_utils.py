from orangepi.__init__ import *
from orangepi.imports import *


def load_model(
    model_path,
    model_type,
    num_classes=5,
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


def make_prediction(model, frame, model_type, task):
    """В зависимости от типа модели делаем предсказания для переданных данных"""
    raw_pred = None
    predicted_classes = None
    if model_type == "deeplab_smp":
        raw_pred = model(frame)
        predicted_classes = torch.argmax(raw_pred, dim=1).to(torch.int64)
    else:
        if model_type == "deeplab":
            raw_pred = model(frame)["out"]
        if model_type == "unet_mobile":
            raw_pred = model(frame)
        if model_type == "segformer":
            raw_pred = model(pixel_values=frame).logits
            raw_pred = F.interpolate(
                raw_pred, size=(512, 512), mode="bilinear", align_corners=False
            )
        if task == "binary":
            predicted_classes = (raw_pred.squeeze(1) > 0.5).int().to(torch.int64)
        else:
            predicted_classes = torch.argmax(raw_pred, dim=1).to(torch.int64)
    
    return raw_pred, predicted_classes


def train(
    model,
    train_dataloader,
    valid_dataloader,
    criterion,
    lr,
    epochs,
    model_name,
    num_classes=1,
    iterations=100,
    device=DEVICE,
    verbose=False,
    is_scheduler=False,
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    task="multiclass" if num_classes > 2 else "binary"
    iou_score = JaccardIndex(task=task, num_classes=num_classes).to(device)
    # пока что Dice score только для бинарной сегментации стоит
    if task == "multiclass":
        dice_score = Dice(num_classes=num_classes, average="macro", multiclass=(task == "multiclass")).to(device)
    else:
        dice_score = Dice().to(device)
    recall_score = Recall(task=task).to(DEVICE)

    if is_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.5)

    # Получаем первый батч валидации для подсчета скоров
    val_data, val_true_mask = next(iter(valid_dataloader))
    val_data, val_true_mask = val_data.to(device), val_true_mask.to(device)

    best_model = None
    best_val_loss = float("inf")  # Инициализируем наихудшее значение лосса
    best_val_iou_score = 0
    best_val_dice_score = 0
    best_val_recall_score = 0

    loss_history = {"train": [], "valid": []}
    iou_score_history = {"train": [], "valid": []}
    dice_score_history = {"train": [], "valid": []}
    recall_score_history = {"train": [], "valid": []}
    # Начальная проверка
    model.eval()
    with torch.no_grad():
        val_pred, predicted_classes = make_prediction(model, val_data, model_name, task)
        print(val_pred.shape, val_true_mask.shape)
        initial_val_loss = criterion(val_pred.squeeze(1), val_true_mask.float()).item()
        initial_val_iou_score = iou_score(predicted_classes, val_true_mask).detach()
        initial_val_dice_score = dice_score(predicted_classes, val_true_mask).detach()
        initial_val_recall_score = recall_score(predicted_classes, val_true_mask).detach()

    print(f"Initial validation loss: {initial_val_loss:.4f}")
    print(f"Initial validation IoU score: {initial_val_iou_score:.4f}")
    print(f"Initial validation Dice score: {initial_val_dice_score:.4f}")
    print(f"Initial validation Recall score: {initial_val_recall_score:.4f}")

    counter = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Обучение
        model.train()
        total_loss = 0
        total_iou_score = 0
        total_dice_score = 0
        total_recall_score = 0

        for data, true_mask in tqdm(train_dataloader, desc="Training", leave=False):
            if len(data) == 1:
                continue
            data, true_mask = data.to(device), true_mask.to(device)
            optimizer.zero_grad()

            prediction, predicted_classes = make_prediction(model, data, model_name, task)

            loss = criterion(prediction.squeeze(1), true_mask.float())

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_iou_score += iou_score(predicted_classes, true_mask).item()
            total_dice_score += dice_score(predicted_classes, true_mask).item()
            total_recall_score += recall_score(predicted_classes, true_mask).item()

            # counter += 1
            # if counter % iterations == 0:
            #     loss_value = total_loss / iterations
            #     iou_value = total_iou_score / iterations
            #     dice_value = total_dice_score / iterations
            #     recall_value = total_recall_score / iterations
            #     print(f"Loss: {loss_value:.4f}")
            #     print(f"IoU score: {iou_value:.4f}")
            #     print(f"Dice score: {dice_value:.4f}")
            #     print(f"Recall score: {recall_value:.4f}")
        print(torch.unique(predicted_classes))
        print(torch.unique(true_mask))

        train_loss = total_loss / len(train_dataloader)
        train_iou_score = total_iou_score / len(train_dataloader)
        train_dice_score = total_dice_score / len(train_dataloader)
        train_recall_score = total_recall_score / len(train_dataloader)

        if is_scheduler:
            scheduler.step()

        # Оценка на валидации
        torch.cuda.empty_cache()  # Очищаем кеш перед валидацией
        model.eval()
        with torch.no_grad():
            val_pred, predicted_classes = make_prediction(model, val_data, model_name, task)
            
            val_loss = criterion(val_pred.squeeze(1), val_true_mask.float()).item()
            val_iou_score = iou_score(predicted_classes, val_true_mask).detach()
            val_dice_score = dice_score(predicted_classes, val_true_mask).detach()
            val_recall_score = recall_score(predicted_classes, val_true_mask).detach()

        # Сохраняем лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_iou_score = val_iou_score
            best_val_dice_score = val_dice_score
            best_val_recall_score = val_recall_score
            best_model = deepcopy(model)

        # Логирование
        loss_history["train"].append(train_loss)
        loss_history["valid"].append(val_loss)
        iou_score_history["train"].append(train_iou_score)
        iou_score_history["valid"].append(val_iou_score)
        dice_score_history["train"].append(train_dice_score)
        dice_score_history["valid"].append(val_dice_score)
        recall_score_history["train"].append(train_recall_score)
        recall_score_history["valid"].append(val_recall_score)

        if verbose:
            print(f"Train Loss: {train_loss:.4f},\t\tVal Loss: {val_loss:.4f}")
            print(
            f"Train IoU score: {train_iou_score:.4f},\tVal IoU score: {val_iou_score:.4f}"
            )
            print(
            f"Train Dice score: {train_dice_score:.4f},\tVal Dice score: {val_dice_score:.4f}"
            )
            print(
            f"Train Recall score: {train_recall_score:.4f},\tVal Recall score: {val_recall_score:.4f}"
            )
            print()

    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Validation IoU score: {best_val_iou_score:.4f}")
    print(f"Best Validation Dice score: {best_val_dice_score:.4f}")
    print(f"Best Validation Recall score: {best_val_recall_score:.4f}")

    return best_model, loss_history, iou_score_history, dice_score_history, recall_score_history


def visualize_losses_and_scores(losses, iou_scores):
    _, axes = plt.subplots(1, 3, figsize=(18, 6))
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

    scores = [iou_scores]
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

        axes[i+1].plot(epochs, train_scores, c="r", label="Train")
        axes[i+1].scatter(epochs, train_scores, c="r")
        axes[i+1].plot(epochs, valid_scores, c="b", label="Validation")
        axes[i+1].scatter(epochs, valid_scores, c="b")
        axes[i+1].set_title(f"Train and validation {"IoU" if i == 0 else "Dice"} score")
        axes[i+1].set_xlabel("Epoch")
        axes[i+1].set_ylabel(f"{"IoU" if i == 0 else "Dice"} score")
        axes[i+1].legend()
        axes[i+1].grid()

    plt.show()


def test(model, dataloader, criterion, model_type, iterations=100, num_classes=1, device=DEVICE, num_examples=1):
    model = model.to(device)
    task = "multiclass" if num_classes > 2 else "binary"
    iou_score = JaccardIndex(
        task=task, num_classes=num_classes
    ).to(device)
    dice_score = Dice().to(device)

    model.eval()
    images, _ = next(iter(dataloader))
    output = make_prediction(model, images.to(DEVICE), model_type)
    if task == "binary":
        predicted_classes = (output.squeeze(1) > 0.5).to(torch.int64)
    else:
        predicted_classes = torch.argmax(output, dim=1).to(torch.int64)
    _, axes = plt.subplots(3, num_examples, figsize=(12, 6))
    for i in range(max(num_examples, len(images))):
        predicted_mask = predicted_classes[i].detach().cpu().numpy()
        axes[0, i].imshow(images[i].permute(1, 2, 0))
        axes[0, i].axis("off")
        axes[0, i].set_title("Original Image")

        # Предсказанная маска
        axes[1, i].imshow(predicted_mask, cmap="jet", alpha=0.7)
        axes[1, i].axis("off")
        axes[1, i].set_title("Predicted Segmentation")

        # Наложение маски на изображение
        axes[2, i].imshow(images[i].permute(1, 2, 0))
        axes[2, i].imshow(predicted_mask, cmap="jet", alpha=0.5)
        axes[2, i].axis("off")
        axes[2, i].set_title("Overlay")
    plt.show()

    for i, data_and_mask in tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing", leave=False):
        torch.cuda.empty_cache() 

        data, true_mask = data_and_mask
        total_loss = 0
        total_iou_score  = 0
        total_dice_score = 0
        data, true_mask = data.to(device), true_mask.to(device)

        prediction = make_prediction(model, data, model_type)
        loss = criterion(prediction.squeeze(1), true_mask.float())

        loss.backward()

        total_loss += loss.item()
        total_iou_score += iou_score(
            prediction.squeeze(1), true_mask.squeeze(1)
        ).item()
        total_dice_score += dice_score(prediction, true_mask).item()
        if i % iterations == 0:
            loss_value = total_loss / iterations
            iou_value = total_iou_score / iterations
            dice_value = total_dice_score / iterations
            print(f"Loss: {loss_value:.4f}")
            print(f"IoU score: {iou_value:.4f}")
            print(f"Dice score: {dice_value:.4f}")


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
        frame: изображение для сегментации
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
    "test"
]
