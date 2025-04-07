import os
import torch
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
from torchinfo import summary
from albumentations.pytorch import ToTensorV2
from model import SegmentationModel
import time


def infer_image(model_checkpoint: Path, image_path: Path, output_dir: Path, model_type):
    """Загружает модель, выполняет инференс и сохраняет результат."""

    # Загружаем модель из чекпойнта
    model = SegmentationModel.load_from_checkpoint(str(model_checkpoint))
    # summary(
    #     model=model,
    #     input_size=(1, 3, 480, 640),
    #     col_names=["input_size", "output_size", "num_params", "trainable"],
    #     col_width=20,
    #     row_settings=["var_names"],
    # )
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = A.Compose([
        A.Resize(480, 640),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    start_time = time.time()
    image = transform(image=image)["image"].unsqueeze(0)    # модели поступали батчи, поэтому добавляем размерность
    # image = image.float()

    with torch.no_grad():
        image = image.to("cuda" if torch.cuda.is_available() else "cpu")
        pred = model(image)
        pred = torch.sigmoid(pred)
        pred = pred.cpu().numpy()[0, 0]

    pred = (pred > 0.5).astype(np.uint8) * 255
    end_time = time.time()
    print(f"Время инференса = {end_time - start_time}")

    # Сохраняем результат
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_mask_{model_type}.png"
    cv2.imwrite(str(output_path), pred)
    print(f"Сохранён результат: {output_path}")

base_dir = Path("../test_images")
paths = os.listdir(base_dir)
for path in paths:
    path = base_dir / path
    infer_image(
        model_checkpoint=Path("../checkpoints/Unet/unet_600_aug_4.ckpt"),
        image_path=Path(path),
        output_dir=Path("../predictions/Unet_600"),
        model_type="Unet_600_aug"
    )
