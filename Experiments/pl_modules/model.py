import os
import cv2
import wandb
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from pathlib import Path
from torchinfo import summary
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchmetrics import JaccardIndex
from torchvision.transforms import functional as F
import pytorch_lightning as pl
import segmentation_models_pytorch as smp


class SegmentationModel(pl.LightningModule):
    """Класс для инициализации модели"""
    def __init__(
        self,
        encoder="resnet50",
        encoder_weights="imagenet",
        lr=1e-4,
        num_classes=1,
        loss_fn=nn.BCEWithLogitsLoss(),
        freeze_layers=0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.backward_counter = 0

        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes
        )

        self.lr = lr
        self.loss_fn = loss_fn
        self.iou_metric = JaccardIndex(
            task="binary",
            num_classes=num_classes
        )

        # Замораживаем первые freeze_layers слоев энкодера
        if freeze_layers > 0:
            encoder_layers = list(self.model.encoder.children())
            for layer in encoder_layers[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Описываем, что должно произойти при обучении"""
        images, masks = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, masks)
        
        preds = torch.sigmoid(outputs)
        iou = self.iou_metric(preds, masks)
        
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_iou", iou, prog_bar=True, on_epoch=True, on_step=False)
        
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        
        loss = self.loss_fn(outputs, masks)
        preds = torch.sigmoid(outputs)
        iou = self.iou_metric(preds, masks)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_iou", iou, prog_bar=True)

    def get_target_images_and_masks(self, base_image_dir, image_paths, base_mask_dir, mask_paths):
        images = []
        normalized_images = []
        masks = []

        transform = A.Compose([
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.7),
            A.RandomRotate90(p=0.4),
            A.Perspective(scale=(0.0, 0.3)),
            A.ShiftScaleRotate(rotate_limit=(-120, 120)),
            #A.ThinPlateSpline(),
            A.Resize(480, 640),
        ], additional_targets={"mask": "mask"})       
        for image_path, mask_path in zip(image_paths, mask_paths):
            image_path = os.path.join(base_image_dir, image_path)
            mask_path = os.path.join(base_mask_dir, mask_path)

            orig_image = cv2.imread(str(image_path))
            orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            orig_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            orig_mask = (orig_mask > 128).astype("float32")
            
            transformed = transform(image=orig_image, mask=orig_mask)
            image = A.CoarseDropout()(image=transformed["image"])["image"]
            mask = transformed["mask"]
            tensors = ToTensorV2()(image=image, mask=mask)
            images.append(tensors["image"].unsqueeze(0))
            masks.append(tensors["mask"].unsqueeze(0))

            normalize = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406),  std=(0.229, 0.224, 0.225)),
                A.ToTensorV2()
            ])
            image = normalize(image=image)["image"].unsqueeze(0)
            normalized_images.append(image)

        return torch.stack(normalized_images), torch.stack(images), torch.stack(masks)

    def on_train_epoch_end(self):
        torch.cuda.empty_cache()
        base_image_dir = Path("../target_images")
        base_mask_dir = Path("../target_masks")
        image_paths = sorted(os.listdir(base_image_dir))
        mask_paths = sorted(os.listdir(base_mask_dir))
        normalized_images, orig_images, masks = self.get_target_images_and_masks(
            base_image_dir, image_paths,
            base_mask_dir, mask_paths
        )
        preds = []
        with torch.inference_mode():
            for i in range(len(normalized_images)):
                image = normalized_images[i].to(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                output = self(image)
                pred = torch.sigmoid(output).cpu().numpy()[0, 0]
                pred = (pred > 0.5).astype(np.uint8) * 255
                preds.append(pred)

        self.log_predictions(orig_images, masks, preds)
   
    def on_after_backward(self):
        # следим за градиентами
        if self.current_epoch % 5 == 0 and self.backward_counter % 5 == 0:
            value_norm = 0
            grad_norm = 0
            for name, value in self.named_parameters():
                if value.grad is not None:
                    value_norm += value.norm(p=float("inf"))
                    grad_norm += value.grad.norm(p=float("inf"))
            print(f"dW / W = {grad_norm / value_norm}")
        self.backward_counter += 1

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30], gamma=0.5)
        return [optimizer]#, [scheduler]
    
    def log_predictions(self, images, masks, preds):
        """Логирование предсказаний в WandB"""
        images = images#.cpu()
        masks = masks#.cpu()
        preds = preds#.cpu()

        logs = []
        for i in range(min(4, images.shape[0])):  # Логируем 4 изображения
            img = F.to_pil_image(images[i][0])
            mask = F.to_pil_image(masks[i])
            pred_mask = F.to_pil_image(preds[i])

            logs.append(wandb.Image(img, caption="Input Image"))
            logs.append(wandb.Image(mask, caption="Ground Truth"))
            logs.append(wandb.Image(pred_mask, caption="Prediction"))

        self.logger.experiment.log({"Predictions": logs})
