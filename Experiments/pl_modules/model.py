import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import wandb
import torch
import torch.optim as optim
import torch.nn as nn
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
        model_encoder=None,
        num_classes=1,
        loss_fn=nn.BCEWithLogitsLoss(),
        freeze_layers=0,
    ):
        super().__init__()
        self.save_hyperparameters()

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
        print(f"lr = {lr}")
        print(f"Encoder = {encoder}")
        print(f"Loss function = {loss_fn}")
        # Замораживаем первые `freeze_layers` слоев энкодера
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

        if self.current_epoch % 2 == 0 and batch_idx == 0:
            self.log_predictions(images, masks, preds)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_iou", iou, prog_bar=True)

    def test_step(self):
        pass

    def predict_step(self, batch, batch_idx):
        images, masks = batch
        preds = self(images)
        preds = torch.sigmoid(preds)
        iou = self.iou_metric(preds, masks)
        return iou.item()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return {"optimizer": optimizer}
    
    def log_predictions(self, images, masks, preds):
        """Логирование предсказаний в WandB"""
        images = images.cpu()
        masks = masks.cpu()
        preds = torch.sigmoid(preds).cpu()  # Применяем сигмоиду для бинарных масок

        logs = []
        for i in range(min(2, images.shape[0])):  # Логируем 2 изображения
            img = F.to_pil_image(images[i])
            mask = F.to_pil_image(masks[i])
            pred_mask = F.to_pil_image(preds[i])

            logs.append(wandb.Image(img, caption="Validation Input Image"))
            logs.append(wandb.Image(mask, caption="Ground Truth"))
            logs.append(wandb.Image(pred_mask, caption="Prediction"))

        self.logger.experiment.log({"Predictions": logs})
