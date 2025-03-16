import torch
import torch.nn as nn
import pytorch_lightning as pl
import optuna
import segmentation_models_pytorch as smp
from optuna.integration import PyTorchLightningPruningCallback
from pathlib import Path
from dataset import RoadSegmentationDataModule
from model import SegmentationModel

def objective(trial: optuna.Trial):
    """Функция, оптимизируемая Optuna"""
    
    # Гиперпараметры, которые будет подбирать Optuna
    encoder = trial.suggest_categorical("encoder", ["resnet50", "efficientnet-b0", "mobilenet_v2"])
    learning_rate = trial.suggest_categorical("learning_rate", [1e-3, 3e-3, 7e-4, 4e-4])
    loss_function_name = trial.suggest_categorical("loss_function", ["dice"])
    scheduler_name = trial.suggest_categorical("scheduler", ["cosine", "step"])
    num_frozen_layers = 0#trial.suggest_int("num_frozen_layers", 0, 20)  # Количество слоев, которые останутся замороженными

    # Функции потерь
    loss_functions = {
        "bce": nn.BCEWithLogitsLoss(),
        "dice": smp.losses.DiceLoss(mode="binary"),
        "focal": smp.losses.FocalLoss(mode="binary"),
        "tversky": smp.losses.TverskyLoss(mode="binary")
    }
    loss_function = loss_functions[loss_function_name]

    # Датамодуль
    dm = RoadSegmentationDataModule(
        train_path=Path("../data/DeepGlobeDataset/train"),
        val_path=Path("../data/DeepGlobeDataset/valid"),
        test_path=Path("../data/DeepGlobeDataset/test"),
        batch_size=2,
    )

    # Модель
    model = SegmentationModel(
        encoder=encoder,
        lr=learning_rate,
        model_encoder=encoder,
        num_classes=1,
        loss_fn=loss_function,
        freeze_layers=num_frozen_layers,
    )

    # LR scheduler
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
    elif scheduler_name == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    elif scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR
    else:
        scheduler = None
    print(f"Sheduler is {scheduler_name}")
    # Callbacks
    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        pl.callbacks.ModelCheckpoint(
            dirpath=Path("../checkpoints"),
            filename=f"trial_{trial.number}",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
        ),
    ]

    # Trainer
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        precision=32,
        max_epochs=3,
        callbacks=callbacks,
    )

    # Запуск тренировки
    trainer.fit(model, datamodule=dm)

    # Оптимизируем валидационный лосс
    return trainer.callback_metrics["val_loss"].item()


if __name__ == "__main__":
    pl.seed_everything(42)

    study = optuna.create_study(direction="minimize")  # Минимизируем val_loss
    study.optimize(objective, n_trials=20, timeout=9000)  # Запускаем 20 экспериментов или 1 час

    # Выводим лучшие параметры
    print("Лучшие параметры:", study.best_params)
