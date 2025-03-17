import torch
import torch.nn as nn
import pytorch_lightning as pl
import optuna
import segmentation_models_pytorch as smp
from pathlib import Path
from dataset import RoadSegmentationDataModule
from model import SegmentationModel


def train_model(encoder, learning_rate, loss_function_name, scheduler_name, num_frozen_layers, max_epochs=10):
    """Функция для обучения модели с заданными параметрами"""

    # Определяем функцию потерь
    loss_functions = {
        "bce": nn.BCEWithLogitsLoss(),
        "dice": smp.losses.DiceLoss(mode="binary"),
        "focal": smp.losses.FocalLoss(mode="binary"),
        "tversky": smp.losses.TverskyLoss(mode="binary"),
    }
    loss_function = loss_functions[loss_function_name]

    # Создаём датамодуль
    dm = RoadSegmentationDataModule(
        train_path=Path("../data/DeepGlobeDataset/train"),
        val_path=Path("../data/DeepGlobeDataset/valid"),
        test_path=Path("../data/DeepGlobeDataset/test"),
        batch_size=2,
    )

    # Создаём модель
    model = SegmentationModel(
        encoder=encoder,
        lr=learning_rate,
        model_encoder=encoder,
        num_classes=1,
        loss_fn=loss_function,
        freeze_layers=num_frozen_layers,
    )

    loggers = [
        pl.loggers.WandbLogger(project="road-segmentation", name="DeepLabV3Plus_DeepGlobe"),
    ]

    # Callbacks
    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        pl.callbacks.ModelCheckpoint(
            dirpath=Path("../checkpoints/DeepLabV3Plus_DeepGlobe"),
            filename="best_model",
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
        max_epochs=max_epochs,
        logger=loggers,
        # profiler="simple",
        num_sanity_val_steps=3,
        callbacks=callbacks,
    )

    # Запускаем обучение
    trainer.fit(model, datamodule=dm)

    return trainer.callback_metrics.get("val_loss", torch.tensor(float("inf"))).item()


def objective(trial: optuna.Trial):
    """Функция для Optuna"""

    # Выбор параметров
    encoder = trial.suggest_categorical("encoder", ["efficientnet-b1"])
    learning_rate = trial.suggest_float("learning_rate", [1e-3])
    loss_function_name = trial.suggest_categorical("loss_function", ["dice"])
    scheduler_name = trial.suggest_categorical("scheduler", ["cosine"])
    num_frozen_layers = 0  # trial.suggest_int("num_frozen_layers", 0, 20)

    # Запускаем тренировку и возвращаем метрику
    return train_model(encoder, learning_rate, loss_function_name, scheduler_name, num_frozen_layers, max_epochs=3)


if __name__ == "__main__":
    pl.seed_everything(42)

    run_optuna = False  # для запуска оптимизации параметров

    if run_optuna:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20, timeout=9000)
        print("Лучшие параметры:", study.best_params)

    # тренировка с фиксированными параметрами
    else:
        best_params = {
            "encoder": "efficientnet-b1",
            "learning_rate": 1e-3,
            "loss_function_name": "dice",
            "scheduler_name": "cosine",
            "num_frozen_layers": 0,
        }
        train_model(**best_params, max_epochs=12)
