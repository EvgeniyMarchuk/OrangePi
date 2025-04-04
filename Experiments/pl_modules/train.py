import torch
import torch.nn as nn
import pytorch_lightning as pl
import optuna
import segmentation_models_pytorch as smp
from pathlib import Path
from dataset import RoadSegmentationDataModule
from model import SegmentationModel


def train_model(
        encoder,
        learning_rate,
        loss_function_name,
        num_frozen_layers,
        checkpoint_path="../checkpoints/PSPNet/pspnet.ckpt",
        output_path="../checkpoints/PSPNet",
        max_epochs=10
    ):
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
        train_path=Path("../data/road_simulator/train"),
        val_path=Path("../data/road_simulator/val"),
        test_path=Path("../data/road_simulator/val"),
        batch_size=8,
    )

    # Создаём модель
    print(f"🔹 Загружаем модель из чекпойнта: {checkpoint_path}")
    model = SegmentationModel.load_from_checkpoint(
        checkpoint_path,
        encoder=encoder,
        lr=learning_rate,
        model_encoder=encoder,
        num_classes=1,
        loss_fn=loss_function,
        freeze_layers=num_frozen_layers,
    )

    loggers = [
        pl.loggers.WandbLogger(project="simulator-roads", name=f"PSPNet_600_tuning"),
    ]

    # Callbacks
    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        pl.callbacks.ModelCheckpoint(
            dirpath=Path(output_path),
            filename="pspnet_600_2",
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
        num_sanity_val_steps=3,
        callbacks=callbacks,
    )
    # Запускаем обучение
    trainer.fit(model, datamodule=dm)

    return trainer.callback_metrics.get("val_loss", torch.tensor(float("inf"))).item()


def objective(trial: optuna.Trial):
    """Функция для Optuna"""

    # Выбор параметров
    encoder = trial.suggest_categorical("encoder", ["efficientnet-b3"])
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
            "encoder": "efficientnet-b3",
            "learning_rate": 3e-4,
            "loss_function_name": "dice",
            "num_frozen_layers": 25,
        }

        train_model(
            **best_params,
            checkpoint_path="../checkpoints/PSPNet/pspnet.ckpt",
            output_path="../checkpoints/PSPNet",
            max_epochs=50
        )
