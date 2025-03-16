import numpy as np
from pathlib import Path
import torch
import pytorch_lightning as pl
from .dataset import RoadSegmentationDataModule
from .model import SegmentationModel

def main():
    dm = RoadSegmentationDataModule(
        train_path=Path("../data/train"),
        val_path=Path("../data/valid"),
        test_path=Path("../data/test"),
        batch_size=2,
    )

    model = SegmentationModel.load_from_checkpoint(
        Path("../checkpoints/name-of-model")
    )
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto"
    )

    iou_scores = trainer.predict(model, datamodule=dm)
    print(f"Test iou_score = {np.mean(iou_scores)}")