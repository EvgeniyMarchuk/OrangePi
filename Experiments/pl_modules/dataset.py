import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os


class RoadSegmentationDataset(Dataset):
    def __init__(self, images_path, masks_path, transform=None):
        self.image_paths = sorted(os.listdir(str(images_path)))
        self.mask_paths = sorted(os.listdir(str(masks_path)))
        self.base_images_path = images_path
        self.base_masks_path = masks_path
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.base_images_path, self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.base_masks_path, self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 128).astype("float32")  # Приводим к 0 и 1

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].unsqueeze(0)  # [H, W] -> [1, H, W]

        return image, mask

class RoadSegmentationDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_path,
            val_path,
            test_path,
            batch_size=2
        ):
        super().__init__()
        self.save_hyperparameters()

        self.train_images_path = train_path / "images"
        self.train_masks_path = train_path / "masks"
        self.val_images_path = val_path / "images"
        self.val_masks_path = val_path / "masks"
        self.test_images_path = test_path / "images"
        self.test_masks_path = test_path / "masks"

        self.batch_size = batch_size
        self.transform = A.Compose([
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.RandomRotate90(p=0.4),
            A.CoarseDropout(),
            A.Perspective(scale=(0.0, 0.3)),
            A.ShiftScaleRotate(rotate_limit=(-120, 120)),
            A.ThinPlateSpline(),
            A.Resize(480, 640),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        self.test_transform = A.Compose([
            A.Resize(480, 640),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        self.prepare_data_per_node = False
    
    def prepare_data(self):
        """Используется для подготовки данных, если они расположены не локально в папке
        и нужно загружать данные извне, например"""
        pass

    def setup(self, stage):
        """Запускается данный метод перед тем как запускается процесс обучения / валидации / теста"""
        if stage == "fit" or stage is None:
            self.train_dataset = RoadSegmentationDataset(
                self.train_images_path,
                self.train_masks_path,
                transform=self.transform
            )
            self.val_dataset = RoadSegmentationDataset(
                self.val_images_path,
                self.val_masks_path,
                transform=self.transform
            )

        if stage == "test" or stage is None:
            self.test_dataset = RoadSegmentationDataset(
                self.test_images_path,
                self.test_masks_path,
                transform=self.test_transform
            )

    def train_dataloader(self):
        """Создаем dataloader для тренировки"""
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )
        return train_dataloader
    
    def val_dataloader(self):
        """Создаем dataloader для валидации"""
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )
        return val_dataloader
    # def test_dataloader(self):
    #     """Создаем dataloader для тестирования"""
    #     return DataLoader(
    #         self.test_dataset,
    #         batch_size=self.batch_size,
    #         shuffle=True,
    #     )
    
    def teardown(self, stage=None):
        """Если хотим, например, очистить данные после прохождения эпохи и т.п."""
        pass
