from orangepi.imports import *
from orangepi.__init__ import *


def split_dataset(src_path, dest_path, train_ratio=0.7, test_ratio=0.2, valid_ratio=0.1):
    """
    Разделяет файлы из src_path на train, test и valid в указанном соотношении.
    
    Args:
        src_path (str): Путь к папке с исходными изображениями и масками.
        train_ratio (float): Доля данных для train.
        test_ratio (float): Доля данных для test.
        valid_ratio (float): Доля данных для valid.

    """
    assert np.isclose(train_ratio + test_ratio + valid_ratio, 1.0), "Сумма долей должна быть равна 1"

    # Создаём выходные директории
    train_dir = os.path.join(dest_path, "train")
    test_dir = os.path.join(dest_path, "test")
    valid_dir = os.path.join(dest_path, "valid")

    for directory in [train_dir, test_dir, valid_dir]:
        os.makedirs(os.path.join(directory, "images"), exist_ok=True)
        os.makedirs(os.path.join(directory, "masks"), exist_ok=True)

    # Получаем список всех номеров файлов (уникальные number)
    files = os.listdir(src_path)
    numbers = sorted(set(f.split("_")[0] for f in files if f.endswith("_sat.jpg")))

    # Перемешиваем номера для случайного разбиения
    random.shuffle(numbers)

    # Определяем границы разбиения
    train_count = int(len(numbers) * train_ratio)
    test_count = int(len(numbers) * test_ratio)

    train_numbers = numbers[:train_count]
    test_numbers = numbers[train_count:train_count + test_count]
    valid_numbers = numbers[train_count + test_count:]

    # Функция для перемещения файлов
    def move_files(numbers_list, target_dir):
        for num in numbers_list:
            image_file = f"{num}_sat.jpg"
            mask_file = f"{num}_mask.png"
            shutil.move(os.path.join(src_path, image_file), os.path.join(target_dir, "images", image_file))
            shutil.move(os.path.join(src_path, mask_file), os.path.join(target_dir, "masks", mask_file))

    # Перемещаем файлы в соответствующие директории
    move_files(train_numbers, train_dir)
    move_files(test_numbers, test_dir)
    move_files(valid_numbers, valid_dir)

    print("Разбиение завершено!")


class DeepGlobeDataset(Dataset):
    def __init__(
        self, root_dir, model_type, transforms=None, processor=None, image_size=512
    ):
        self.model_type = model_type
        self.transforms = transforms
        self.processor = processor
        self.image_size = image_size

        images_dir = root_dir / "images"
        masks_dir = root_dir / "masks"

        self.image_paths = sorted(
            [images_dir / file_name for file_name in os.listdir(images_dir)]
        )
        self.mask_paths = sorted(
            [masks_dir / file_name for file_name in os.listdir(masks_dir)]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.model_type == "segformer":
            encoding = self.processor(image, return_tensors="pt")
            image = encoding["pixel_values"].squeeze(0)  # [3, H, W]
            mask = mask.resize(
                (self.image_size, self.image_size), resample=Image.NEAREST
            )
        else:
            image = self.transforms(image)
            mask = mask.resize(
                (self.image_size, self.image_size), resample=Image.NEAREST
            )

        mask = torch.tensor(np.array(mask), dtype=torch.long)
        mask[mask == 255] = 1

        return image, mask
    

class LandDataset(Dataset):
    def __init__(
        self, root_dir, model_type, transforms=None, processor=None, image_size=512
    ):
        self.model_type = model_type
        self.transforms = transforms
        self.processor = processor
        self.image_size = image_size

        images_dir = root_dir / "images"
        masks_dir = root_dir / "masks"

        self.image_paths = sorted(
            [images_dir / file_name for file_name in os.listdir(images_dir)]
        )
        self.mask_paths = sorted(
            [masks_dir / file_name for file_name in os.listdir(masks_dir)]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.model_type == "segformer":
            encoding = self.processor(image, return_tensors="pt")
            image = encoding["pixel_values"].squeeze(0)  # [3, H, W]
            mask = mask.resize(
                (self.image_size, self.image_size), resample=Image.NEAREST
            )
        else:
            image = self.transforms(image)
            mask = mask.resize(
                (self.image_size, self.image_size), resample=Image.NEAREST
            )

        mask = torch.tensor(np.array(mask), dtype=torch.long)
        mask[mask == 255] = 1

        return image, mask


__all__ = [
    "split_dataset",
    "DeepGlobeDataset",
    "LandDataset"
]
