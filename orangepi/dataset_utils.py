from orangepi.imports import *
from orangepi.__init__ import *


def split_dataset(dataset_path, train_ratio=0.7, test_ratio=0.2, valid_ratio=0.1):
    """
    Разделяет файлы из dataset_path на train, test и valid в указанном соотношении.
    
    Args:
        dataset_path (str): Путь к папке с исходными изображениями и масками.
        train_ratio (float): Доля данных для train.
        test_ratio (float): Доля данных для test.
        valid_ratio (float): Доля данных для valid.

    """
    assert train_ratio + test_ratio + valid_ratio == 1, "Сумма долей должна быть равна 1"

    # Создаём выходные директории
    train_dir = os.path.join(dataset_path, "train")
    test_dir = os.path.join(dataset_path, "test")
    valid_dir = os.path.join(dataset_path, "valid")

    for directory in [train_dir, test_dir, valid_dir]:
        os.makedirs(os.path.join(directory, "images"), exist_ok=True)
        os.makedirs(os.path.join(directory, "masks"), exist_ok=True)

    # Получаем список всех номеров файлов (уникальные number)
    files = os.listdir(dataset_path)
    numbers = sorted(set(f.split("_")[0] for f in files if f.endswith("_sat.png")))

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
            image_file = f"{num}_sat.png"
            mask_file = f"{num}_mask.png"
            shutil.move(os.path.join(dataset_path, image_file), os.path.join(target_dir, "images", image_file))
            shutil.move(os.path.join(dataset_path, mask_file), os.path.join(target_dir, "masks", mask_file))

    # Перемещаем файлы в соответствующие директории
    move_files(train_numbers, train_dir)
    move_files(test_numbers, test_dir)
    move_files(valid_numbers, valid_dir)

    print("Разбиение завершено!")


def DeepGlobeDataset(Dataset):
    """Класс для загрузки датасета DeepGlobe"""
    def __init__(
        self, root_path, model_type, transforms=None, processor=None, image_size=1024
    ):
        self.root_path = root_path
        self.model_type = model_type
        self.image_paths = []
        self.mask_paths = []
        self.processor = processor
        self.transform = transforms
        self.image_size = image_size

        img_dir = os.path.join(root_path, "images")
        mask_dir = os.path.join(root_path, "masks")

        images_list = sorted(os.listdir(img_dir))
        masks_list = sorted(os.listdir(mask_dir))
        for i, img_name in enumerate(images_list):
            self.image_paths.append(os.path.join(img_dir, img_name))
            self.mask_paths.append(os.path.join(mask_dir, masks_list[i]))


    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")  # Grayscale

        if self.model_type == "segformer":
            encoding = self.processor(image, return_tensors="pt")
            image = encoding["pixel_values"].squeeze(0)  # [3, H, W]
        else:
            image = self.transform(image)
        mask = mask.resize(
            (self.image_size, self.image_size), resample=Image.NEAREST
        )
        mask = torch.tensor(np.array(mask), dtype=torch.long)

        return image, mask

__all__ = [
    "split_dataset",
    "deepGloobeDataset",
]
