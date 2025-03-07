import sys
sys.path.insert(0, "/home/user/Projects/OrangePi")


from orangepi.imports import *
from orangepi.__init__ import *
from orangepi.dataset_utils import DeepGlobeDataset
from orangepi.model_utils import *

processor = None
transform = T.Compose([T.Resize((1024, 1024)), T.ToTensor()])

model = deeplab_model()
model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
model = model.to(DEVICE)

model_type = "deeplab"

dataset_path = Path("./data/DeepGlobeDataset/")
train_dir = dataset_path / "train"
valid_dir = dataset_path / "valid"
test_dir = dataset_path / "test"

train_dataset = DeepGlobeDataset(
    train_dir, model_type, transforms=transform, processor=processor, image_size=1024
)
valid_dataset = DeepGlobeDataset(
    valid_dir, model_type, transforms=transform, processor=processor, image_size=1024
)
test_dataset = DeepGlobeDataset(
    test_dir, model_type, transforms=transform, processor=processor, image_size=1024
)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=True)

epochs = 10
lr = 1e-3

# Запускаем обучение
model_save_name = "DeeplabV3_DeepGlobe_V1"
assert model_save_name is not None, "Enter model's name for saving"

torch.cuda.empty_cache()  # Очищаем кеш
criterion = nn.BCEWithLogitsLoss()
best_model, loss_history, iou_score_history, dice_score_history = train(
    model,
    train_loader,
    valid_loader,
    criterion,
    lr,
    epochs,
    model_type,
    verbose=True,
    is_scheduler=False,
)

# --- 6. Сохранение модели ---
torch.save(best_model.state_dict(), f"../models/{model_save_name}.pth")
print("Модель сохранена!")

