import os
import numpy as np
from pathlib import Path

image_height = 480
labels_path = Path("./data/road_simulator/all_new_labels/")
new_labels_path = Path("./data/road_simulator/correct_labels")
label_paths = sorted(os.listdir(labels_path))
for label_path in label_paths:
    new_lines = []
    with open(labels_path / label_path, 'r') as f:
        for line in f.readlines():
            numbers = list(map(float, line.split()))
            y_coords = np.array(numbers[2::2])
            y_coords -= 60 / 480
            y_coords *= 4 / 3
            numbers[2::2] = list(np.clip(y_coords, 0.0, 1.0))
            numbers[0] = int(numbers[0])
            new_lines.append(' '.join(map(str, numbers)) + '\n')
    with open(new_labels_path / label_path, 'w') as f:
        f.writelines(new_lines)
        print(len(new_lines))
        print("Wrote in file", new_labels_path / label_path)

