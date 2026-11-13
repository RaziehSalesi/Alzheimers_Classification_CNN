
import random
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
import os
import torch
from torch.utils.data import Dataset

# 1. Load image paths and labels

def load_image_paths(data_dir):

    data_dir = Path(data_dir)

    image_paths = []
    labels = []

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for class_name in ["AD", "MCI"]:

        class_dir = data_dir / class_name

        if not class_dir.exists():
            raise FileNotFoundError(f"Directory not found: {class_dir}")

        for img_filename in os.listdir(class_dir):
            img_path = class_dir / img_filename
            if img_path.suffix.lower() in valid_extensions:
                image_paths.append(str(img_path))
                labels.append(class_name)

    return image_paths, labels



# 2. Split dataset
def split_dataset(image_paths, labels):

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        image_paths,
        labels,
        test_size=0.05,
        stratify=labels,
        random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.1,
        stratify=y_train_val,
        random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# 3. Data augmentation

def Data_Augmentation(img_path_list, lbl_list):

    augmentation_factors = [
        "horizontal_flip",
        "shearing",
        "shifting",
        "rotation",
        "zooming"
    ]

    augmented_images = []
    augmented_labels = []

    for i in range(len(img_path_list)):

        img = Image.open(img_path_list[i]).convert("L")
        img = img.resize((64, 64))

        label = lbl_list[i]

        # Keep original image
        augmented_images.append(np.array(img))
        augmented_labels.append(label)

        # Generate four augmented images
        for _ in range(4):

            augmentation_factor = random.choice(augmentation_factors)

            if augmentation_factor == "horizontal_flip":
                img1 = transforms.RandomHorizontalFlip()(img)

            elif augmentation_factor == "shearing":
                img1 = transforms.RandomAffine(
                    degrees=0, shear=20
                )(img)

            elif augmentation_factor == "shifting":
                img1 = transforms.RandomAffine(
                    degrees=0, translate=(0.2, 0.2)
                )(img)

            elif augmentation_factor == "rotation":
                img1 = transforms.RandomRotation(
                    degrees=45
                )(img)

            elif augmentation_factor == "zooming":
                img1 = transforms.RandomResizedCrop(
                    size=64, scale=(0.8, 1.2)
                )(img)

            augmented_images.append(np.array(img1))
            augmented_labels.append(label)

    return np.array(augmented_images), np.array(augmented_labels)



# 4. Prepare validation and test images
def Prepare_Images(img_path_list):

    prepared_images = []

    for image_path in img_path_list:

        img = Image.open(image_path).convert("L")

        img = img.resize((64, 64))

        prepared_images.append(np.array(img))

    return np.array(prepared_images)





class_to_idx = {
    "AD": 0,
    "MCI": 1
}


class AdniDataset(Dataset):

    def __init__(self, data_array, label_array):
        self.data_array = data_array
        self.label_array = label_array

    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, index):

        data = self.data_array[index]

        # Add channel dimension: (64, 64) -> (1, 64, 64)
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)

        # Normalize pixel values to [0, 1]
        # data = data / 255.0  


        label = self.label_array[index]
        label = class_to_idx[label]
        label = torch.tensor(label, dtype=torch.long)

        return data, label