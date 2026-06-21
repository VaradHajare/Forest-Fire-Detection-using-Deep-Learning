"""
Dataset utilities for the Forest Fire Detection project.

Handles data splitting, transforms/augmentation, and DataLoader creation.
Uses torchvision.datasets.ImageFolder for directory-based image loading.
"""

import os
import shutil
import random
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ImageNet normalization stats - required when using pretrained models.
# These are the mean and std of the RGB channels across the ImageNet dataset.
# Pretrained models expect inputs normalized this way because that is how they were trained.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(mode="train", image_size=224):
    """
    Returns image transformation pipeline for the given mode.

    Why different transforms for train vs val/test?
    - Training: We apply random augmentations (flips, rotations, color changes) to
      artificially increase dataset diversity. This helps the model generalize better
      and reduces overfitting, especially important with only ~1900 images.
    - Validation/Test: We apply only deterministic transforms (resize + center crop)
      so that evaluation results are reproducible and reflect true model performance.

    Args:
        mode: One of "train", "val", or "test"
        image_size: Target image size (224 is standard for ImageNet-pretrained models)

    Returns:
        torchvision.transforms.Compose pipeline
    """
    if mode == "train":
        return transforms.Compose([
            # Resize to slightly larger than target, then randomly crop.
            # This forces the model to learn fire at different scales and positions.
            transforms.Resize(256),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),

            # Random horizontal flip: fire can appear on either side.
            # We skip vertical flip because fire does not appear upside down naturally.
            transforms.RandomHorizontalFlip(p=0.5),

            # Small rotation to handle varied camera angles
            transforms.RandomRotation(15),

            # Color jitter is critical for fire detection because lighting varies hugely
            # (daylight, dusk, night, smoke-filtered light). But we keep hue shift small
            # because fire's red/orange color is an important signal we do not want to destroy.
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            ),

            transforms.ToTensor(),

            # Normalize using ImageNet statistics.
            # This is mandatory when using pretrained ImageNet models because the model
            # expects inputs in this range (that is what it saw during pretraining).
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def create_data_splits(raw_dir, output_dir, train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Splits the DeepFire dataset into train/val/test sets with stratification.

    The DeepFire dataset comes with an 80/20 Training/Testing split but no validation set.
    We need a validation set for:
    - Monitoring for overfitting during training (comparing train vs val loss)
    - Early stopping (halt training when val loss stops improving)
    - Hyperparameter tuning without contaminating the test set

    We re-split ALL images into 70/15/15 train/val/test with stratification
    (equal class proportions in each split).

    Args:
        raw_dir: Path to the raw DeepFire dataset (contains Training/ and Testing/ dirs)
        output_dir: Path to write the split data (will create train/, val/, test/ subdirs)
        train_ratio: Fraction of data for training (default 0.70)
        val_ratio: Fraction of data for validation (default 0.15)
        seed: Random seed for reproducibility

    Returns:
        dict with split counts: {"train": N, "val": N, "test": N}
    """
    random.seed(seed)
    np.random.seed(seed)

    test_ratio = 1.0 - train_ratio - val_ratio
    assert test_ratio > 0, "train_ratio + val_ratio must be less than 1.0"

    # Collect all image paths and labels from both Training/ and Testing/ folders
    all_images = []
    all_labels = []

    for split_folder in ["Training", "Testing"]:
        split_path = Path(raw_dir) / split_folder
        if not split_path.exists():
            raise FileNotFoundError(f"Expected folder not found: {split_path}")

        if split_folder == "Training":
            for class_name in ["fire", "nofire"]:
                class_path = split_path / class_name
                if not class_path.exists():
                    raise FileNotFoundError(f"Expected class folder not found: {class_path}")

                for img_file in class_path.iterdir():
                    if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                        all_images.append(str(img_file))
                        all_labels.append(class_name)
        else:
            # Testing folder has images directly, prefixed with 'fire' or 'nofire'
            for img_file in split_path.iterdir():
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    name = img_file.name.lower()
                    if name.startswith("nofire"):
                        all_images.append(str(img_file))
                        all_labels.append("nofire")
                    elif name.startswith("fire"):
                        all_images.append(str(img_file))
                        all_labels.append("fire")

    print(f"Found {len(all_images)} total images")
    print(f"  fire: {all_labels.count('fire')}, nofire: {all_labels.count('nofire')}")

    # Stratified split: first split off test, then split remaining into train/val.
    # Stratification ensures each split has the same fire/nofire ratio as the full dataset.
    train_imgs, temp_imgs, train_lbls, temp_lbls = train_test_split(
        all_images, all_labels,
        test_size=(val_ratio + test_ratio),
        stratify=all_labels,
        random_state=seed
    )

    relative_val = val_ratio / (val_ratio + test_ratio)
    val_imgs, test_imgs, val_lbls, test_lbls = train_test_split(
        temp_imgs, temp_lbls,
        test_size=(1 - relative_val),
        stratify=temp_lbls,
        random_state=seed
    )

    # Copy files into the new directory structure
    splits = {
        "train": (train_imgs, train_lbls),
        "val": (val_imgs, val_lbls),
        "test": (test_imgs, test_lbls),
    }

    counts = {}
    for split_name, (images, labels) in splits.items():
        for img_path, label in zip(images, labels):
            dest_dir = Path(output_dir) / split_name / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / Path(img_path).name

            # Avoid copying if file already exists (idempotent)
            if not dest_file.exists():
                shutil.copy2(img_path, dest_file)

        counts[split_name] = len(images)
        fire_count = labels.count("fire")
        nofire_count = labels.count("nofire")
        print(f"  {split_name}: {len(images)} images (fire: {fire_count}, nofire: {nofire_count})")

    return counts


def get_dataloaders(data_dir, batch_size=32, num_workers=2, image_size=224):
    """
    Creates DataLoaders for train, val, and test splits.

    Uses torchvision.datasets.ImageFolder which automatically:
    - Assigns integer labels based on alphabetical folder order (fire=0, nofire=1)
    - Loads images as PIL Images

    Args:
        data_dir: Root directory containing train/, val/, test/ subdirectories
        batch_size: Number of images per batch (32 is a good default for ~1900 images)
        num_workers: Number of parallel data loading workers
        image_size: Target image size

    Returns:
        tuple of (train_loader, val_loader, test_loader), class_names list
    """
    data_dir = Path(data_dir)

    # Create datasets with appropriate transforms
    train_dataset = datasets.ImageFolder(
        root=data_dir / "train",
        transform=get_transforms("train", image_size)
    )

    val_dataset = datasets.ImageFolder(
        root=data_dir / "val",
        transform=get_transforms("val", image_size)
    )

    test_dataset = datasets.ImageFolder(
        root=data_dir / "test",
        transform=get_transforms("test", image_size)
    )

    # ImageFolder assigns labels alphabetically: fire=0, nofire=1
    class_names = train_dataset.classes
    print(f"Classes: {class_names}")
    print(f"Class to index mapping: {train_dataset.class_to_idx}")
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create DataLoaders
    # shuffle=True for training to randomize batch composition each epoch
    # shuffle=False for val/test for reproducible evaluation
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Speeds up CPU-to-GPU transfer
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return (train_loader, val_loader, test_loader), class_names


def verify_images(data_dir):
    """
    Checks for corrupt images that cannot be opened by PIL.

    Args:
        data_dir: Directory to scan recursively for images

    Returns:
        list of paths to corrupt images
    """
    corrupt = []
    data_path = Path(data_dir)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

    for img_path in data_path.rglob("*"):
        if img_path.suffix.lower() in image_extensions:
            try:
                img = Image.open(img_path)
                img.verify()  # Verify the image can be decoded
            except Exception as e:
                print(f"Corrupt image: {img_path} - {e}")
                corrupt.append(str(img_path))

    if corrupt:
        print(f"\nFound {len(corrupt)} corrupt images!")
    else:
        print("All images verified successfully.")

    return corrupt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare the forest fire dataset")
    parser.add_argument("--raw-dir", type=str, default="data/raw/forest-fire-dataset",
                        help="Path to the raw DeepFire dataset")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory for train/val/test splits")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--verify", action="store_true",
                        help="Verify images for corruption before splitting")

    args = parser.parse_args()

    if args.verify:
        print("Verifying images...")
        corrupt = verify_images(args.raw_dir)
        if corrupt:
            print("Remove corrupt images before proceeding.")
            exit(1)

    print("\nCreating data splits...")
    counts = create_data_splits(args.raw_dir, args.output_dir, seed=args.seed)
    print(f"\nDone! Split counts: {counts}")
