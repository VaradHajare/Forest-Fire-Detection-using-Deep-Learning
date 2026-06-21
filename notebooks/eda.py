"""
Exploratory Data Analysis for the Forest Fire Detection project.

This script analyzes the DeepFire dataset to understand:
1. Class distribution (fire vs no-fire balance)
2. Sample images from each class
3. Image size/resolution distribution
4. Corrupt image detection
5. Mean pixel intensity per class

Run after downloading and splitting the dataset:
    python notebooks/eda.py
"""

import sys
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def get_image_stats(data_dir):
    """Collects statistics about all images in the dataset."""
    data_path = Path(data_dir)
    stats = {
        "sizes": [],
        "class_counts": Counter(),
        "corrupt": [],
        "per_class_intensities": {},
    }

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for split in ["train", "val", "test"]:
        split_path = data_path / split
        if not split_path.exists():
            continue

        for class_dir in sorted(split_path.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            if class_name not in stats["per_class_intensities"]:
                stats["per_class_intensities"][class_name] = []

            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() not in image_extensions:
                    continue

                try:
                    img = Image.open(img_file)
                    stats["sizes"].append(img.size)  # (width, height)
                    stats["class_counts"][(split, class_name)] += 1

                    # Sample pixel intensities (for a subset to save time)
                    if len(stats["per_class_intensities"][class_name]) < 200:
                        img_array = np.array(img.convert("RGB"))
                        mean_intensity = img_array.mean(axis=(0, 1))  # Per-channel mean
                        stats["per_class_intensities"][class_name].append(mean_intensity)

                except Exception as e:
                    stats["corrupt"].append((str(img_file), str(e)))

    return stats


def plot_class_distribution(stats, save_dir):
    """Bar chart showing class balance across splits."""
    splits = ["train", "val", "test"]
    classes = sorted(set(cls for _, cls in stats["class_counts"].keys()))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, split in enumerate(splits):
        counts = [stats["class_counts"].get((split, cls), 0) for cls in classes]
        colors = ["#FF5722" if cls == "fire" else "#4CAF50" for cls in classes]

        bars = axes[i].bar(classes, counts, color=colors, edgecolor="white", linewidth=1.5)
        axes[i].set_title(f"{split.capitalize()} Split", fontsize=13, fontweight="bold")
        axes[i].set_ylabel("Number of Images")

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                        str(count), ha="center", va="bottom", fontsize=11, fontweight="bold")

        axes[i].set_ylim(0, max(counts) * 1.15)
        axes[i].grid(axis="y", alpha=0.3)

    plt.suptitle("Class Distribution Across Splits", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: class_distribution.png")


def plot_sample_images(data_dir, save_dir, n_per_class=8):
    """Grid of sample images from each class."""
    data_path = Path(data_dir) / "train"

    fig, axes = plt.subplots(2, n_per_class, figsize=(n_per_class * 2.5, 6))

    for row, class_name in enumerate(["fire", "nofire"]):
        class_path = data_path / class_name
        if not class_path.exists():
            continue

        images = sorted(class_path.iterdir())[:n_per_class]

        for col, img_path in enumerate(images):
            try:
                img = Image.open(img_path).convert("RGB")
                axes[row, col].imshow(img)
            except Exception:
                pass
            axes[row, col].axis("off")

        # Row label
        axes[row, 0].set_ylabel(class_name.upper(), fontsize=12, fontweight="bold",
                                rotation=0, labelpad=50, va="center")

    plt.suptitle("Sample Images from Each Class", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "sample_images.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: sample_images.png")


def plot_image_sizes(stats, save_dir):
    """Distribution of image dimensions."""
    widths = [s[0] for s in stats["sizes"]]
    heights = [s[1] for s in stats["sizes"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(widths, bins=30, color="#2196F3", edgecolor="white", alpha=0.8)
    ax1.set_xlabel("Width (px)")
    ax1.set_ylabel("Count")
    ax1.set_title("Image Width Distribution")
    ax1.grid(axis="y", alpha=0.3)

    ax2.hist(heights, bins=30, color="#FF9800", edgecolor="white", alpha=0.8)
    ax2.set_xlabel("Height (px)")
    ax2.set_ylabel("Count")
    ax2.set_title("Image Height Distribution")
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("Image Size Distribution", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "image_sizes.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Print summary
    unique_sizes = Counter(stats["sizes"])
    print(f"\nImage size summary:")
    print(f"  Unique sizes: {len(unique_sizes)}")
    for size, count in unique_sizes.most_common(5):
        print(f"  {size[0]}x{size[1]}: {count} images")
    print("Saved: image_sizes.png")


def plot_pixel_intensity(stats, save_dir):
    """Mean pixel intensity per channel per class."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    channel_names = ["Red", "Green", "Blue"]
    channel_colors = ["#f44336", "#4caf50", "#2196f3"]

    for ch_idx, (ch_name, ch_color) in enumerate(zip(channel_names, channel_colors)):
        for class_name, intensities in stats["per_class_intensities"].items():
            if not intensities:
                continue
            values = [v[ch_idx] for v in intensities]
            label_color = "#FF5722" if class_name == "fire" else "#4CAF50"
            axes[ch_idx].hist(values, bins=25, alpha=0.6, label=class_name,
                            color=label_color, edgecolor="white")

        axes[ch_idx].set_title(f"{ch_name} Channel", fontsize=12)
        axes[ch_idx].set_xlabel("Mean Intensity (0-255)")
        axes[ch_idx].set_ylabel("Count")
        axes[ch_idx].legend()
        axes[ch_idx].grid(axis="y", alpha=0.3)

    plt.suptitle("Mean Pixel Intensity Distribution by Class", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "pixel_intensity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: pixel_intensity.png")


def main():
    data_dir = Path(__file__).parent.parent / "data"
    save_dir = Path(__file__).parent.parent / "results"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS - Forest Fire Detection")
    print("=" * 60)

    # Check if data exists
    if not (data_dir / "train").exists():
        print(f"\nERROR: No train/ directory found at {data_dir}")
        print("Run data splitting first: python src/dataset.py")
        return

    # Collect stats
    print("\nCollecting image statistics...")
    stats = get_image_stats(data_dir)

    # Print summary
    print(f"\nDataset Summary:")
    print(f"  Total images: {len(stats['sizes'])}")
    print(f"  Corrupt images: {len(stats['corrupt'])}")

    total_per_class = Counter()
    for (split, cls), count in stats["class_counts"].items():
        total_per_class[cls] += count

    for cls, count in sorted(total_per_class.items()):
        print(f"  {cls}: {count} images")

    print(f"\nPer-split breakdown:")
    for split in ["train", "val", "test"]:
        counts_str = ", ".join(
            f"{cls}: {stats['class_counts'].get((split, cls), 0)}"
            for cls in sorted(total_per_class.keys())
        )
        total = sum(stats["class_counts"].get((split, cls), 0) for cls in total_per_class.keys())
        print(f"  {split}: {total} ({counts_str})")

    if stats["corrupt"]:
        print(f"\nCorrupt images:")
        for path, error in stats["corrupt"]:
            print(f"  {path}: {error}")

    # Generate plots
    print("\nGenerating visualizations...")
    plot_class_distribution(stats, save_dir)
    plot_sample_images(data_dir, save_dir)
    plot_image_sizes(stats, save_dir)
    plot_pixel_intensity(stats, save_dir)

    print(f"\nEDA complete! All plots saved to {save_dir}/")


if __name__ == "__main__":
    main()
