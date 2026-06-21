"""
Grad-CAM visualization for the Forest Fire Detection project.

Grad-CAM (Gradient-weighted Class Activation Mapping) shows which regions
of an image the model focuses on when making its prediction. This is valuable for:
1. Debugging: Verify the model looks at fire/smoke regions, not background artifacts
2. Trust: Show that predictions are based on meaningful visual evidence
3. Portfolio storytelling: "The model attends to flame regions, not irrelevant background"

How Grad-CAM works (simplified):
1. Pass an image through the model up to a target convolutional layer
2. Compute the gradient of the predicted class score with respect to that layer's output
3. Average the gradients across spatial dimensions to get "importance weights"
4. Multiply each feature map by its importance weight and sum them
5. Apply ReLU (keep only positive influences) and overlay on the original image

We use the pytorch-grad-cam library which handles all of this cleanly.
"""

import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Add src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dataset import get_transforms, IMAGENET_MEAN, IMAGENET_STD
from model import get_transfer_model, BaselineCNN
from predict import load_model, CLASS_NAMES


def get_target_layer(model, model_name="resnet18"):
    """
    Returns the appropriate target layer for Grad-CAM.

    For Grad-CAM, we typically target the last convolutional layer because:
    - It has the richest semantic information (objects, patterns)
    - It retains enough spatial resolution to produce meaningful heatmaps
    - Earlier layers would show low-level features (edges, textures) instead
    """
    if model_name == "resnet18":
        return [model.layer4]
    elif model_name == "mobilenetv2":
        return [model.features[-1]]
    elif model_name == "baseline":
        # For our BaselineCNN, target the last conv block (last Conv2d in features)
        return [model.features[-3]]
    elif model_name == "improved":
        # Target the last layer before pooling
        return [model.features[-3]]
    else:
        raise ValueError(f"Unknown model: {model_name}")


def generate_gradcam(model, image_path, model_name="resnet18", device="cpu", method="gradcam"):
    """
    Generates a Grad-CAM heatmap for a single image.

    Args:
        model: Trained model (in eval mode)
        image_path: Path to the image
        model_name: Architecture name (for selecting target layer)
        device: torch.device
        method: "gradcam" or "eigencam"

    Returns:
        rgb_image: Original image as numpy array (0-1 range)
        heatmap: Grad-CAM heatmap as numpy array
        overlay: Heatmap overlaid on original image
        prediction: (class_name, confidence)
    """
    # Load and preprocess
    transform = get_transforms("test")
    pil_image = Image.open(image_path).convert("RGB")

    # Resize to match model input for overlay
    pil_image = pil_image.resize((224, 224))
    rgb_image = np.array(pil_image) / 255.0  # Normalize to 0-1 for overlay

    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # Get target layer
    target_layers = get_target_layer(model, model_name)

    # Create CAM
    if method == "eigencam":
        cam = EigenCAM(model=model, target_layers=target_layers)
    else:
        cam = GradCAM(model=model, target_layers=target_layers)

    # Generate heatmap (targets=None means use the highest-scoring class)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    heatmap = grayscale_cam[0, :]

    # Create overlay
    overlay = show_cam_on_image(rgb_image.astype(np.float32), heatmap, use_rgb=True)

    # Get prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    prediction = (CLASS_NAMES[predicted.item()], confidence.item())

    return rgb_image, heatmap, overlay, prediction


def visualize_gradcam_grid(model, image_paths, model_name="resnet18", device="cpu",
                           save_path="results/gradcam_visualization.png", method="gradcam"):
    """
    Creates a grid visualization of Grad-CAM results for multiple images.

    Shows original image | Grad-CAM overlay side by side for each image,
    with prediction labels.
    """
    n = len(image_paths)
    fig, axes = plt.subplots(n, 2, figsize=(10, 5 * n))

    if n == 1:
        axes = axes.reshape(1, -1)

    for i, img_path in enumerate(image_paths):
        rgb_image, heatmap, overlay, (pred_class, confidence) = generate_gradcam(
            model, img_path, model_name, device, method
        )

        # Original image
        axes[i, 0].imshow(rgb_image)
        axes[i, 0].set_title("Original", fontsize=11)
        axes[i, 0].axis("off")

        # Grad-CAM overlay
        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title(
            f"Grad-CAM | Pred: {pred_class} ({confidence:.2f})",
            fontsize=11
        )
        axes[i, 1].axis("off")

    plt.suptitle(f"Grad-CAM Visualization ({method.upper()})", fontsize=14)
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grad-CAM visualization saved to {save_path}")


def visualize_by_category(model, test_dir, model_name="resnet18", device="cpu",
                          save_dir="results", n_per_category=4, method="gradcam"):
    """
    Creates separate Grad-CAM visualizations for:
    - True Positives (correctly identified fire)
    - True Negatives (correctly identified no-fire)
    - False Positives (false alarms)
    - False Negatives (missed fires)
    """
    from predict import predict_image

    test_path = Path(test_dir)
    categories = {
        "True Positives (Correct Fire)": [],
        "True Negatives (Correct No-Fire)": [],
        "False Positives (False Alarm)": [],
        "False Negatives (Missed Fire)": [],
    }

    # Classify all test images
    for class_dir in ["fire", "nofire"]:
        class_path = test_path / class_dir
        if not class_path.exists():
            continue

        for img_file in sorted(class_path.iterdir()):
            if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            pred_class, confidence, _ = predict_image(model, str(img_file), device)
            true_class = class_dir

            if true_class == "fire" and pred_class == "fire":
                categories["True Positives (Correct Fire)"].append((str(img_file), confidence))
            elif true_class == "nofire" and pred_class == "nofire":
                categories["True Negatives (Correct No-Fire)"].append((str(img_file), confidence))
            elif true_class == "nofire" and pred_class == "fire":
                categories["False Positives (False Alarm)"].append((str(img_file), confidence))
            elif true_class == "fire" and pred_class == "nofire":
                categories["False Negatives (Missed Fire)"].append((str(img_file), confidence))

    # Visualize each category
    for cat_name, items in categories.items():
        if not items:
            print(f"No images for: {cat_name}")
            continue

        # Sort by confidence and take top n
        items.sort(key=lambda x: x[1], reverse=True)
        selected_paths = [p for p, c in items[:n_per_category]]

        safe_name = cat_name.split("(")[0].strip().lower().replace(" ", "_")
        save_path = Path(save_dir) / f"gradcam_{safe_name}.png"

        print(f"\n{cat_name}: {len(items)} images")
        visualize_gradcam_grid(model, selected_paths, model_name, device, str(save_path), method)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations")
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["baseline", "resnet18", "mobilenetv2"])
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single image (optional)")
    parser.add_argument("--test-dir", type=str, default="data/test",
                        help="Path to test directory for category visualization")
    parser.add_argument("--method", type=str, default="gradcam",
                        choices=["gradcam", "eigencam"])
    parser.add_argument("--n", type=int, default=4,
                        help="Number of images per category")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.model, device=device)

    if args.image:
        # Single image
        rgb_image, heatmap, overlay, (pred, conf) = generate_gradcam(
            model, args.image, args.model, device, args.method
        )
        save_path = "results/gradcam_single.png"

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(rgb_image)
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(overlay)
        axes[1].set_title(f"Pred: {pred} ({conf:.2f})")
        axes[1].axis("off")
        plt.tight_layout()
        Path("results").mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved to {save_path}")
    else:
        # Category visualization
        visualize_by_category(
            model, args.test_dir, args.model, device,
            save_dir="results", n_per_category=args.n, method=args.method
        )


if __name__ == "__main__":
    main()
