"""
Single-image inference for the Forest Fire Detection project.

Usage:
    python src/predict.py --image path/to/image.jpg --model resnet18 --checkpoint models/resnet18_finetuned_best.pth
"""

import sys
from pathlib import Path

import torch
from PIL import Image

# Add src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dataset import get_transforms
from model import get_transfer_model, BaselineCNN, ImprovedCNN


CLASS_NAMES = ["fire", "nofire"]


def load_model(checkpoint_path, model_name="resnet18", num_classes=2, device="cpu"):
    """
    Loads a trained model from a checkpoint.

    Args:
        checkpoint_path: Path to the .pth checkpoint file
        model_name: Architecture name ("baseline", "improved", "resnet18", or "mobilenetv2")
        num_classes: Number of output classes
        device: Device to load the model onto

    Returns:
        model in eval mode
    """
    if model_name == "baseline":
        model = BaselineCNN(num_classes=num_classes)
    elif model_name == "improved":
        model = ImprovedCNN(num_classes=num_classes)
    else:
        model = get_transfer_model(model_name, num_classes=num_classes, freeze_backbone=False)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


def predict_image(model, image_path, device="cpu"):
    """
    Predicts whether an image contains fire or not.

    Args:
        model: Trained model (in eval mode)
        image_path: Path to the image file
        device: torch.device

    Returns:
        class_name: Predicted class ("fire" or "nofire")
        confidence: Prediction confidence (0-1)
        probabilities: dict mapping class names to probabilities
    """
    # Load and preprocess image using the same transforms as evaluation
    transform = get_transforms("test")
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    class_name = CLASS_NAMES[predicted.item()]
    confidence_val = confidence.item()

    probabilities = {name: probs[0][i].item() for i, name in enumerate(CLASS_NAMES)}

    return class_name, confidence_val, probabilities


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Predict fire in a single image")
    parser.add_argument("--image", type=str, required=True, help="Path to the image")
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["baseline", "resnet18", "mobilenetv2"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.model, device=device)

    print(f"Predicting: {args.image}")
    class_name, confidence, probabilities = predict_image(model, args.image, device)

    print(f"\nPrediction: {class_name}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Probabilities: {probabilities}")


if __name__ == "__main__":
    main()
