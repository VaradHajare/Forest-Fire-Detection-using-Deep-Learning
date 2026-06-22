"""
Evaluation module for the Forest Fire Detection project.

Generates comprehensive evaluation metrics including:
- Classification report (precision, recall, F1 per class)
- Confusion matrix
- ROC-AUC curve
- Misclassified image analysis
- Model comparison table

Key insight for fire detection: Recall on the fire class is more important
than overall accuracy. A missed fire (false negative) could mean lives lost,
while a false alarm (false positive) is merely inconvenient.
"""

import sys
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
)

# Add src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dataset import get_dataloaders, IMAGENET_MEAN, IMAGENET_STD
from model import get_transfer_model, BaselineCNN, ImprovedCNN


def evaluate_model(model, test_loader, device):
    """
    Runs inference on the test set and collects predictions.

    Args:
        model: Trained model
        test_loader: Test DataLoader
        device: torch.device

    Returns:
        y_true: numpy array of true labels
        y_pred: numpy array of predicted labels
        y_prob: numpy array of prediction probabilities (for ROC curve)
    """
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


def generate_classification_report(y_true, y_pred, y_prob, class_names, save_dir="results", model_name="model"):
    """
    Generates and saves a full classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities [N, num_classes]
        class_names: List of class names
        save_dir: Directory to save results
        model_name: Name prefix for saved files

    Returns:
        dict of metrics
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Text classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(f"\nClassification Report ({model_name}):")
    print(report)

    # Compute individual metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )

    # ROC-AUC (use probability of the positive class).
    # For binary classification, we use the probability of the fire class.
    fire_idx = class_names.index("fire") if "fire" in class_names else 0
    try:
        auc = roc_auc_score(y_true, y_prob[:, fire_idx], multi_class="ovr")
    except ValueError:
        auc = 0.0

    metrics = {
        "model_name": model_name,
        "accuracy": float(accuracy),
        "roc_auc": float(auc),
        "per_class": {},
    }

    for i, name in enumerate(class_names):
        metrics["per_class"][name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    # Save metrics to JSON
    metrics_file = save_dir / f"{model_name}_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_file}")

    # Save text report
    report_file = save_dir / f"{model_name}_report.txt"
    with open(report_file, "w") as f:
        f.write(f"Classification Report: {model_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
        f.write(f"\nROC-AUC: {auc:.4f}\n")
        f.write(f"\nNote: For fire detection, recall on the 'fire' class is\n")
        f.write(f"especially important. A missed fire (false negative) is\n")
        f.write(f"far more costly than a false alarm (false positive).\n")

    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names, save_path="results/confusion_matrix.png", title="Confusion Matrix"):
    """
    Creates and saves a confusion matrix heatmap.

    The confusion matrix shows:
    - True Positives (correctly predicted fire)
    - True Negatives (correctly predicted no-fire)
    - False Positives (predicted fire, actually no-fire) = false alarm
    - False Negatives (predicted no-fire, actually fire) = MISSED FIRE (dangerous!)
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, annot_kws={"size": 14}
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_roc_curve(y_true, y_prob, class_names, save_path="results/roc_curve.png", title="ROC Curve"):
    """
    Plots the ROC (Receiver Operating Characteristic) curve.

    The ROC curve plots True Positive Rate vs False Positive Rate at various
    classification thresholds. AUC (Area Under Curve) summarizes performance:
    - AUC = 1.0: perfect classifier
    - AUC = 0.5: random guessing
    """
    fire_idx = class_names.index("fire") if "fire" in class_names else 0

    fpr, tpr, thresholds = roc_curve(y_true, y_prob[:, fire_idx], pos_label=fire_idx)
    auc = roc_auc_score(y_true, y_prob[:, fire_idx])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC Curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC curve saved to {save_path}")


def show_misclassified(model, test_loader, device, class_names, n=16, save_path="results/misclassified.png"):
    """
    Displays the most confidently misclassified images.

    This is crucial for understanding model failures. Common patterns in fire detection:
    - Sunsets/sunrises confused for fire (similar orange/red colors)
    - Dense red/orange foliage misclassified as fire
    - Smoke-only scenes (no visible flame) missed by the model
    - Images with artificial red/orange objects triggering false positives
    """
    model.eval()

    misclassified = []

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            max_probs, predicted = torch.max(probs, 1)

            # Find misclassified samples
            wrong_mask = predicted != labels
            if wrong_mask.any():
                wrong_indices = wrong_mask.nonzero(as_tuple=True)[0]
                for idx in wrong_indices:
                    misclassified.append({
                        "image": images[idx].cpu(),
                        "true_label": labels[idx].item(),
                        "pred_label": predicted[idx].item(),
                        "confidence": max_probs[idx].item(),
                    })

    if not misclassified:
        print("No misclassified images found! The model got everything right.")
        return

    # Sort by confidence (most confidently wrong first)
    misclassified.sort(key=lambda x: x["confidence"], reverse=True)
    n = min(n, len(misclassified))

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(n):
        ax = axes[i]
        item = misclassified[i]

        # Denormalize for display
        img = item["image"] * std + mean
        img = img.clamp(0, 1)
        img = img.permute(1, 2, 0).numpy()

        ax.imshow(img)
        true_name = class_names[item["true_label"]]
        pred_name = class_names[item["pred_label"]]
        ax.set_title(
            f"True: {true_name}\nPred: {pred_name} ({item['confidence']:.2f})",
            fontsize=9, color="red"
        )
        ax.axis("off")

    # Hide unused axes
    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.suptitle(f"Top {n} Misclassified Images (sorted by confidence)", fontsize=14)
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Misclassified images saved to {save_path} ({len(misclassified)} total misclassified)")


def compare_models(results_dir="results", save_path="results/model_comparison.png"):
    """
    Generates a comparison table of all trained models.
    """
    results_dir = Path(results_dir)
    metrics_files = list(results_dir.glob("*_metrics.json"))

    if not metrics_files:
        print("No metrics files found!")
        return

    all_metrics = []
    for f in sorted(metrics_files):
        with open(f) as fh:
            all_metrics.append(json.load(fh))

    # Print comparison table
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    header = f"{'Model':<25} {'Accuracy':<10} {'AUC':<10} {'Fire Recall':<12} {'Fire F1':<10}"
    print(header)
    print("-" * 80)

    model_names = []
    accuracies = []
    fire_recalls = []
    fire_f1s = []

    for m in all_metrics:
        fire_recall = m["per_class"].get("fire", {}).get("recall", 0)
        fire_f1 = m["per_class"].get("fire", {}).get("f1", 0)
        print(f"{m['model_name']:<25} {m['accuracy']:<10.4f} {m['roc_auc']:<10.4f} {fire_recall:<12.4f} {fire_f1:<10.4f}")

        model_names.append(m["model_name"])
        accuracies.append(m["accuracy"])
        fire_recalls.append(fire_recall)
        fire_f1s.append(fire_f1)

    # Bar chart comparison
    if len(all_metrics) > 1:
        # Try to use a cleaner style if available
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            pass

        x = np.arange(len(model_names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Modern color palette
        colors = ['#3B82F6', '#EF4444', '#10B981']
        
        rects1 = ax.bar(x - width, accuracies, width, label="Accuracy", color=colors[0], edgecolor='white', linewidth=1)
        rects2 = ax.bar(x, fire_recalls, width, label="Fire Recall", color=colors[1], edgecolor='white', linewidth=1)
        rects3 = ax.bar(x + width, fire_f1s, width, label="Fire F1", color=colors[2], edgecolor='white', linewidth=1)

        ax.set_xlabel("Model Architecture", fontsize=12, fontweight='bold', labelpad=10)
        ax.set_ylabel("Score", fontsize=12, fontweight='bold', labelpad=10)
        ax.set_title("Performance Comparison Across Models", fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        
        # Clean up model names
        clean_names = [n.replace("_", " ").title().replace("Cnn", "CNN") for n in model_names]
        ax.set_xticklabels(clean_names, fontsize=11)
        ax.legend(fontsize=11, loc='upper left', bbox_to_anchor=(1, 1))
        
        # Focus the y-axis on the relevant range to highlight differences
        ax.set_ylim(0.90, 1.01)
        
        # Add value labels on top of bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 4),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
        
        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)

        ax.grid(True, alpha=0.3, axis="y", linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"\nComparison chart saved to {save_path}")


def main():
    """Main entry point for evaluation from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate fire detection model")
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["baseline", "improved", "resnet18", "mobilenetv2"],
                        help="Model architecture")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Root data directory with test/ split")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--compare", action="store_true",
                        help="Run model comparison across all saved metrics")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    (_, _, test_loader), class_names = get_dataloaders(args.data_dir, batch_size=args.batch_size)

    # Load model
    if args.model == "baseline":
        model = BaselineCNN(num_classes=2)
    elif args.model == "improved":
        model = ImprovedCNN(num_classes=2)
    else:
        model = get_transfer_model(args.model, num_classes=2, freeze_backbone=False)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    model_name = checkpoint.get("model_name", args.model)

    # Evaluate
    print(f"\nEvaluating {model_name} on test set...")
    y_true, y_pred, y_prob = evaluate_model(model, test_loader, device)

    # Generate reports
    metrics = generate_classification_report(
        y_true, y_pred, y_prob, class_names,
        save_dir="results", model_name=model_name
    )

    plot_confusion_matrix(
        y_true, y_pred, class_names,
        save_path=f"results/{model_name}_confusion_matrix.png",
        title=f"Confusion Matrix - {model_name}"
    )

    plot_roc_curve(
        y_true, y_prob, class_names,
        save_path=f"results/{model_name}_roc_curve.png",
        title=f"ROC Curve - {model_name}"
    )

    show_misclassified(
        model, test_loader, device, class_names,
        n=16, save_path=f"results/{model_name}_misclassified.png"
    )

    # Model comparison
    if args.compare:
        compare_models()


if __name__ == "__main__":
    main()
