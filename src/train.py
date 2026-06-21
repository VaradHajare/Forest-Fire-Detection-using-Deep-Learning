"""
Training loop for the Forest Fire Detection project.

Implements the full training pipeline including:
- Per-epoch training and validation
- Learning rate scheduling
- Early stopping to prevent overfitting
- Model checkpointing (saving the best model)
- Training history logging to JSON

Training Strategy:
    We use a two-stage approach for transfer learning:

    Stage 1 (Head-only): Freeze the pretrained backbone, train only the new
    classification head. Uses a higher learning rate (1e-3) since the head
    is randomly initialized.

    Stage 2 (Fine-tuning): Unfreeze the last backbone layer (layer4 for ResNet18),
    train end-to-end with a lower learning rate (1e-4) for the backbone to avoid
    destroying pretrained features, and a moderate rate (1e-3) for the head.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dataset import get_dataloaders
from model import get_transfer_model, BaselineCNN, ImprovedCNN, unfreeze_layers, count_parameters


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model: The neural network
        loader: Training DataLoader
        criterion: Loss function
        optimizer: Optimizer (e.g., AdamW)
        device: torch.device (cuda or cpu)

    Returns:
        tuple of (average_loss, accuracy) for this epoch
    """
    model.train()  # Set model to training mode (enables dropout, batch norm updates)

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()  # Clear gradients from previous step
        loss.backward()        # Compute gradients
        optimizer.step()       # Update weights

        # Track metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)  # Get class with highest score
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on validation/test data.

    Args:
        model: The neural network
        loader: Validation/test DataLoader
        criterion: Loss function
        device: torch.device

    Returns:
        tuple of (average_loss, accuracy)
    """
    model.eval()  # Set to evaluation mode (disables dropout, uses running batch norm stats)

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation for efficiency
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


class EarlyStopping:
    """
    Stops training when validation loss stops improving.

    Why early stopping?
    With a small dataset, the model can easily memorize the training data
    (overfitting). Early stopping monitors the validation loss and halts
    training when it has not improved for 'patience' consecutive epochs.
    This gives us the model at its best generalization point.
    """

    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"\nEarly stopping triggered after {self.counter} epochs without improvement")


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=25,
    learning_rate=1e-3,
    weight_decay=1e-4,
    patience=7,
    save_dir="models",
    model_name="model",
    device=None,
    param_groups=None,
):
    """
    Full training loop with validation, scheduling, early stopping, and checkpointing.

    Args:
        model: The neural network to train
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        num_epochs: Maximum number of training epochs
        learning_rate: Initial learning rate
        weight_decay: L2 regularization strength (penalizes large weights)
        patience: Early stopping patience
        save_dir: Directory to save model checkpoints
        model_name: Name prefix for saved files
        device: torch.device (auto-detected if None)
        param_groups: Optional list of parameter groups with different learning rates
                     for differential fine-tuning.

    Returns:
        dict containing training history (losses, accuracies per epoch)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = model.to(device)

    # Loss function: CrossEntropyLoss combines LogSoftmax + NLLLoss.
    # It is the standard choice for multi-class (including binary) classification.
    criterion = nn.CrossEntropyLoss()

    # Optimizer: AdamW (Adam with decoupled weight decay).
    # Weight decay is L2 regularization that penalizes large weights,
    # helping prevent overfitting. AdamW handles this more correctly than Adam.
    if param_groups:
        optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    else:
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    # Learning rate scheduler: reduces LR when validation loss plateaus.
    # factor=0.5 means LR is halved, patience=3 means wait 3 epochs before reducing.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    early_stopping = EarlyStopping(patience=patience)

    # Training history for plotting
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch + 1}/{num_epochs} (lr: {current_lr:.2e})")
        print("-" * 50)

        # Training phase
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validation phase
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        # Log metrics
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "model_name": model_name,
            }
            torch.save(checkpoint, save_path / f"{model_name}_best.pth")
            print(f"  ** Saved best model (val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f})")

        # Check early stopping
        early_stopping(val_loss)
        if early_stopping.should_stop:
            break

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed / 60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Save training history
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    history_file = results_dir / f"{model_name}_history.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_file}")

    return history


def plot_training_curves(history, save_path="results/training_curves.png", title="Training Curves"):
    """
    Plots training and validation loss/accuracy curves.

    These curves are essential for diagnosing training issues:
    - If train_loss decreases but val_loss increases: OVERFITTING
    - If both losses are high and not decreasing: UNDERFITTING
    - If both decrease together: GOOD generalization
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=3)
    ax1.plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} - Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2.plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=3)
    ax2.plot(epochs, history["val_acc"], "r-o", label="Val Acc", markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{title} - Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {save_path}")


def main():
    """Main entry point for training from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Train fire detection model")
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["baseline", "improved", "resnet18", "mobilenetv2"],
                        help="Model architecture to train")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Root data directory with train/val/test splits")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Maximum number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--patience", type=int, default=7,
                        help="Early stopping patience")
    parser.add_argument("--fine-tune", action="store_true",
                        help="Fine-tune backbone (unfreeze last layer)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from (for fine-tuning)")

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print("\nLoading data...")
    (train_loader, val_loader, test_loader), class_names = get_dataloaders(
        args.data_dir, batch_size=args.batch_size
    )

    # Create model
    param_groups = None

    print(f"\nCreating model: {args.model}")
    if args.model == "baseline":
        model = BaselineCNN(num_classes=2)
        model_name = "baseline_cnn"
    elif args.model == "improved":
        model = ImprovedCNN(num_classes=2)
        model_name = "improved_cnn"
    else:
        if args.fine_tune and args.checkpoint:
            # Load the head-trained model and unfreeze backbone layers
            model = get_transfer_model(args.model, num_classes=2, freeze_backbone=True)
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            unfreeze_layers(model, name=args.model)
            model_name = f"{args.model}_finetuned"

            # Use differential learning rates:
            # Lower rate for backbone (pretrained), higher for head (new)
            if args.model == "resnet18":
                param_groups = [
                    {"params": model.layer4.parameters(), "lr": args.lr * 0.1},
                    {"params": model.fc.parameters(), "lr": args.lr},
                ]
            elif args.model == "mobilenetv2":
                param_groups = [
                    {"params": model.features[14:].parameters(), "lr": args.lr * 0.1},
                    {"params": model.classifier.parameters(), "lr": args.lr},
                ]
        else:
            model = get_transfer_model(args.model, num_classes=2, freeze_backbone=True)
            model_name = f"{args.model}_frozen"

    trainable, total = count_parameters(model)
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    # Train
    print(f"\nStarting training: {model_name}")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
        model_name=model_name,
        device=device,
        param_groups=param_groups,
    )

    # Plot curves
    plot_training_curves(history, save_path=f"results/{model_name}_curves.png", title=model_name)

    print(f"\nTraining complete! Model saved to models/{model_name}_best.pth")
    print(f"Training curves saved to results/{model_name}_curves.png")


if __name__ == "__main__":
    main()
