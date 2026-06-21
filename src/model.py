"""
Model definitions for the Forest Fire Detection project.

Contains:
1. BaselineCNN: A simple 3-block CNN trained from scratch (establishes lower bound)
2. Transfer learning models: ResNet18/MobileNetV2 pretrained on ImageNet

Transfer Learning Explained:
    Instead of training a CNN from scratch on our small dataset (~1900 images),
    we take a model already trained on ImageNet (1.2M images, 1000 classes).
    The early layers have learned universal visual features (edges, textures, shapes)
    that are useful for almost any image task. We keep these "frozen" (do not update them)
    and only train a new classification head for our specific task (fire vs no-fire).

    This works because:
    - Early CNN layers learn generic features (edges, corners, textures)
    - Middle layers learn more complex patterns (shapes, object parts)
    - Only the final layers are task-specific (ImageNet classes vs fire/nofire)

    Fine-tuning goes one step further: after training the head, we "unfreeze" some
    of the later backbone layers and train them with a very small learning rate.
    This lets the model adapt its feature extraction slightly to our specific domain
    (fire/forest images) without destroying the useful pretrained features.
"""

import torch
import torch.nn as nn
from torchvision import models


class BaselineCNN(nn.Module):
    """
    Simple 3-block CNN for binary classification.

    This serves as a baseline: a lower bound on performance to demonstrate
    the value of transfer learning. It is intentionally simple (3 convolutional
    blocks followed by a classifier head).

    Architecture:
        Block 1: Conv(3->32) -> BatchNorm -> ReLU -> MaxPool
        Block 2: Conv(32->64) -> BatchNorm -> ReLU -> MaxPool
        Block 3: Conv(64->128) -> BatchNorm -> ReLU -> MaxPool
        Classifier: AdaptiveAvgPool -> Flatten -> FC(128->64) -> ReLU -> Dropout -> FC(64->2)
    """

    def __init__(self, num_classes=2):
        super(BaselineCNN, self).__init__()

        # Feature extraction blocks.
        # Each block: convolution -> batch normalization -> activation -> pooling.
        # BatchNorm normalizes activations between layers, helping training stability.
        self.features = nn.Sequential(
            # Block 1: Input 3 channels (RGB) -> 32 feature maps
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Halves spatial dimensions: 224->112

            # Block 2: 32 -> 64 feature maps
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112->56

            # Block 3: 64 -> 128 feature maps
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56->28
        )

        # Adaptive average pooling reduces any spatial size to 1x1.
        # This makes the model input-size agnostic.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # Randomly zeros 30% of neurons during training to reduce overfitting
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


class ImprovedCNN(nn.Module):
    """
    Improved CNN for binary classification.
    Features 5 convolutional blocks, double convolutions (VGG-style) in later blocks, 
    and stronger regularization to improve upon the simple BaselineCNN.
    """
    def __init__(self, num_classes=2):
        super(ImprovedCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 32 filters
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 64 filters
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 128 filters
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 256 filters
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 5: 512 filters
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


def get_transfer_model(name="resnet18", num_classes=2, freeze_backbone=True):
    """
    Creates a pretrained model with a new classification head for binary classification.

    What happens here:
    1. Load a model pretrained on ImageNet (1.2 million images, 1000 classes)
    2. Optionally freeze all backbone layers (so their weights will not change during training)
    3. Replace the final classification layer with a new one for our 2 classes

    Why freeze the backbone?
    - With only ~1900 images, updating all 11M+ parameters would likely overfit
    - The pretrained features (edges, textures, shapes) are already excellent
    - We only need to train the new head (~1000 parameters) to map those features
      to fire/no-fire predictions

    Args:
        name: Model architecture ("resnet18" or "mobilenetv2")
        num_classes: Number of output classes (2 for fire/nofire)
        freeze_backbone: If True, freeze all pretrained layers

    Returns:
        Modified model ready for training
    """
    if name == "resnet18":
        # ResNet18: ~11.7M parameters, good balance of accuracy and size.
        # Uses residual connections (skip connections) that help gradient flow
        # during training, making it easier to train deeper networks.
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        # Replace the final fully connected layer.
        # Original: Linear(512, 1000) for ImageNet's 1000 classes.
        # New: Dropout + Linear(512, 2) for our binary classification.
        num_features = model.fc.in_features  # 512 for ResNet18
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes),
        )

    elif name == "mobilenetv2":
        # MobileNetV2: ~3.4M parameters, designed for mobile/edge deployment.
        # Uses depthwise separable convolutions (cheaper than standard convolutions)
        # and inverted residual blocks.
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        # MobileNetV2's classifier is a Sequential with Dropout + Linear
        num_features = model.classifier[1].in_features  # 1280 for MobileNetV2
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes),
        )
    else:
        raise ValueError(f"Unknown model: {name}. Choose 'resnet18' or 'mobilenetv2'.")

    return model


def unfreeze_layers(model, name="resnet18", layer_name="layer4"):
    """
    Unfreezes specific backbone layers for fine-tuning.

    After training the classification head with a frozen backbone, we can
    "unfreeze" (enable gradient updates for) some of the later backbone layers.
    This lets the model fine-tune its feature extraction for our specific domain.

    We only unfreeze the LAST few layers because:
    - Early layers have generic features (edges, textures) that work for any task
    - Later layers have more task-specific features that benefit from adaptation
    - Unfreezing too many layers risks overfitting on our small dataset

    Important: Use a LOWER learning rate for unfrozen backbone layers (e.g., 1e-4)
    than for the classification head (e.g., 1e-3). This prevents large gradients
    from destroying the useful pretrained features.

    Args:
        model: The model to partially unfreeze
        name: Model architecture name
        layer_name: Which layer to unfreeze (default "layer4" for ResNet18)
    """
    if name == "resnet18":
        # ResNet18 has: conv1, bn1, layer1, layer2, layer3, layer4, fc
        # We typically unfreeze layer4 (the last residual block)
        target_layer = getattr(model, layer_name, None)
        if target_layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in model")

        for param in target_layer.parameters():
            param.requires_grad = True

    elif name == "mobilenetv2":
        # MobileNetV2 has features[0] through features[18].
        # Unfreeze the last few feature blocks (features[14:]).
        for param in model.features[14:].parameters():
            param.requires_grad = True

    print(f"Unfroze {layer_name} for fine-tuning")
    trainable, total = count_parameters(model)
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


def count_parameters(model):
    """
    Counts trainable and total parameters in a model.

    Returns:
        tuple of (trainable_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
