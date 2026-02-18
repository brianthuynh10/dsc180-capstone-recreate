import torch.nn as nn
from torchvision import models

def make_vgg16_model(out_dim=1, pretrained=True, freeze=False):
    """
    VGG16 regression model for grayscale X-rays.
    Resolution-agnostic (supports 256×256).
    """
    # --- Load backbone ---
    if pretrained:
        backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    else:
        backbone = models.vgg16(weights=None)

    # --- Convert first conv to grayscale ---
    w = backbone.features[0].weight.data.mean(dim=1, keepdim=True)
    backbone.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
    backbone.features[0].weight.data = w

    # --- Freeze or unfreeze backbone ---
    for p in backbone.features.parameters():
        p.requires_grad = not freeze

    # --- Make spatial size resolution-agnostic ---
    backbone.avgpool = nn.AdaptiveAvgPool2d((7, 7))

    # --- Regression head ---
    backbone.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(1024, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, out_dim)
    )

    return backbone


def make_resnet50_model(out_dim=1, pretrained=True, freeze=False):
    """
    ResNet50 regression model for grayscale X-rays.
    Resolution-agnostic by design.
    """
    # --- Load backbone ---
    if pretrained:
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )
    else:
        backbone = models.resnet50(weights=None)

    # --- Convert first conv to grayscale ---
    # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    w = backbone.conv1.weight.data.mean(dim=1, keepdim=True)
    backbone.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    backbone.conv1.weight.data = w

    # --- Freeze or unfreeze backbone ---
    for p in backbone.parameters():
        p.requires_grad = not freeze

    # --- Regression head ---
    # ResNet50 outputs 2048-d features after global avg pool
    backbone.fc = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(1024, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, out_dim)
    )

    return backbone
