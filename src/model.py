import torch
import torch.nn as nn
from torchvision import models


def make_vgg16_model(out_dim=1, pretrained=True, freeze=False):
    """
    Create a VGG16-based regression model for grayscale X-ray input.
    Matches the modular design of the ResNet18 setup.
    """
    if pretrained:
        backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    else:
        backbone = models.vgg16(weights=None)

    w = backbone.features[0].weight.data.mean(dim=1, keepdim=True)
    backbone.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
    backbone.features[0].weight.data = w

    if freeze:
        for p in backbone.features.parameters():
            p.requires_grad = False
    else: 
        for p in backbone.features.parameters():
            p.requires_grad = True

    backbone.classifier = nn.Sequential(
        nn.Linear(25088, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(1024, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, out_dim)
    )

    model = backbone
    return model
