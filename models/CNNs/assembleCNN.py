import os
import torch
import torch.nn as nn
from torchvision import models


def build_grayscale_cnn(out_dim=1, model_name="resnet50"):
    """
    Reconstructs the grayscale ResNet50 architecture to match the trained model checkpoint.
    """
    if model_name == "resnet50":
        # path for the ResNet weights: 
       
        model = models.resnet50(pretrained=False)
        # --- Convert first conv layer to grayscale ---
        model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        model.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, out_dim)
        )
        # load in the pre-trained model weights
        path= "~/teams/b1/cnn_model_weights/resnet50_best_model.pt"
        path = os.path.expanduser(path)
        model.load_state_dict(
            torch.load(
                path,
                map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
        )

        return model

    elif model_name == "vgg16": 

        model = models.vgg16(pretrained=False)
        model.features[0] = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, out_dim)
        )

        path= "~/teams/b1/cnn_model_weights/vgg16_last_model.pt"
        path = os.path.expanduser(path)
        model.load_state_dict(
            torch.load(
                path,
                map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
        )
    return model




