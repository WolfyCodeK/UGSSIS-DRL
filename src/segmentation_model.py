import torch.nn as nn
import segmentation_models_pytorch as smp

class ResNetDeepLab(nn.Module):
    def __init__(self, num_classes=3, backbone_name='resnet50', pretrained_backbone=True):
        super().__init__()
        if pretrained_backbone:
            encoder_weights = 'imagenet'
        else:
            encoder_weights = None

        self.model = smp.DeepLabV3Plus(
            encoder_name=backbone_name,
            encoder_weights=encoder_weights,
            in_channels=5,
            classes=num_classes,
        )

        print(f"Initialised DeepLabV3+ with {backbone_name} backbone using segmentation-models-pytorch.")
        print(f"  Encoder weights: {'ImageNet' if pretrained_backbone else 'None'}")
        print(f"  Input channels: 5")
        print(f"  Output classes: {num_classes}")

    def forward(self, x):
        return self.model(x)