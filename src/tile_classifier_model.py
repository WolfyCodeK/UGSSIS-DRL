import torch.nn as nn
from torchvision import models

class TileClassifier(nn.Module):
    def __init__(self, num_classes=1, input_channels=2):
        super().__init__()

        weights = models.ShuffleNet_V2_X0_5_Weights.DEFAULT
        backbone = models.shufflenet_v2_x0_5(weights=weights)

        original_first_conv = backbone.conv1[0]
        assert isinstance(original_first_conv, nn.Conv2d)
        backbone.conv1[0] = nn.Conv2d(
            in_channels=input_channels,
            out_channels=original_first_conv.out_channels,
            kernel_size=original_first_conv.kernel_size,  # type: ignore
            stride=original_first_conv.stride,  # type: ignore
            padding=original_first_conv.padding,  # type: ignore
            bias=original_first_conv.bias is not None
        )

        if input_channels != 3 and weights is not None:
            original_weights = original_first_conv.weight.data
            mean_weights = original_weights.mean(dim=1, keepdim=True)
            new_weights = mean_weights.repeat(1, input_channels, 1, 1)
            backbone.conv1[0].weight.data = new_weights

        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Linear(num_ftrs, num_classes)

        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)