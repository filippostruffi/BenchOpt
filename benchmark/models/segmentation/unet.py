from typing import Any
import torch
from torch import nn
from benchmark.config.registry import register_model


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(64, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        self.out = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        xb = self.bottleneck(self.pool2(x2))
        x = self.up2(xb)
        x = self.dec2(torch.cat([x, x2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))
        return self.out(x)


@register_model("unet_vanilla")
def build_model(task: Any, dataset: Any) -> nn.Module:
    x, y = dataset[0]
    in_ch = x.shape[0]
    num_classes = int(torch.max(y).item()) + 1
    return UNet(in_channels=in_ch, num_classes=num_classes)


