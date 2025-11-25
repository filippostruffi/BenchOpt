from typing import Any
from torch import nn
from torchvision import models
from benchmark.config.registry import register_model


@register_model("deeplabv3_resnet50")
def build_model(task: Any, dataset: Any) -> nn.Module:
    # Infer number of classes from mask, ignoring 255 if present
    _, mask = dataset[0]
    if mask.dtype.is_floating_point:
        mask = mask.long()
    valid = mask[mask != 255]
    if valid.numel() > 0:
        num_classes = int(valid.max().item()) + 1
    else:
        num_classes = int(mask.max().item()) + 1
    model = models.segmentation.deeplabv3_resnet50(weights=None, num_classes=num_classes)
    return model


