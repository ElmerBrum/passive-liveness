"""
MultiFTNet — the TRAINING wrapper around MiniFASNet.

Ported from the original repo (src/model_lib/MultiFTNet.py) to Python 3.10+
and to reuse our inference MiniFASNet in liveness/model.py (single source of
truth for the architecture).

Two branches:
  - classification: MiniFASNet backbone → logits (num_classes)   [CrossEntropy]
  - Fourier aux   : FTGenerator on the conv_4 feature map → FFT   [MSE]

At training time forward() returns (cls_logits, ft_map).
At eval time it returns just cls_logits — identical to the inference model,
so a trained checkpoint can be loaded straight into liveness/predictor.py
(and exported to ONNX) after stripping the FTGenerator keys.

Input contract: x is FloatTensor [B, 3, 80, 80], BGR, pixels in [0, 255]
(NOT normalised to [0,1] — see docs/01_pixel-range-0-255.md).
"""

import sys
from pathlib import Path

import torch
from torch import nn

# Reuse the inference architecture (DRY).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from liveness.model import MODEL_REGISTRY  # noqa: E402


class FTGenerator(nn.Module):
    """Regresses the (downsampled) Fourier spectrum from the conv_4 feature map."""

    def __init__(self, in_channels: int = 128, out_channels: int = 1):
        super().__init__()
        self.ft = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.ft(x)


class MultiFTNet(nn.Module):
    def __init__(self, model_type: str = "MiniFASNetV2SE", num_classes: int = 2,
                 embedding_size: int = 128, conv6_kernel: tuple[int, int] = (5, 5),
                 img_channel: int = 3):
        super().__init__()
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model_type '{model_type}'. "
                             f"Options: {list(MODEL_REGISTRY)}")
        self.model_type = model_type
        self.num_classes = num_classes
        # The backbone is exactly our inference MiniFASNet variant.
        self.model = MODEL_REGISTRY[model_type](
            embedding_size=embedding_size,
            conv6_kernel=conv6_kernel,
            num_classes=num_classes,
            img_channel=img_channel,
        )
        # conv_4 outputs 128 channels for all keep-dict variants.
        self.FTGenerator = FTGenerator(in_channels=128)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        m = self.model
        x = m.conv1(x)
        x = m.conv2_dw(x)
        x = m.conv_23(x)
        x = m.conv_3(x)
        x = m.conv_34(x)
        x = m.conv_4(x)                 # feature map used by the Fourier branch
        x1 = m.conv_45(x)
        x1 = m.conv_5(x1)
        x1 = m.conv_6_sep(x1)
        x1 = m.conv_6_dw(x1)
        x1 = m.conv_6_flatten(x1)
        if m.embedding_size != 512:
            x1 = m.linear(x1)
        x1 = m.bn(x1)
        x1 = m.drop(x1)
        cls = m.prob(x1)

        if self.training:
            ft = self.FTGenerator(x)
            return cls, ft
        return cls

    def backbone_state_dict(self):
        """State dict of just the MiniFASNet backbone (for inference/ONNX)."""
        return self.model.state_dict()


def build_multiftnet(cfg) -> MultiFTNet:
    """Construct a MultiFTNet from a TrainConfig."""
    return MultiFTNet(
        model_type=cfg.model_type,
        num_classes=cfg.num_classes,
        embedding_size=cfg.embedding_size,
        conv6_kernel=cfg.kernel_size,
        img_channel=cfg.img_channel,
    )
