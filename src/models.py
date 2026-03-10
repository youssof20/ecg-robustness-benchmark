"""
Phase 2 — ECG arrhythmia classifier architectures.

All models: input (batch, 1, 280), output (batch, 5) logits for AAMI classes N, S, V, F, Q.
"""

import torch
import torch.nn as nn

NUM_CLASSES = 5
INPUT_LEN = 280


def _count_parameters(module: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class SimpleCNN(nn.Module):
    """
    3 convolutional blocks (Conv1d → BatchNorm → ReLU → MaxPool), then
    global average pooling and FC 128 → 64 → 5.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        # Block 1: 1 → 32
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        # Block 2: 32 → 64
        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        # Block 3: 64 → 128
        self.conv3 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = x.mean(dim=2)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class ResBlock1D(nn.Module):
    """Residual block: two Conv1d with skip connection."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, padding: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.skip(x))


class ResNet1D(nn.Module):
    """
    4 residual blocks (64 → 128 → 256 → 256), global average pooling, FC 256 → 5.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResBlock1D(64, 64),
            ResBlock1D(64, 128),
            ResBlock1D(128, 256),
            ResBlock1D(256, 256),
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = x.mean(dim=2)
        return self.fc(x)


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise conv followed by pointwise conv."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, padding: int = 3):
        super().__init__()
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch)
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return torch.relu(self.bn(x))


class LightweightNet(nn.Module):
    """
    Depthwise-separable 1D CNN (MobileNet-style). 4 blocks 16→32→64→64,
    global average pooling, FC 64→5. Must have < 50,000 parameters.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.block1 = DepthwiseSeparableConv1d(in_channels, 16)
        self.block2 = DepthwiseSeparableConv1d(16, 32)
        self.block3 = DepthwiseSeparableConv1d(32, 64)
        self.block4 = DepthwiseSeparableConv1d(64, 64)
        self.fc = nn.Linear(64, num_classes)
        n = _count_parameters(self)
        print(f"LightweightNet parameter count: {n} (required < 50,000)")
        if n >= 50_000:
            raise ValueError(f"LightweightNet has {n} params, must have < 50,000")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.mean(dim=2)
        return self.fc(x)


def get_model(name: str, **kwargs) -> nn.Module:
    """Return model by name: 'SimpleCNN', 'ResNet1D', 'LightweightNet'."""
    models = {
        "SimpleCNN": SimpleCNN,
        "ResNet1D": ResNet1D,
        "LightweightNet": LightweightNet,
    }
    if name not in models:
        raise ValueError(f"Unknown model {name}; choose from {list(models)}")
    return models[name](**kwargs)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = torch.randn(4, 1, INPUT_LEN, device=device)
    for name in ["SimpleCNN", "ResNet1D", "LightweightNet"]:
        model = get_model(name).to(device)
        out = model(batch)
        assert out.shape == (4, NUM_CLASSES), f"{name}: {out.shape}"
        n = _count_parameters(model)
        print(f"{name}: output {out.shape}, params={n}")
