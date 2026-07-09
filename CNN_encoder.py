import torch
import torch.nn as nn


# ==========================================================
# Residual Block
# ==========================================================

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:

            self.shortcut = nn.Sequential(

                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),

                nn.BatchNorm2d(out_channels)

            )

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity

        out = self.relu(out)

        return out


# ==========================================================
# CNN Encoder
# ==========================================================

class CNNEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        # --------------------------------------------------
        # Initial Convolution
        # --------------------------------------------------

        self.stem = nn.Sequential(

            nn.Conv2d(
                2,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2)

        )

        # Output:
        # (32,64,64)

        # --------------------------------------------------
        # Residual Stages
        # --------------------------------------------------

        self.layer1 = ResidualBlock(
            32,
            64,
            stride=2
        )

        # (64,32,32)

        self.layer2 = ResidualBlock(
            64,
            128,
            stride=2
        )

        # (128,16,16)

        self.layer3 = ResidualBlock(
            128,
            256,
            stride=2
        )

        # (256,8,8)

        # --------------------------------------------------
        # Global Average Pool
        # --------------------------------------------------

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        # --------------------------------------------------
        # Feature Compression
        # --------------------------------------------------

        self.fc = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256,
                128
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(inplace=True),

            nn.Dropout(0.30),

            nn.Linear(
                128,
                64
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(inplace=True)

        )

    def forward(self, x):

        x = self.stem(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.pool(x)

        x = self.fc(x)

        return x
