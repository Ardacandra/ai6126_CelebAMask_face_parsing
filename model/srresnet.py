import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class SRResNetFaceParsing(nn.Module):
    def __init__(self, num_classes=19, num_residual_blocks=8):
        super(SRResNetFaceParsing, self).__init__()

        self.conv_first = nn.Conv2d(3, 64, 3, padding=1)

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residual_blocks)]
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.pred_head = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        x = self.conv_first(x)
        x = self.residual_blocks(x)
        x = self.decoder(x)
        x = self.pred_head(x)
        return x
