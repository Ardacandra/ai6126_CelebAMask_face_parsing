import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out + residual

class SRResNetBaseline(nn.Module):
    def __init__(
        self,
        num_classes=19,
        num_residual_blocks=16,
        context_dilations=(2, 4, 8),
        decoder_channels=(64, 48, 32),
    ):
        super(SRResNetBaseline, self).__init__()
        
        if len(decoder_channels) != 3:
            raise ValueError("decoder_channels must have exactly 3 values")

        # Initial Convolutional Layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        # 16 Residual Blocks
        res_blocks = []
        for _ in range(num_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Post-Residual Convolution
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        # Context module with dilated convolutions for larger receptive field
        context_layers = []
        for dilation in context_dilations:
            context_layers += [
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
                nn.BatchNorm2d(64),
                nn.PReLU(),
            ]
        self.context = nn.Sequential(*context_layers)

        # Segmentation decoder (keeps spatial size, refines local boundaries)
        self.decoder = nn.Sequential(
            nn.Conv2d(64, decoder_channels[0], kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(decoder_channels[0], decoder_channels[1], kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(decoder_channels[1], decoder_channels[2], kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )

        # Segmentation head (raw logits for CrossEntropyLoss)
        self.segmentation_head = nn.Conv2d(decoder_channels[2], num_classes, kernel_size=1)

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out = self.conv2(out)
        out = out + out1
        out = self.context(out)
        out = self.decoder(out)
        out = self.segmentation_head(out)
        return out

if __name__ == "__main__":
    model = SRResNetBaseline()
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")