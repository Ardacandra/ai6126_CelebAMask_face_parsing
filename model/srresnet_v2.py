import torch.nn as nn

from .srresnet import ResidualBlock


class SRResNetFaceParsingV2(nn.Module):
    def __init__(
        self,
        num_classes=19,
        num_residual_blocks=10,
        channels=96,
        decoder_channels=(48, 24),
    ):
        super(SRResNetFaceParsingV2, self).__init__()

        if len(decoder_channels) != 2:
            raise ValueError("decoder_channels must have exactly 2 values")

        self.conv_first = nn.Conv2d(3, channels, 3, padding=1)

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_residual_blocks)]
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(channels, decoder_channels[0], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[0], decoder_channels[1], 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.pred_head = nn.Conv2d(decoder_channels[1], num_classes, 1)

    def forward(self, x):
        x = self.conv_first(x)
        x = self.residual_blocks(x)
        x = self.decoder(x)
        x = self.pred_head(x)
        return x
