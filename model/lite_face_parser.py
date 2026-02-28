import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        groups=1,
        dilation=1,
        activation=True,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, activation=True):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.dw = ConvBNAct(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            dilation=dilation,
            activation=True,
        )
        self.pw = ConvBNAct(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            activation=activation,
        )

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return x


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        scale = self.pool(x)
        scale = self.fc1(scale)
        scale = self.act(scale)
        scale = self.fc2(scale)
        scale = self.gate(scale)
        return x * scale


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=4, use_se=True):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")

        hidden_channels = int(in_channels * expand_ratio)
        self.use_residual = stride == 1 and in_channels == out_channels

        self.expand = ConvBNAct(
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=True,
        )
        self.depthwise = ConvBNAct(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=hidden_channels,
            activation=True,
        )
        self.se = SqueezeExcitation(hidden_channels) if use_se else nn.Identity()
        self.project = ConvBNAct(
            hidden_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=False,
        )

    def forward(self, x):
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.se(out)
        out = self.project(out)
        if self.use_residual:
            out = out + x
        return out


class LiteASPP(nn.Module):
    def __init__(self, in_channels, branch_channels=64, dilations=(1, 2, 4)):
        super().__init__()
        self.branches = nn.ModuleList()
        self.branches.append(ConvBNAct(in_channels, branch_channels, kernel_size=1, stride=1, padding=0))
        for dilation in dilations:
            self.branches.append(
                DepthwiseSeparableConv(
                    in_channels,
                    branch_channels,
                    kernel_size=3,
                    stride=1,
                    dilation=dilation,
                )
            )

        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNAct(in_channels, branch_channels, kernel_size=1, stride=1, padding=0),
        )

        total_channels = branch_channels * (len(self.branches) + 1)
        self.project = ConvBNAct(total_channels, branch_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h, w = x.shape[-2:]
        features = [branch(x) for branch in self.branches]

        pooled = self.image_pool(x)
        pooled = F.interpolate(pooled, size=(h, w), mode="bilinear", align_corners=False)
        features.append(pooled)

        out = torch.cat(features, dim=1)
        out = self.project(out)
        return out


class FusionBlock(nn.Module):
    def __init__(self, lateral_channels, top_channels, out_channels):
        super().__init__()
        self.lateral_proj = ConvBNAct(lateral_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.top_proj = ConvBNAct(top_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.fuse = DepthwiseSeparableConv(out_channels * 2, out_channels, kernel_size=3, stride=1)

    def forward(self, lateral, top):
        top = F.interpolate(top, size=lateral.shape[-2:], mode="bilinear", align_corners=False)
        lateral = self.lateral_proj(lateral)
        top = self.top_proj(top)
        out = torch.cat([lateral, top], dim=1)
        out = self.fuse(out)
        return out


class LiteFaceParser(nn.Module):
    def __init__(
        self,
        num_classes=19,
        stage_channels=(32, 48, 64, 96),
        expand_ratio=4,
        aspp_dilations=(1, 2, 4),
        aspp_channels=64,
    ):
        super().__init__()

        if len(stage_channels) != 4:
            raise ValueError("stage_channels must have exactly 4 values")

        c1, c2, c3, c4 = stage_channels

        self.stem = ConvBNAct(3, c1, kernel_size=3, stride=2)

        self.stage1 = nn.Sequential(
            InvertedResidual(c1, c1, stride=1, expand_ratio=2, use_se=False),
            InvertedResidual(c1, c1, stride=1, expand_ratio=2, use_se=False),
        )
        self.stage2 = nn.Sequential(
            InvertedResidual(c1, c2, stride=2, expand_ratio=expand_ratio, use_se=True),
            InvertedResidual(c2, c2, stride=1, expand_ratio=expand_ratio, use_se=True),
        )
        self.stage3 = nn.Sequential(
            InvertedResidual(c2, c3, stride=2, expand_ratio=expand_ratio, use_se=True),
            InvertedResidual(c3, c3, stride=1, expand_ratio=expand_ratio, use_se=True),
        )
        self.stage4 = nn.Sequential(
            InvertedResidual(c3, c4, stride=2, expand_ratio=expand_ratio, use_se=True),
            InvertedResidual(c4, c4, stride=1, expand_ratio=expand_ratio, use_se=True),
            InvertedResidual(c4, c4, stride=1, expand_ratio=expand_ratio, use_se=True),
        )

        self.context = LiteASPP(c4, branch_channels=aspp_channels, dilations=aspp_dilations)

        self.decode3 = FusionBlock(c3, aspp_channels, c3)
        self.decode2 = FusionBlock(c2, c3, c2)
        self.decode1 = FusionBlock(c1, c2, c1)

        self.refine = DepthwiseSeparableConv(c1, c1, kernel_size=3, stride=1)
        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.stage1(x1)

        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        context = self.context(x4)

        d3 = self.decode3(x3, context)
        d2 = self.decode2(x2, d3)
        d1 = self.decode1(x1, d2)

        out = self.refine(d1)
        out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        out = self.head(out)
        return out


if __name__ == "__main__":
    model = LiteFaceParser()
    params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {params:,}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {params:,}")
