import torch
import torch.nn as nn
import torch.nn.functional as F

from .lite_face_parser import (
    ConvBNAct,
    DepthwiseSeparableConv,
    InvertedResidual,
    LiteASPP,
)


class ECABlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size for ECA must be odd")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.gate = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W] -> pooled: [B, 1, C] for lightweight channel mixing.
        pooled = self.pool(x).squeeze(-1).transpose(1, 2)
        weights = self.conv(pooled)
        weights = self.gate(weights).transpose(1, 2).unsqueeze(-1)
        return x * weights


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("kernel_size for spatial attention must be 3 or 7")
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.gate(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * attn


class AttentionFusionBlock(nn.Module):
    def __init__(self, lateral_channels, top_channels, out_channels, eca_kernel_size=3):
        super().__init__()
        self.lateral_proj = ConvBNAct(lateral_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.top_proj = ConvBNAct(top_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.fuse = DepthwiseSeparableConv(out_channels * 2, out_channels, kernel_size=3, stride=1)
        self.eca = ECABlock(out_channels, kernel_size=eca_kernel_size)

    def forward(self, lateral, top):
        top = F.interpolate(top, size=lateral.shape[-2:], mode="bilinear", align_corners=False)
        lateral = self.lateral_proj(lateral)
        top = self.top_proj(top)
        out = torch.cat([lateral, top], dim=1)
        out = self.fuse(out)
        out = self.eca(out)
        return out


class LiteFaceParserV5(nn.Module):
    def __init__(
        self,
        num_classes=19,
        stage_channels=(32, 48, 64, 96),
        expand_ratio=4,
        aspp_dilations=(1, 2, 4),
        aspp_channels=64,
        eca_kernel_size=3,
        spatial_attn_kernel=7,
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

        self.decode3 = AttentionFusionBlock(c3, aspp_channels, c3, eca_kernel_size=eca_kernel_size)
        self.decode2 = AttentionFusionBlock(c2, c3, c2, eca_kernel_size=eca_kernel_size)
        self.decode1 = AttentionFusionBlock(c1, c2, c1, eca_kernel_size=eca_kernel_size)

        self.refine = DepthwiseSeparableConv(c1, c1, kernel_size=3, stride=1)
        self.spatial_attn = SpatialAttention(kernel_size=spatial_attn_kernel)
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
        out = self.spatial_attn(out)
        out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        out = self.head(out)
        return out


if __name__ == "__main__":
    model = LiteFaceParserV5()
    params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {params:,}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {params:,}")