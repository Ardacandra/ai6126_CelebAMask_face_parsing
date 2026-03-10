import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        groups=1,
        dilation=1,
        norm_type="gn",
        gn_groups=8,
        activation=True,
    ):
        super().__init__()
        if padding is None:
            padding = (kernel_size // 2) * dilation

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

        if norm_type == "bn":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == "gn":
            groups = min(gn_groups, out_channels)
            while out_channels % groups != 0 and groups > 1:
                groups -= 1
            self.norm = nn.GroupNorm(groups, out_channels)
        else:
            raise ValueError("norm_type must be 'bn' or 'gn'")

        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        norm_type="gn",
        gn_groups=8,
        activation=True,
    ):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.dw = ConvNormAct(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            dilation=dilation,
            norm_type=norm_type,
            gn_groups=gn_groups,
            activation=True,
        )
        self.pw = ConvNormAct(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_type=norm_type,
            gn_groups=gn_groups,
            activation=activation,
        )

    def forward(self, x):
        return self.pw(self.dw(x))


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=4, norm_type="gn", gn_groups=8):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")
        hidden = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels

        self.expand = ConvNormAct(
            in_channels,
            hidden,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_type=norm_type,
            gn_groups=gn_groups,
        )
        self.depthwise = ConvNormAct(
            hidden,
            hidden,
            kernel_size=3,
            stride=stride,
            groups=hidden,
            norm_type=norm_type,
            gn_groups=gn_groups,
        )
        self.project = ConvNormAct(
            hidden,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_type=norm_type,
            gn_groups=gn_groups,
            activation=False,
        )

    def forward(self, x):
        out = self.project(self.depthwise(self.expand(x)))
        if self.use_residual:
            out = out + x
        return out


class AntiAliasDownsample(nn.Module):
    def __init__(self, channels, out_channels, norm_type="gn", gn_groups=8):
        super().__init__()
        self.blur = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = ConvNormAct(
            channels,
            out_channels,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            gn_groups=gn_groups,
        )

    def forward(self, x):
        return self.conv(self.blur(x))


class LiteASPPv2(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=(1, 2, 4, 8), norm_type="gn", gn_groups=8):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                ConvNormAct(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    norm_type=norm_type,
                    gn_groups=gn_groups,
                )
            ]
        )
        for d in dilations:
            self.branches.append(
                DepthwiseSeparableConv(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    dilation=d,
                    norm_type=norm_type,
                    gn_groups=gn_groups,
                )
            )

        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvNormAct(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_type=norm_type,
                gn_groups=gn_groups,
            ),
        )

        total = out_channels * (len(self.branches) + 1)
        self.project = ConvNormAct(
            total,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_type=norm_type,
            gn_groups=gn_groups,
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        feats = [branch(x) for branch in self.branches]
        pooled = self.image_pool(x)
        pooled = F.interpolate(pooled, size=(h, w), mode="bilinear", align_corners=False)
        feats.append(pooled)
        return self.project(torch.cat(feats, dim=1))


class WeightedBiFusion(nn.Module):
    def __init__(self, lateral_channels, top_channels, out_channels, norm_type="gn", gn_groups=8):
        super().__init__()
        self.lateral_proj = ConvNormAct(
            lateral_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_type=norm_type,
            gn_groups=gn_groups,
        )
        self.top_proj = ConvNormAct(
            top_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_type=norm_type,
            gn_groups=gn_groups,
        )
        self.weights = nn.Parameter(torch.ones(2))
        self.fuse = DepthwiseSeparableConv(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            gn_groups=gn_groups,
        )

    def forward(self, lateral, top):
        top = F.interpolate(top, size=lateral.shape[-2:], mode="bilinear", align_corners=False)
        lat = self.lateral_proj(lateral)
        top = self.top_proj(top)

        w = F.relu(self.weights)
        w = w / (w.sum() + 1e-6)
        out = w[0] * lat + w[1] * top
        return self.fuse(out)


class TexturePath(nn.Module):
    def __init__(self, out_channels=24, norm_type="gn", gn_groups=8):
        super().__init__()
        self.proj = ConvNormAct(3, out_channels, kernel_size=3, stride=2, norm_type=norm_type, gn_groups=gn_groups)
        self.refine = DepthwiseSeparableConv(
            out_channels,
            out_channels,
            kernel_size=5,
            stride=2,
            norm_type=norm_type,
            gn_groups=gn_groups,
        )

    def forward(self, x):
        # High-frequency emphasis helps separate patterned shirts and subtle boundaries.
        blurred = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high_freq = x - blurred
        return self.refine(self.proj(high_freq))


class EdgeHead(nn.Module):
    def __init__(self, in_channels, norm_type="gn", gn_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableConv(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                norm_type=norm_type,
                gn_groups=gn_groups,
            ),
            nn.Conv2d(in_channels, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


class PrototypeRefinement(nn.Module):
    def __init__(self, channels, num_classes, proj_dim=32):
        super().__init__()
        self.proj_q = nn.Conv2d(channels, proj_dim, kernel_size=1)
        self.coarse_head = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.key = nn.Linear(channels, proj_dim)
        self.value = nn.Linear(channels, channels)
        self.out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, feats):
        b, c, h, w = feats.shape
        n = h * w

        coarse_logits = self.coarse_head(feats)
        probs = torch.softmax(coarse_logits, dim=1)

        feats_flat = feats.view(b, c, n)
        probs_flat = probs.view(b, probs.shape[1], n)

        denom = probs_flat.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        prototypes = torch.einsum("bkn,bcn->bkc", probs_flat, feats_flat) / denom

        q = self.proj_q(feats).view(b, -1, n).transpose(1, 2)
        k = self.key(prototypes)
        v = self.value(prototypes)

        attn = torch.softmax(torch.matmul(q, k.transpose(1, 2)) / math.sqrt(k.shape[-1]), dim=-1)
        ctx = torch.matmul(attn, v)
        ctx = ctx.transpose(1, 2).view(b, c, h, w)

        return feats + self.out(ctx), coarse_logits


class LiteFaceParserV2(nn.Module):
    def __init__(
        self,
        num_classes=19,
        stage_channels=(44, 76, 120, 176),
        stage_repeats=(3, 3, 4),
        detail_channels=44,
        texture_channels=32,
        expand_ratio=4,
        aspp_dilations=(1, 2, 4, 8),
        aspp_channels=140,
        norm_type="gn",
        gn_groups=8,
    ):
        super().__init__()
        if len(stage_channels) != 4:
            raise ValueError("stage_channels must have exactly 4 values")
        if len(stage_repeats) != 3:
            raise ValueError("stage_repeats must have exactly 3 values for stage2-stage4")
        if min(stage_repeats) < 1:
            raise ValueError("All stage_repeats values must be >= 1")

        c1, c2, c3, c4 = stage_channels
        r2, r3, r4 = stage_repeats

        # Context encoder
        self.stem = ConvNormAct(3, c1, kernel_size=3, stride=2, norm_type=norm_type, gn_groups=gn_groups)
        self.aa_down = AntiAliasDownsample(c1, c1, norm_type=norm_type, gn_groups=gn_groups)

        stage2_blocks = [
            InvertedResidual(c1, c2, stride=1, expand_ratio=expand_ratio, norm_type=norm_type, gn_groups=gn_groups)
        ]
        stage2_blocks.extend(
            InvertedResidual(c2, c2, stride=1, expand_ratio=expand_ratio, norm_type=norm_type, gn_groups=gn_groups)
            for _ in range(r2 - 1)
        )
        self.stage2 = nn.Sequential(*stage2_blocks)

        stage3_blocks = [
            InvertedResidual(c2, c3, stride=2, expand_ratio=expand_ratio, norm_type=norm_type, gn_groups=gn_groups)
        ]
        stage3_blocks.extend(
            InvertedResidual(c3, c3, stride=1, expand_ratio=expand_ratio, norm_type=norm_type, gn_groups=gn_groups)
            for _ in range(r3 - 1)
        )
        self.stage3 = nn.Sequential(*stage3_blocks)

        stage4_blocks = [
            InvertedResidual(c3, c4, stride=2, expand_ratio=expand_ratio, norm_type=norm_type, gn_groups=gn_groups)
        ]
        stage4_blocks.extend(
            InvertedResidual(c4, c4, stride=1, expand_ratio=expand_ratio, norm_type=norm_type, gn_groups=gn_groups)
            for _ in range(r4 - 1)
        )
        self.stage4 = nn.Sequential(*stage4_blocks)

        self.context = LiteASPPv2(
            c4,
            aspp_channels,
            dilations=aspp_dilations,
            norm_type=norm_type,
            gn_groups=gn_groups,
        )

        # Detail and texture paths at 1/4 scale
        self.detail_path = nn.Sequential(
            ConvNormAct(3, detail_channels // 2, kernel_size=3, stride=2, norm_type=norm_type, gn_groups=gn_groups),
            DepthwiseSeparableConv(
                detail_channels // 2,
                detail_channels,
                kernel_size=3,
                stride=2,
                norm_type=norm_type,
                gn_groups=gn_groups,
            ),
            DepthwiseSeparableConv(
                detail_channels,
                detail_channels,
                kernel_size=3,
                stride=1,
                norm_type=norm_type,
                gn_groups=gn_groups,
            ),
        )
        self.texture_path = TexturePath(texture_channels, norm_type=norm_type, gn_groups=gn_groups)

        # Bidirectional fusion decoder
        self.td3 = WeightedBiFusion(c3, aspp_channels, c3, norm_type=norm_type, gn_groups=gn_groups)
        self.td2 = WeightedBiFusion(c2, c3, c2, norm_type=norm_type, gn_groups=gn_groups)
        self.bu3 = WeightedBiFusion(c3, c2, c3, norm_type=norm_type, gn_groups=gn_groups)
        self.final_1_4 = WeightedBiFusion(c2, c3, c2, norm_type=norm_type, gn_groups=gn_groups)

        # Class-prototype context refinement
        self.prototype_refine = PrototypeRefinement(c2, num_classes)

        # Edge-guided fusion
        self.edge_head = EdgeHead(detail_channels, norm_type=norm_type, gn_groups=gn_groups)
        fused_channels = c2 + detail_channels + texture_channels
        self.fusion_proj = DepthwiseSeparableConv(
            fused_channels,
            c2,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            gn_groups=gn_groups,
        )

        self.seg_head = nn.Conv2d(c2, num_classes, kernel_size=1)

    def forward(self, x):
        # Context path
        x1 = self.stem(x)             # 1/2
        x2 = self.aa_down(x1)         # 1/4
        x2 = self.stage2(x2)          # 1/4
        x3 = self.stage3(x2)          # 1/8
        x4 = self.stage4(x3)          # 1/16
        ctx = self.context(x4)        # 1/16

        # Bidirectional decoder
        td3 = self.td3(x3, ctx)       # 1/8
        td2 = self.td2(x2, td3)       # 1/4
        bu3 = self.bu3(td3, td2)      # 1/8 (top is resized)
        dec = self.final_1_4(td2, bu3)

        # Prototype-guided refinement for minority classes
        dec, aux_coarse_logits = self.prototype_refine(dec)

        # Detail and texture cues
        detail = self.detail_path(x)
        texture = self.texture_path(x)

        edge_logits = self.edge_head(detail)
        edge_attn = torch.sigmoid(edge_logits)

        dec = dec * (1.0 + edge_attn)
        fused = torch.cat([dec, detail, texture], dim=1)
        fused = self.fusion_proj(fused)

        logits = self.seg_head(fused)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

        aux = {
            "edge_logits": F.interpolate(edge_logits, size=x.shape[-2:], mode="bilinear", align_corners=False),
            "coarse_logits": F.interpolate(aux_coarse_logits, size=x.shape[-2:], mode="bilinear", align_corners=False),
        }
        return logits, aux


if __name__ == "__main__":
    model = LiteFaceParserV2()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {params:,}")
