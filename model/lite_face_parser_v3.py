import torch
import torch.nn as nn
import torch.nn.functional as F

from .lite_face_parser import (
    ConvBNAct,
    DepthwiseSeparableConv,
    FusionBlock,
    InvertedResidual,
    LiteASPP,
)


class BoundaryRefinementBlock(nn.Module):
    def __init__(self, feat_channels, refine_channels):
        super().__init__()
        self.refine = nn.Sequential(
            DepthwiseSeparableConv(
                feat_channels + 1,
                refine_channels,
                kernel_size=3,
                stride=1,
            ),
            DepthwiseSeparableConv(
                refine_channels,
                feat_channels,
                kernel_size=3,
                stride=1,
            ),
        )

    def forward(self, features, boundary_logits, gate_strength=1.0):
        boundary_prob = torch.sigmoid(boundary_logits)
        gated = features * (1.0 + gate_strength * boundary_prob)
        return self.refine(torch.cat([gated, boundary_prob], dim=1))


class LiteFaceParserV3(nn.Module):
    def __init__(
        self,
        num_classes=19,
        stage_channels=(32, 48, 64, 96),
        expand_ratio=4,
        aspp_dilations=(1, 2, 4),
        aspp_channels=64,
        boundary_refine_channels=48,
        boundary_gate_strength=1.0,
    ):
        super().__init__()
        if len(stage_channels) != 4:
            raise ValueError("stage_channels must have exactly 4 values")

        c1, c2, c3, c4 = stage_channels
        self.boundary_gate_strength = float(boundary_gate_strength)

        # Backbone and decoder follow LiteFaceParser (v1) directly.
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

        # Dedicated boundary head predicts edges from decoder features and refines full-res features.
        self.boundary_head = nn.Sequential(
            DepthwiseSeparableConv(
                c1,
                c1,
                kernel_size=3,
                stride=1,
            ),
            nn.Conv2d(c1, 1, kernel_size=1),
        )
        self.boundary_refine = BoundaryRefinementBlock(
            feat_channels=c1,
            refine_channels=boundary_refine_channels,
        )

        self.seg_head = nn.Conv2d(c1, num_classes, kernel_size=1)

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
        fused = self.refine(d1)

        # Boundary-aware refinement at full resolution to reduce class bleeding across facial edges.
        boundary_logits_1_4 = self.boundary_head(fused)
        full_res_feat = F.interpolate(fused, size=x.shape[-2:], mode="bilinear", align_corners=False)
        boundary_logits = F.interpolate(boundary_logits_1_4, size=x.shape[-2:], mode="bilinear", align_corners=False)
        refined_full_res = self.boundary_refine(
            full_res_feat,
            boundary_logits,
            gate_strength=self.boundary_gate_strength,
        )

        logits = self.seg_head(refined_full_res)

        aux = {
            "boundary_logits": boundary_logits,
        }
        return logits, aux


if __name__ == "__main__":
    model = LiteFaceParserV3()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {params:,}")