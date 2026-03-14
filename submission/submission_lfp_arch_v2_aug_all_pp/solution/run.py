import argparse
import math
import cv2

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


PALETTE = np.array([[i, i, i] for i in range(256)], dtype=np.uint8)
PALETTE[:19] = np.array(
    [
        [0, 0, 0],
        [204, 0, 0],
        [76, 153, 0],
        [204, 204, 0],
        [51, 51, 255],
        [204, 0, 204],
        [0, 255, 255],
        [255, 204, 204],
        [102, 51, 0],
        [255, 0, 0],
        [102, 204, 0],
        [255, 255, 0],
        [0, 0, 153],
        [0, 0, 204],
        [255, 51, 153],
        [0, 204, 204],
        [0, 51, 0],
        [255, 153, 51],
        [0, 204, 0],
    ],
    dtype=np.uint8,
)


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
        c1, c2, c3, c4 = stage_channels
        r2, r3, r4 = stage_repeats

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

        self.td3 = WeightedBiFusion(c3, aspp_channels, c3, norm_type=norm_type, gn_groups=gn_groups)
        self.td2 = WeightedBiFusion(c2, c3, c2, norm_type=norm_type, gn_groups=gn_groups)
        self.bu3 = WeightedBiFusion(c3, c2, c3, norm_type=norm_type, gn_groups=gn_groups)
        self.final_1_4 = WeightedBiFusion(c2, c3, c2, norm_type=norm_type, gn_groups=gn_groups)

        self.prototype_refine = PrototypeRefinement(c2, num_classes)

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
        x1 = self.stem(x)
        x2 = self.aa_down(x1)
        x2 = self.stage2(x2)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        ctx = self.context(x4)

        td3 = self.td3(x3, ctx)
        td2 = self.td2(x2, td3)
        bu3 = self.bu3(td3, td2)
        dec = self.final_1_4(td2, bu3)

        dec, aux_coarse_logits = self.prototype_refine(dec)

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


def extract_logits(model_outputs):
    if isinstance(model_outputs, torch.Tensor):
        return model_outputs
    if isinstance(model_outputs, (tuple, list)) and len(model_outputs) > 0:
        return model_outputs[0]
    if isinstance(model_outputs, dict):
        if "logits" in model_outputs:
            return model_outputs["logits"]
        if "out" in model_outputs:
            return model_outputs["out"]
    raise TypeError("Unsupported model output format")


def get_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


# ---------------------------------------------------------------------------
# Postprocessing methods (combined_all: SCR -> majority_filter_3x3)
# ---------------------------------------------------------------------------

def small_component_removal(
    mask: np.ndarray,
    num_classes: int = 19,
    min_size_default: int = 32,
    min_size_by_class: dict = None,
) -> np.ndarray:
    if min_size_by_class is None:
        min_size_by_class = {}
    result = mask.copy()
    for class_id in range(1, num_classes):
        class_mask = (result == class_id).astype(np.uint8)
        if class_mask.sum() == 0:
            continue
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
        threshold = int(min_size_by_class.get(str(class_id), min_size_default))
        for component_id in range(1, n_labels):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area >= threshold:
                continue
            component_mask = labels == component_id
            ys, xs = np.where(component_mask)
            if ys.size == 0:
                continue
            y0 = max(0, ys.min() - 1)
            y1 = min(result.shape[0], ys.max() + 2)
            x0 = max(0, xs.min() - 1)
            x1 = min(result.shape[1], xs.max() + 2)
            neighborhood = result[y0:y1, x0:x1]
            neighborhood_component = component_mask[y0:y1, x0:x1]
            neighbors = neighborhood[~neighborhood_component]
            replacement = int(np.bincount(neighbors).argmax()) if neighbors.size > 0 else 0
            result[component_mask] = replacement
    return result


def majority_filter_3x3(mask: np.ndarray, num_classes: int = 19) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.float32)
    scores = np.zeros((num_classes, mask.shape[0], mask.shape[1]), dtype=np.float32)
    for class_id in range(num_classes):
        class_binary = (mask == class_id).astype(np.float32)
        scores[class_id] = cv2.filter2D(class_binary, -1, kernel, borderType=cv2.BORDER_REFLECT)
    return np.argmax(scores, axis=0).astype(np.uint8)


def postprocess(mask: np.ndarray, num_classes: int = 19) -> np.ndarray:
    # combined_all (dense_crf disabled): SCR then majority filter
    mask = small_component_removal(mask, num_classes=num_classes)
    mask = majority_filter_3x3(mask, num_classes=num_classes)
    return mask


def main(input, output, weights):
    # Load the input image
    img = cv2.imread(input)
    if img is None:
        raise FileNotFoundError(f"Failed to load input image: {input}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: Initialize the neural network model
    # Example:
    # from models import YourSegModel
    # model = YourSegModel()
    model = LiteFaceParserV2(
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
    ).to(device)

    # Load the checkpoint
    ckpt = torch.load(weights, map_location=device)
    # NOTE: Make sure that the weights are saved in the "state_dict" key
    # DO NOT CHANGE THIS VALUE, i.e., ckpt["state_dict"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Inference with the model (Update as needed)
    # Normalize the image.
    # NOTE: Make sure it is aligned with the training data
    # Example: img = (img / 255.0 - 0.5) * 2.0
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    tensor = get_transform(512)(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(tensor)

    # Convert PyTorch Tensor to numpy array
    logits = extract_logits(prediction)
    mask = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

    # Apply postprocessing (combined_all: small_component_removal + majority_filter_3x3)
    mask = postprocess(mask)

    # Save the prediction
    mask_img = Image.fromarray(mask, mode="P")
    mask_img.putpalette(PALETTE.reshape(-1).tolist())
    if mask_img.size != (512, 512):
        mask_img = mask_img.resize((512, 512), resample=Image.NEAREST)
    mask_img.save(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--weights", type=str, default="ckpt.pth")
    args = parser.parse_args()
    main(args.input, args.output, args.weights)
