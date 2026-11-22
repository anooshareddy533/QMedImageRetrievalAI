"""Multi-scale CNN / eight-head transformer with optional ResNet50 backbone."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv(in_ch, out_ch, stride=1):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
                         nn.BatchNorm2d(out_ch), nn.GELU())


class MultiScaleCNNTransformer(nn.Module):
    """Produces one 512-d image descriptor from a single-channel image."""
    def __init__(self, backbone="resnet50", pretrained=False, feature_dim=512,
                 transformer_layers=2, heads=8, tokens_per_scale=16, dropout=.3):
        super().__init__()
        self.backbone_name = backbone
        if backbone == "resnet50":
            from torchvision.models import resnet50, ResNet50_Weights
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            net = resnet50(weights=weights)
            original_conv = net.conv1
            net.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
            if pretrained:
                with torch.no_grad():
                    net.conv1.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))
            self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4
            channels = [512, 1024, 2048]
            model_dim = feature_dim
        elif backbone == "tiny":
            self.stem = nn.Sequential(_conv(1, 16, 2), _conv(16, 16))
            self.layer1 = nn.Identity()
            self.layer2 = nn.Sequential(_conv(16, 32, 2), _conv(32, 32))
            self.layer3 = nn.Sequential(_conv(32, 64, 2), _conv(64, 64))
            self.layer4 = nn.Sequential(_conv(64, 128, 2), _conv(128, 128))
            channels = [32, 64, 128]
            model_dim = 128  # efficient smoke path, still outputs full 512-d features
        else:
            raise ValueError("backbone should be resnet50 or tiny")
        side = math.isqrt(tokens_per_scale)
        if side * side != tokens_per_scale:
            raise ValueError("tokens_per_scale must be square (e.g., 4 or 16)")
        self.token_side = side
        self.projections = nn.ModuleList([nn.Linear(c, model_dim) for c in channels])
        self.scale_logits = nn.Parameter(torch.zeros(3))
        self.positional = nn.Parameter(torch.zeros(1, 3 * tokens_per_scale, model_dim))
        nn.init.trunc_normal_(self.positional, std=.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=heads, dim_feedforward=2 * model_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers,
                                                 enable_nested_tensor=False)
        self.output = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, feature_dim))

    def forward(self, images):
        x = self.stem(images)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        tokens = []
        weights = torch.softmax(self.scale_logits, dim=0)
        for i, fmap in enumerate((x2, x3, x4)):
            pooled = F.adaptive_avg_pool2d(fmap, (self.token_side, self.token_side))
            pooled = pooled.flatten(2).transpose(1, 2)
            tokens.append(weights[i] * self.projections[i](pooled))
        seq = torch.cat(tokens, dim=1) + self.positional
        seq = self.transformer(seq)
        return self.output(seq.mean(dim=1))
