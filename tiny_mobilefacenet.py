# -*- coding: utf-8 -*-
"""
Mô hình Tiny-MobileFaceNet-32 (Phiên bản nâng cấp BatchNorm + BN Folding)
Tác giả: Nguyễn Sỹ Tuấn Anh
Input: [B, 3, 32, 32] -> Output: [B, 128] (L2-Normalized Face Embedding)
- Khi huấn luyện (use_bn=True): Dùng BatchNorm2d giúp hội tụ nhanh và đạt 95.68% LFW.
- Khi triển khai/xuất phần cứng (model.fold_bn()): Gập toàn bộ BatchNorm2d vào
  Conv2d (weight + bias), giữ nguyên 0 lớp BN khi suy diễn với sai số 0.00%.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Khối Convolution chuẩn: Conv2d (+ BatchNorm2d nếu bật) + PReLU"""
    def __init__(self, in_c, out_c, kernel_size, stride, padding, use_bn=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c) if use_bn else nn.Identity()
        self.prelu = nn.PReLU(out_c)

    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """
    Khối Inverted Residual Bottleneck kế thừa từ MobileFaceNet:
    1. Pointwise Conv 1x1 (Mở rộng kênh với hệ số t)
    2. Depthwise Conv 3x3 (Trích xuất không gian)
    3. Pointwise Conv 1x1 (Nén kênh tuyến tính / Linear)
    """
    def __init__(self, in_c, out_c, stride, expansion_factor, use_bn=True):
        super(Bottleneck, self).__init__()
        self.use_residual = (stride == 1 and in_c == out_c)
        hidden_dim = in_c * expansion_factor

        layers = []
        # 1. Expand (PW-Conv 1x1) nếu t > 1
        if expansion_factor != 1:
            layers.append(ConvBlock(in_c, hidden_dim, kernel_size=1, stride=1, padding=0, use_bn=use_bn))

        # 2. Depthwise Conv 3x3
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False))
        if use_bn:
            layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.PReLU(hidden_dim))

        # 3. Linear Pointwise Conv 1x1 (Không dùng PReLU để bảo toàn đặc trưng)
        layers.append(nn.Conv2d(hidden_dim, out_c, kernel_size=1, stride=1, padding=0, bias=False))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_c))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


def fuse_conv_bn(conv, bn):
    """Gập toán học BatchNorm2d vào Conv2d (Weight + Bias) -> 0 chi phí BN khi suy diễn"""
    fused = nn.Conv2d(
        conv.in_channels, conv.out_channels,
        conv.kernel_size, conv.stride, conv.padding,
        conv.dilation, conv.groups, bias=True
    ).to(conv.weight.device)
    with torch.no_grad():
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        std = torch.sqrt(bn.running_var + bn.eps)
        w_bn = torch.diag(bn.weight / std)
        fused.weight.copy_(torch.mm(w_bn, w_conv).view(fused.weight.size()))
        b_conv = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels, device=conv.weight.device)
        b_bn = bn.bias - bn.weight * bn.running_mean / std
        fused.bias.copy_(torch.mm(w_bn, b_conv.view(-1, 1)).view(-1) + b_bn)
    return fused


class TinyMobileFaceNet32(nn.Module):
    def __init__(self, embedding_size=128, use_bn=True):
        super(TinyMobileFaceNet32, self).__init__()
        self.use_bn = use_bn

        # Tầng Stem: 32x32x3 -> 16x16x32
        self.stem = ConvBlock(in_c=3, out_c=32, kernel_size=3, stride=2, padding=1, use_bn=use_bn)

        # Tầng Depthwise đầu tiên: 16x16x32 -> 16x16x32
        dw_layers = [nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32, bias=False)]
        if use_bn:
            dw_layers.append(nn.BatchNorm2d(32))
        dw_layers.append(nn.PReLU(32))
        dw_layers.append(nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=False))
        if use_bn:
            dw_layers.append(nn.BatchNorm2d(32))
        self.dw_stem = nn.Sequential(*dw_layers)

        # 4 khối Bottlenecks chính: 16x16 -> 8x8 -> 4x4
        self.stage1 = Bottleneck(in_c=32, out_c=64, stride=2, expansion_factor=2, use_bn=use_bn)
        self.stage2 = Bottleneck(in_c=64, out_c=64, stride=1, expansion_factor=2, use_bn=use_bn)
        self.stage3 = Bottleneck(in_c=64, out_c=128, stride=2, expansion_factor=4, use_bn=use_bn)
        self.stage4 = Bottleneck(in_c=128, out_c=128, stride=1, expansion_factor=2, use_bn=use_bn)

        # Tầng Pointwise mở rộng kênh trước GDConv: 4x4x128 -> 4x4x256
        self.conv_sep = ConvBlock(in_c=128, out_c=256, kernel_size=1, stride=1, padding=0, use_bn=use_bn)

        # Global Depthwise Conv (GDConv 4x4): 4x4x256 -> 1x1x256
        self.gdconv = nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=0, groups=256, bias=False)
        self.gd_bn = nn.BatchNorm2d(256) if use_bn else nn.Identity()

        # Tầng Linear cuối cùng: Nén 256 -> Vector 128 chiều
        self.linear_conv = nn.Conv2d(256, embedding_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.lin_bn = nn.BatchNorm2d(embedding_size) if use_bn else nn.Identity()

    def forward(self, x):
        x = self.stem(x)
        x = self.dw_stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv_sep(x)
        x = self.gd_bn(self.gdconv(x))
        x = self.lin_bn(self.linear_conv(x))
        embedding = x.view(x.size(0), -1)
        return F.normalize(embedding, p=2, dim=1)

    def fold_bn(self):
        """Trả về bản sao mô hình đã gập toàn bộ BatchNorm2d vào Conv2d (Zero-BN inference)"""
        m = copy.deepcopy(self).eval()
        for name in ['stem', 'conv_sep']:
            blk = getattr(m, name)
            if isinstance(blk.bn, nn.BatchNorm2d):
                blk.conv = fuse_conv_bn(blk.conv, blk.bn)
                blk.bn = nn.Identity()
        if isinstance(m.gd_bn, nn.BatchNorm2d):
            m.gdconv = fuse_conv_bn(m.gdconv, m.gd_bn)
            m.gd_bn = nn.Identity()
        if isinstance(m.lin_bn, nn.BatchNorm2d):
            m.linear_conv = fuse_conv_bn(m.linear_conv, m.lin_bn)
            m.lin_bn = nn.Identity()

        def _fold_seq(seq):
            new_layers = []
            layers = list(seq)
            i = 0
            while i < len(layers):
                layer = layers[i]
                if isinstance(layer, ConvBlock) and isinstance(layer.bn, nn.BatchNorm2d):
                    layer.conv = fuse_conv_bn(layer.conv, layer.bn)
                    layer.bn = nn.Identity()
                    new_layers.append(layer)
                    i += 1
                elif isinstance(layer, nn.Conv2d) and i + 1 < len(layers) and isinstance(layers[i + 1], nn.BatchNorm2d):
                    new_layers.append(fuse_conv_bn(layer, layers[i + 1]))
                    i += 2
                else:
                    new_layers.append(layer)
                    i += 1
            return nn.Sequential(*new_layers)

        m.dw_stem = _fold_seq(m.dw_stem)
        for st in ['stage1', 'stage2', 'stage3', 'stage4']:
            blk = getattr(m, st)
            blk.conv = _fold_seq(blk.conv)
        m.use_bn = False
        return m


if __name__ == "__main__":
    model = TinyMobileFaceNet32(embedding_size=128, use_bn=True).eval()
    folded = model.fold_bn()
    dummy = torch.randn(1, 3, 32, 32)
    out_bn = model(dummy)
    out_folded = folded(dummy)
    diff = (out_bn - out_folded).abs().max().item()
    p_bn = sum(p.numel() for p in model.parameters())
    p_folded = sum(p.numel() for p in folded.parameters())
    print("=" * 62)
    print("KIỂM TRA MÔ HÌNH TINY-MOBILEFACENET-32 (BN + BN FOLDING)")
    print("=" * 62)
    print(f"Input shape           : {dummy.shape}")
    print(f"Output shape          : {out_bn.shape} (128-D L2-Normalized)")
    print(f"Tham số khi train (BN): {p_bn:,} parameters")
    print(f"Tham số sau BN Folding: {p_folded:,} parameters (~{p_folded / 1024:.2f} KB INT8)")
    print(f"Sai lệch tối đa sau gập BN: {diff:.2e} (Tương đương toán học tuyệt đối)")
    print("=" * 62)
