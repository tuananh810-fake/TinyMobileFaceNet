# -*- coding: utf-8 -*-
"""
Mô hình Tiny-MobileFaceNet-32 tối ưu cho cuộc thi LSI Contest 2026
Tác giả: Dương Văn Tuyên & Nguyễn Sỹ Tuấn Anh (VNU-UET)
Input: [B, 3, 32, 32] -> Output: [B, 128] (Face Embedding)
"""

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Khối Convolution chuẩn: Conv + PReLU"""
    def __init__(self, in_c, out_c, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)
        self.prelu = nn.PReLU(out_c)

    def forward(self, x):
        return self.prelu(self.conv(x))


class Bottleneck(nn.Module):
    """
    Khối Inverted Residual Bottleneck kế thừa từ MobileFaceNets:
    1. Pointwise Conv 1x1 (Mở rộng kênh với hệ số t)
    2. Depthwise Conv 3x3 (Trích xuất không gian)
    3. Pointwise Conv 1x1 (Nén kênh - Tuyến tính / Linear)
    """
    def __init__(self, in_c, out_c, stride, expansion_factor):
        super(Bottleneck, self).__init__()
        self.use_residual = (stride == 1 and in_c == out_c)
        hidden_dim = in_c * expansion_factor

        layers = []
        # 1. Expand (PW-Conv 1x1) nếu t > 1
        if expansion_factor != 1:
            layers.append(ConvBlock(in_c, hidden_dim, kernel_size=1, stride=1, padding=0))
        
        # 2. Depthwise Conv 3x3
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False))
        layers.append(nn.PReLU(hidden_dim))
        
        # 3. Linear Pointwise Conv 1x1 (Không dùng PReLU để bảo toàn đặc trưng)
        layers.append(nn.Conv2d(hidden_dim, out_c, kernel_size=1, stride=1, padding=0, bias=False))
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class TinyMobileFaceNet32(nn.Module):
    def __init__(self, embedding_size=128):
        super(TinyMobileFaceNet32, self).__init__()
        
        # Tầng Stem: 32x32x3 -> 16x16x32 (Thu nhỏ nhanh từ đầu)
        self.stem = ConvBlock(in_c=3, out_c=32, kernel_size=3, stride=2, padding=1)
        
        # Tầng Depthwise đầu tiên: 16x16x32 -> 16x16x32
        self.dw_stem = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32, bias=False),
            nn.PReLU(32),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=False)
        )
        
        # Các khối Bottlenecks (Hệ số t = 2, 4)
        # Giảm kích thước 16x16 -> 8x8 -> 4x4
        self.stage1 = Bottleneck(in_c=32, out_c=64, stride=2, expansion_factor=2)  # 16x16 -> 8x8x64
        self.stage2 = Bottleneck(in_c=64, out_c=64, stride=1, expansion_factor=2)  # 8x8x64
        
        self.stage3 = Bottleneck(in_c=64, out_c=128, stride=2, expansion_factor=4) # 8x8 -> 4x4x128
        self.stage4 = Bottleneck(in_c=128, out_c=128, stride=1, expansion_factor=2) # 4x4x128
        
        # Tầng Pointwise mở rộng kênh trước GDConv: 4x4x128 -> 4x4x256
        self.conv_sep = ConvBlock(in_c=128, out_c=256, kernel_size=1, stride=1, padding=0)
        
        # TẦNG PHÁT MINH QUAN TRỌNG: Global Depthwise Conv (GDConv 4x4)
        # Gom không gian 4x4 về 1x1 cho từng kênh độc lập (groups=256)
        self.gdconv = nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=0, groups=256, bias=False)
        
        # Tầng Linear cuối cùng: Nén từ 256 kênh về Vector 128 chiều
        self.linear_conv = nn.Conv2d(256, embedding_size, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.stem(x)          # [B, 32, 16, 16]
        x = self.dw_stem(x)       # [B, 32, 16, 16]
        x = self.stage1(x)        # [B, 64, 8, 8]
        x = self.stage2(x)        # [B, 64, 8, 8]
        x = self.stage3(x)        # [B, 128, 4, 4]
        x = self.stage4(x)        # [B, 128, 4, 4]
        x = self.conv_sep(x)      # [B, 256, 4, 4]
        x = self.gdconv(x)        # [B, 256, 1, 1] (GDConv 4x4)
        x = self.linear_conv(x)   # [B, 128, 1, 1]
        
        # Làm phẳng thành vector 128 chiều
        embedding = x.view(x.size(0), -1)  # [B, 128]
        
        # Chuẩn hóa L2-Norm (Mỗi vector có độ dài bằng 1 để so khớp Cosine)
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding


# KIỂM THỬ KÍCH THƯỚC VÀ ĐỘ TƯƠNG THÍCH
if __name__ == "__main__":
    model = TinyMobileFaceNet32(embedding_size=128)
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    
    # Đếm số lượng tham số
    total_params = sum(p.numel() for p in model.parameters())
    
    print("=" * 60)
    print("KIỂM TRA MÔ HÌNH TINY-MOBILEFACENET-32")
    print("=" * 60)
    print(f"Kích thước ảnh vào : {dummy_input.shape}")
    print(f"Kích thước vector ra: {output.shape} (Vector 128 chiều)")
    print(f"Tổng số tham số    : {total_params:,} parameters")
    print(f"Dung lượng bộ nhớ INT8 ước tính: {total_params / 1024:.2f} KB (Nhét vừa khít BRAM!)")
    print("=" * 60)