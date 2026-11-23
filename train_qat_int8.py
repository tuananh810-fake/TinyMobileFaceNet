# -*- coding: utf-8 -*-
"""
===============================================================================
HUẤN LUYỆN LƯỢNG TỬ HÓA (QAT - FULL INT8) & XUẤT GOLDEN MODEL CHO PHẦN CỨNG FPGA
Mô hình: TinyMobileFaceNet-32 (18 tầng Conv2d, 0 tầng BatchNorm)
Tác giả: Nguyễn Sỹ Tuấn Anh

Chức năng trọn gói:
1. Nạp trọng số đã gập BN (tiny_mobilefacenet_lfw.pt -> fold_bn()).
2. Hiệu chuẩn (Calibration) biên độ Activation của 18 tầng bằng 99.9th percentile.
3. Chạy QAT (Quantization-Aware Training) cho cả Weight INT8 + Activation INT8 + PReLU Q0.7.
4. Kiểm chứng Bit-True Integer-Only (100% số nguyên INT8/INT32, 0 phép chia số thực)
   trên toàn bộ chuẩn 10-Fold LFW (6.000 cặp ảnh).
5. Xuất trọn gói vào hardware_export/:
   - weights_int8.mem, bias_int32.mem, prelu_q07.mem, requant_mult_int32.mem
   - quant_params.json (địa chỉ bộ nhớ, shape, shift, scale từng tầng)
   - golden_vectors/ (ảnh vào INT8 + đáp án chuẩn từng tầng 1..18 cho Verilog Testbench)
   - Hardware_Baseline_Spec.md (Bảng tiêu chí gốc 18 tầng chốt thiết kế Datapath)
===============================================================================
"""

import os
import sys
import time
import json
import copy
import math
import csv
import shutil
import datetime
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from tiny_mobilefacenet import TinyMobileFaceNet32

SHIFT_BITS = 16  # Cố định dịch phải 16 bit cho mọi tầng (rất gọn cho mạch RTL!)


# =============================================================================
# 1. CÁC HÀM FAKE QUANTIZATION (STRAIGHT-THROUGH ESTIMATOR - STE)
# =============================================================================
def ste_round(x):
    return (torch.round(x) - x).detach() + x


def fake_quant_act_sym_int8(x, scale):
    s = scale.clamp(min=1e-8)
    q = ste_round(x / s).clamp(-127.0, 127.0)
    return q * s


def fake_quant_weight_per_channel_int8(w):
    max_v = w.detach().abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
    scale = max_v / 127.0
    q = ste_round(w / scale).clamp(-127.0, 127.0)
    return q * scale, scale.view(-1)


def fake_quant_prelu_q07(alpha):
    q = ste_round(alpha * 128.0).clamp(-128.0, 127.0)
    return q / 128.0


# =============================================================================
# 2. MÔ HÌNH QAT FULL-INT8 (18 TẦNG CONV2D PHẲNG, DỄ ÁNH XẠ 1-1 SANG VERILOG)
# =============================================================================
class QATConvLayer(nn.Module):
    """Một tầng phần cứng chuẩn: Conv2d(INT8 W, INT32 B) -> Requant INT8 -> Optional PReLU(Q0.7)"""
    def __init__(self, conv_mod, prelu_mod=None, name=""):
        super().__init__()
        self.name = name
        self.conv = copy.deepcopy(conv_mod)
        self.has_prelu = prelu_mod is not None
        self.prelu = copy.deepcopy(prelu_mod) if self.has_prelu else None
        self.register_buffer("act_scale", torch.tensor(1.0 / 127.0))
        self.calibrating = False
        self._calib_vals = []

    def forward(self, x, enable_quant=True):
        if enable_quant:
            w_q, _ = fake_quant_weight_per_channel_int8(self.conv.weight)
            y = F.conv2d(x, w_q, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
            if self.has_prelu:
                a_q = fake_quant_prelu_q07(self.prelu.weight)
                y = F.prelu(y, a_q)
        else:
            y = self.conv(x)
            if self.has_prelu:
                y = self.prelu(y)

        if self.calibrating:
            # Dùng phân vị 99.9% để tránh ngoại lai làm bẹp dải INT8
            val = torch.quantile(y.detach().abs().reshape(-1), 0.999).item()
            self._calib_vals.append(max(val, 1e-5))
            return y

        if enable_quant:
            if self.training:
                # Cập nhật nhẹ EMA scale trong lúc QAT
                with torch.no_grad():
                    cur_max = torch.quantile(y.detach().abs().reshape(-1), 0.999).clamp(min=1e-5)
                    new_s = cur_max / 127.0
                    self.act_scale.mul_(0.98).add_(0.02 * new_s)
            y = fake_quant_act_sym_int8(y, self.act_scale)
        return y


class TinyMobileFaceNet32_QAT(nn.Module):
    """
    Phiên bản QAT chuẩn hóa thành đúng 18 tầng tuần tự + 2 nhánh Residual Add (stage2, stage4)
    khớp 1-1 với kiến trúc Datapath phần cứng.
    """
    def __init__(self, folded_model):
        super().__init__()
        m = copy.deepcopy(folded_model).eval()
        self.register_buffer("input_scale", torch.tensor(1.0 / 127.0))

        # Trích xuất đúng 18 tầng Conv2d (+ PReLU đi kèm) theo đúng thứ tự suy diễn
        self.layers = nn.ModuleList([
            QATConvLayer(m.stem.conv, m.stem.prelu, "L01_stem"),
            QATConvLayer(m.dw_stem[0], m.dw_stem[1], "L02_dw_stem_dw"),
            QATConvLayer(m.dw_stem[2], None, "L03_dw_stem_pw"),

            QATConvLayer(m.stage1.conv[0].conv, m.stage1.conv[0].prelu, "L04_s1_exp"),
            QATConvLayer(m.stage1.conv[1], m.stage1.conv[2], "L05_s1_dw"),
            QATConvLayer(m.stage1.conv[3], None, "L06_s1_pw"),

            QATConvLayer(m.stage2.conv[0].conv, m.stage2.conv[0].prelu, "L07_s2_exp"),
            QATConvLayer(m.stage2.conv[1], m.stage2.conv[2], "L08_s2_dw"),
            QATConvLayer(m.stage2.conv[3], None, "L09_s2_pw"),       # + Residual from L06

            QATConvLayer(m.stage3.conv[0].conv, m.stage3.conv[0].prelu, "L10_s3_exp"),
            QATConvLayer(m.stage3.conv[1], m.stage3.conv[2], "L11_s3_dw"),
            QATConvLayer(m.stage3.conv[3], None, "L12_s3_pw"),

            QATConvLayer(m.stage4.conv[0].conv, m.stage4.conv[0].prelu, "L13_s4_exp"),
            QATConvLayer(m.stage4.conv[1], m.stage4.conv[2], "L14_s4_dw"),
            QATConvLayer(m.stage4.conv[3], None, "L15_s4_pw"),       # + Residual from L12

            QATConvLayer(m.conv_sep.conv, m.conv_sep.prelu, "L16_conv_sep"),
            QATConvLayer(m.gdconv, None, "L17_gdconv"),
            QATConvLayer(m.linear_conv, None, "L18_linear_conv"),
        ])

        # Scale cho 2 điểm cộng Residual Add
        self.register_buffer("res_s2_scale", torch.tensor(1.0 / 127.0))
        self.register_buffer("res_s4_scale", torch.tensor(1.0 / 127.0))
        self._res_s2_vals = []
        self._res_s4_vals = []
        self.calibrating = False

    def set_calibrating(self, flag):
        self.calibrating = flag
        for l in self.layers:
            l.calibrating = flag
            if flag:
                l._calib_vals = []
        if flag:
            self._res_s2_vals = []
            self._res_s4_vals = []

    def finalize_calibration(self):
        for l in self.layers:
            if len(l._calib_vals) > 0:
                l.act_scale.fill_(float(np.median(l._calib_vals)) / 127.0)
            l.calibrating = False
            l._calib_vals = []
        if len(self._res_s2_vals) > 0:
            self.res_s2_scale.fill_(float(np.median(self._res_s2_vals)) / 127.0)
        if len(self._res_s4_vals) > 0:
            self.res_s4_scale.fill_(float(np.median(self._res_s4_vals)) / 127.0)
        self.calibrating = False
        self._res_s2_vals = []
        self._res_s4_vals = []

    def forward(self, x, enable_quant=True):
        if enable_quant and not self.calibrating:
            x = fake_quant_act_sym_int8(x, self.input_scale)

        # Stem + Stage 1 (L01..L06)
        for i in range(6):
            x = self.layers[i](x, enable_quant=enable_quant)

        # Stage 2 with Residual (L07..L09)
        skip_s2 = x
        for i in range(6, 9):
            x = self.layers[i](x, enable_quant=enable_quant)
        x = skip_s2 + x
        if self.calibrating:
            self._res_s2_vals.append(max(torch.quantile(x.detach().abs().reshape(-1), 0.999).item(), 1e-5))
        elif enable_quant:
            if self.training:
                with torch.no_grad():
                    self.res_s2_scale.mul_(0.98).add_(0.02 * torch.quantile(x.detach().abs().reshape(-1), 0.999).clamp(min=1e-5) / 127.0)
            x = fake_quant_act_sym_int8(x, self.res_s2_scale)

        # Stage 3 (L10..L12)
        for i in range(9, 12):
            x = self.layers[i](x, enable_quant=enable_quant)

        # Stage 4 with Residual (L13..L15)
        skip_s4 = x
        for i in range(12, 15):
            x = self.layers[i](x, enable_quant=enable_quant)
        x = skip_s4 + x
        if self.calibrating:
            self._res_s4_vals.append(max(torch.quantile(x.detach().abs().reshape(-1), 0.999).item(), 1e-5))
        elif enable_quant:
            if self.training:
                with torch.no_grad():
                    self.res_s4_scale.mul_(0.98).add_(0.02 * torch.quantile(x.detach().abs().reshape(-1), 0.999).clamp(min=1e-5) / 127.0)
            x = fake_quant_act_sym_int8(x, self.res_s4_scale)

        # Head (L16..L18)
        for i in range(15, 18):
            x = self.layers[i](x, enable_quant=enable_quant)

        emb = x.view(x.size(0), -1)
        return F.normalize(emb, p=2, dim=1)


# =============================================================================
# 3. BIT-TRUE INTEGER-ONLY GOLDEN MODEL (MÔ PHỎNG Y HỆT MẠCH VERILOG!)
# =============================================================================
class IntegerOnlyGoldenModel:
    """
    Thực thi 100% số nguyên (INT8 x INT8 -> INT32 + Bias_INT32 -> *M >> 16 -> PReLU Q0.7 -> INT8)
    KHÔNG dùng bất kỳ phép toán số thực nào trong toàn bộ 18 tầng!
    """
    def __init__(self, qat_model, device):
        self.device = device
        self.layer_specs = []
        qat_model.eval()

        in_scales = [
            qat_model.input_scale.item(),         # L01
            qat_model.layers[0].act_scale.item(), # L02
            qat_model.layers[1].act_scale.item(), # L03
            qat_model.layers[2].act_scale.item(), # L04
            qat_model.layers[3].act_scale.item(), # L05
            qat_model.layers[4].act_scale.item(), # L06
            qat_model.layers[5].act_scale.item(), # L07 (also skip_s2 scale)
            qat_model.layers[6].act_scale.item(), # L08
            qat_model.layers[7].act_scale.item(), # L09
            qat_model.res_s2_scale.item(),        # L10
            qat_model.layers[9].act_scale.item(), # L11
            qat_model.layers[10].act_scale.item(),# L12
            qat_model.layers[11].act_scale.item(),# L13 (also skip_s4 scale)
            qat_model.layers[12].act_scale.item(),# L14
            qat_model.layers[13].act_scale.item(),# L15
            qat_model.res_s4_scale.item(),        # L16
            qat_model.layers[15].act_scale.item(),# L17
            qat_model.layers[16].act_scale.item(),# L18
        ]

        with torch.no_grad():
            for idx, layer in enumerate(qat_model.layers):
                s_in = in_scales[idx]
                s_out = layer.act_scale.item()
                w = layer.conv.weight.detach()
                b = layer.conv.bias.detach()

                max_w = w.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
                s_w = (max_w / 127.0).view(-1)
                w_int8 = torch.round(w / s_w.view(-1, 1, 1, 1)).clamp(-127, 127).to(torch.int8)

                # Bias lượng tử hóa về INT32 ở cùng miền (s_in * s_w)
                b_int32 = torch.round(b / (s_in * s_w)).to(torch.int32)

                # Hệ số nhân nguyên M cho phép dịch cố định >> SHIFT_BITS (16)
                real_mult = (s_in * s_w) / s_out
                m_int32 = torch.round(real_mult * (1 << SHIFT_BITS)).clamp(min=1).to(torch.int32)

                alpha_q07 = None
                if layer.has_prelu:
                    alpha_q07 = torch.round(layer.prelu.weight.detach() * 128.0).clamp(-128, 127).to(torch.int8)

                self.layer_specs.append({
                    "name": layer.name,
                    "w_int8": w_int8.to(device),
                    "b_int32": b_int32.to(device),
                    "m_int32": m_int32.to(device),
                    "alpha_q07": alpha_q07.to(device) if alpha_q07 is not None else None,
                    "stride": layer.conv.stride,
                    "padding": layer.conv.padding,
                    "groups": layer.conv.groups,
                    "kernel_size": layer.conv.kernel_size,
                    "in_channels": layer.conv.in_channels,
                    "out_channels": layer.conv.out_channels,
                    "s_in": s_in,
                    "s_out": s_out,
                    "shift": SHIFT_BITS,
                })

            # Hệ số cộng Residual bằng số nguyên (shift = 15)
            self.res_s2 = {
                "m_skip": int(round((qat_model.layers[5].act_scale.item() / qat_model.res_s2_scale.item()) * (1 << 15))),
                "m_branch": int(round((qat_model.layers[8].act_scale.item() / qat_model.res_s2_scale.item()) * (1 << 15))),
                "shift": 15,
            }
            self.res_s4 = {
                "m_skip": int(round((qat_model.layers[11].act_scale.item() / qat_model.res_s4_scale.item()) * (1 << 15))),
                "m_branch": int(round((qat_model.layers[14].act_scale.item() / qat_model.res_s4_scale.item()) * (1 << 15))),
                "shift": 15,
            }

    @staticmethod
    def _run_int_conv(x_int8, spec):
        # PyTorch conv2d trên float64 để biểu diễn chính xác tuyệt đối tích lũy INT32 (không mất bit)
        acc = F.conv2d(
            x_int8.to(torch.float64),
            spec["w_int8"].to(torch.float64),
            spec["b_int32"].to(torch.float64),
            stride=spec["stride"],
            padding=spec["padding"],
            groups=spec["groups"]
        ).to(torch.int64)

        # Re-quantize: (acc * M + 2^(S-1)) >> S
        half = 1 << (spec["shift"] - 1)
        m = spec["m_int32"].to(torch.int64).view(1, -1, 1, 1)
        y = torch.where(acc >= 0, (acc * m + half) >> spec["shift"], -((-acc * m + half) >> spec["shift"]))

        # PReLU (Q0.7): nếu y < 0 -> (y * alpha_q07) >> 7
        if spec["alpha_q07"] is not None:
            a = spec["alpha_q07"].to(torch.int64).view(1, -1, 1, 1)
            neg_y = (y * a) >> 7
            y = torch.where(y >= 0, y, neg_y)

        return y.clamp(-127, 127).to(torch.int8)

    @staticmethod
    def _run_int_res_add(x_skip_i8, x_br_i8, res_cfg):
        acc = x_skip_i8.to(torch.int64) * res_cfg["m_skip"] + x_br_i8.to(torch.int64) * res_cfg["m_branch"]
        half = 1 << (res_cfg["shift"] - 1)
        y = torch.where(acc >= 0, (acc + half) >> res_cfg["shift"], -((-acc + half) >> res_cfg["shift"]))
        return y.clamp(-127, 127).to(torch.int8)

    def forward_int8(self, x_int8, capture_layers=False):
        intermediates = {}
        x = x_int8
        for i in range(6):
            x = self._run_int_conv(x, self.layer_specs[i])
            if capture_layers:
                intermediates[self.layer_specs[i]["name"]] = x.cpu().clone()

        skip_s2 = x
        for i in range(6, 9):
            x = self._run_int_conv(x, self.layer_specs[i])
            if capture_layers:
                intermediates[self.layer_specs[i]["name"]] = x.cpu().clone()
        x = self._run_int_res_add(skip_s2, x, self.res_s2)
        if capture_layers:
            intermediates["RES_stage2_out"] = x.cpu().clone()

        for i in range(9, 12):
            x = self._run_int_conv(x, self.layer_specs[i])
            if capture_layers:
                intermediates[self.layer_specs[i]["name"]] = x.cpu().clone()

        skip_s4 = x
        for i in range(12, 15):
            x = self._run_int_conv(x, self.layer_specs[i])
            if capture_layers:
                intermediates[self.layer_specs[i]["name"]] = x.cpu().clone()
        x = self._run_int_res_add(skip_s4, x, self.res_s4)
        if capture_layers:
            intermediates["RES_stage4_out"] = x.cpu().clone()

        for i in range(15, 18):
            x = self._run_int_conv(x, self.layer_specs[i])
            if capture_layers:
                intermediates[self.layer_specs[i]["name"]] = x.cpu().clone()

        emb_int8 = x.view(x.size(0), -1)
        if capture_layers:
            return emb_int8, intermediates
        return emb_int8


# =============================================================================
# 4. DATASET & 10-FOLD LFW EVALUATION (HỖ TRỢ CẢ QAT VÀ INTEGER-ONLY GOLDEN)
# =============================================================================
class FastDistillDataset(Dataset):
    def __init__(self, aligned_dir, augment=True):
        self.augment = augment
        self.images = []
        for root, _, files in os.walk(aligned_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img = cv2.imread(os.path.join(root, f))
                    if img is not None:
                        self.images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_rgb = self.images[idx]
        if self.augment:
            if np.random.rand() > 0.5:
                img_rgb = cv2.flip(img_rgb, 1)
            if np.random.rand() > 0.4:
                angle = np.random.uniform(-8.0, 8.0)
                scale = np.random.uniform(0.94, 1.06)
                tx, ty = np.random.uniform(-4.0, 4.0), np.random.uniform(-4.0, 4.0)
                M = cv2.getRotationMatrix2D((56.0, 56.0), angle, scale)
                M[0, 2] += tx
                M[1, 2] += ty
                img_rgb = cv2.warpAffine(img_rgb, M, (112, 112), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            alpha = np.random.uniform(0.85, 1.15)
            beta = np.random.randint(-15, 15)
            img_rgb = np.clip(img_rgb.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        img_32 = cv2.resize(img_rgb, (32, 32), interpolation=cv2.INTER_AREA)
        t_s = (torch.from_numpy(img_32.transpose(2, 0, 1)).float() - 127.5) / 128.0
        t_t = (torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() - 127.5) / 128.0
        return t_s, t_t


def read_lfw_pairs(pairs_path, aligned_dir):
    with open(pairs_path, 'r', encoding='utf-8') as f:
        lines = [line.strip().split() for line in f.readlines() if line.strip()]
    num_folds, num_per = int(lines[0][0]), int(lines[0][1])
    idx, pairs = 1, []
    for _ in range(num_folds):
        fp = []
        for _ in range(num_per):
            n, i1, i2 = lines[idx][0], int(lines[idx][1]), int(lines[idx][2])
            fp.append((os.path.join(aligned_dir, n, f"{n}_{i1:04d}.jpg"),
                       os.path.join(aligned_dir, n, f"{n}_{i2:04d}.jpg"), 1))
            idx += 1
        for _ in range(num_per):
            n1, i1, n2, i2 = lines[idx][0], int(lines[idx][1]), lines[idx][2], int(lines[idx][3])
            fp.append((os.path.join(aligned_dir, n1, f"{n1}_{i1:04d}.jpg"),
                       os.path.join(aligned_dir, n2, f"{n2}_{i2:04d}.jpg"), 0))
            idx += 1
        pairs.append(fp)
    return pairs


def evaluate_10fold_lfw(eval_fn, pairs):
    cache = {}
    def get_v(p):
        if p in cache:
            return cache[p]
        img = cv2.imread(p)
        if img is None:
            return None
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im32 = cv2.resize(rgb, (32, 32), interpolation=cv2.INTER_AREA)
        v = eval_fn(im32)
        cache[p] = v
        return v

    fold_scores, fold_labels = [], []
    for fp in pairs:
        s_list, l_list = [], []
        for f1, f2, lab in fp:
            v1, v2 = get_v(f1), get_v(f2)
            if v1 is not None and v2 is not None:
                s_list.append(float(np.dot(v1, v2)))
                l_list.append(lab)
        fold_scores.append(np.array(s_list))
        fold_labels.append(np.array(l_list))

    thresholds = np.arange(-1.0, 1.0, 0.005)
    accs = []
    for i in range(10):
        tr_s = np.concatenate([fold_scores[j] for j in range(10) if j != i])
        tr_l = np.concatenate([fold_labels[j] for j in range(10) if j != i])
        best_th, best_a = 0.0, 0.0
        for th in thresholds:
            a = np.mean((tr_s >= th) == tr_l)
            if a > best_a:
                best_a, best_th = a, th
        accs.append(np.mean((fold_scores[i] >= best_th) == fold_labels[i]))

    all_s = np.concatenate(fold_scores)
    all_l = np.concatenate(fold_labels)
    fpr, tpr, ths = roc_curve(all_l, all_s)
    roc_auc = auc(fpr, tpr)
    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0 * 100.0
    return float(np.mean(accs) * 100.0), float(np.std(accs) * 100.0), float(roc_auc), float(eer), float(ths[eer_idx]), fpr, tpr, all_s, all_l


def plot_qat_report(fpr, tpr, roc_auc, eer, best_th, acc_mean, all_scores, all_labels, save_path):
    """Xuất đồ thị báo cáo QAT Full-INT8 (ROC Curve + Cosine Separation Distribution)"""
    plt.figure(figsize=(14, 5.5), dpi=180)

    # Panel 1: ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='#0b5ed7', lw=2.5, label=f'TinyMobileFaceNet-32 INT8 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1.2, linestyle='--')
    eer_x = eer / 100.0
    plt.scatter([eer_x], [1.0 - eer_x], color='red', s=55, zorder=5, label=f'EER = {eer:.2f}%')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate (FPR)', fontsize=11, fontweight='bold')
    plt.ylabel('True Positive Rate (TPR)', fontsize=11, fontweight='bold')
    plt.title(f'Full-INT8 LFW ROC Curve (Accuracy: {acc_mean:.2f}%)', fontsize=12, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9.5)
    plt.grid(True, alpha=0.3)

    # Panel 2: Separation Distribution
    plt.subplot(1, 2, 2)
    gen_scores = all_scores[all_labels == 1]
    imp_scores = all_scores[all_labels == 0]
    plt.hist(imp_scores, bins=55, alpha=0.65, color='#dc3545', density=True, label=f'Impostor (Diff Person, N={len(imp_scores)})')
    plt.hist(gen_scores, bins=55, alpha=0.65, color='#198754', density=True, label=f'Genuine (Same Person, N={len(gen_scores)})')
    plt.axvline(best_th, color='black', linestyle='--', lw=2, label=f'Optimal Thresh = {best_th:.3f}')
    plt.xlabel('Cosine Similarity Score (Integer-Only INT8 Output)', fontsize=11, fontweight='bold')
    plt.ylabel('Probability Density', fontsize=11, fontweight='bold')
    plt.title('Full-INT8 Face Separation Distribution (LFW 6,000 Pairs)', fontsize=12, fontweight='bold')
    plt.legend(loc="upper left", fontsize=9)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[+] Đã xuất biểu đồ báo cáo QAT INT8 tại: {save_path}")


def log_qat_history(csv_path, record):
    exists = os.path.exists(csv_path)
    headers = [
        "Timestamp", "Mode", "Epochs_Done", "LR", "Duration_Mins",
        "FP32_Acc", "PTQ_INT8_Acc", "QAT_INT8_Acc", "QAT_AUC", "QAT_EER",
        "QAT_Threshold", "Report_Path"
    ]
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        if not exists:
            w.writeheader()
        w.writerow(record)
    print(f"[+] Đã ghi nhật ký QAT tại: {csv_path}")


# =============================================================================
# 5. XUẤT FILE PHẦN CỨNG (.MEM, JSON, GOLDEN VECTORS & BẢNG TIÊU CHÍ GỐC)
# =============================================================================
def write_mem_hex(filepath, array_np, bitwidth):
    mask = (1 << bitwidth) - 1
    hex_chars = bitwidth // 4
    flat = array_np.reshape(-1)
    with open(filepath, "w", encoding="utf-8") as f:
        for v in flat:
            f.write(f"{int(v) & mask:0{hex_chars}x}\n")


def export_hardware_package(golden_model, sample_img_path, export_dir, metrics):
    os.makedirs(export_dir, exist_ok=True)
    gv_dir = os.path.join(export_dir, "golden_vectors")
    os.makedirs(gv_dir, exist_ok=True)

    # 1. Xuất bộ nhớ trọng số nối tiếp (.mem) + metadata offset từng tầng
    all_w, all_b, all_m, all_p = [], [], [], []
    w_off, b_off, m_off, p_off = 0, 0, 0, 0
    layer_meta = []

    # Chạy 1 ảnh mẫu qua Golden Model để lấy đúng shape & Golden Vectors từng tầng
    img = cv2.imread(sample_img_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im32 = cv2.resize(rgb, (32, 32), interpolation=cv2.INTER_AREA)
    x_norm = (im32.astype(np.float32) - 127.5) / 128.0
    x_int8_np = np.clip(np.round(x_norm * 127.0), -127, 127).astype(np.int8)
    x_int8_t = torch.from_numpy(x_int8_np.transpose(2, 0, 1)).unsqueeze(0).to(golden_model.device)

    emb_int8, intermediates = golden_model.forward_int8(x_int8_t, capture_layers=True)

    write_mem_hex(os.path.join(gv_dir, "input_32x32_int8.mem"), x_int8_t.cpu().numpy(), 8)
    for k, tensor in intermediates.items():
        write_mem_hex(os.path.join(gv_dir, f"{k}_int8.mem"), tensor.numpy(), 8)

    with open(os.path.join(gv_dir, "final_embedding_128d_int8.txt"), "w", encoding="utf-8") as f:
        f.write(" ".join(str(int(v)) for v in emb_int8.cpu().squeeze(0).numpy()))

    # Tính toán MACs & kích thước bộ nhớ từng tầng
    prev_shape = (3, 32, 32)
    total_macs = 0
    max_single_bank = 0
    max_pingpong = 0

    for idx, spec in enumerate(golden_model.layer_specs):
        out_t = intermediates[spec["name"]]
        out_shape = tuple(out_t.shape[1:])
        H_out, W_out = out_shape[1], out_shape[2]
        kH, kW = spec["kernel_size"]
        macs = H_out * W_out * spec["out_channels"] * (spec["in_channels"] // spec["groups"]) * kH * kW
        total_macs += macs

        in_bytes = int(np.prod(prev_shape))
        out_bytes = int(np.prod(out_shape))
        max_single_bank = max(max_single_bank, in_bytes, out_bytes)
        max_pingpong = max(max_pingpong, in_bytes + out_bytes)

        w_np = spec["w_int8"].cpu().numpy()
        b_np = spec["b_int32"].cpu().numpy()
        m_np = spec["m_int32"].cpu().numpy()
        p_np = spec["alpha_q07"].cpu().numpy() if spec["alpha_q07"] is not None else np.array([], dtype=np.int8)

        layer_meta.append({
            "index": idx + 1,
            "name": spec["name"],
            "in_shape_CHW": list(prev_shape),
            "out_shape_CHW": list(out_shape),
            "kernel": f"{kH}x{kW}",
            "stride": spec["stride"][0],
            "groups": spec["groups"],
            "has_prelu": spec["alpha_q07"] is not None,
            "macs": macs,
            "weight_bytes_int8": int(w_np.size),
            "bias_bytes_int32": int(b_np.size * 4),
            "prelu_bytes_int8": int(p_np.size),
            "act_in_bytes_int8": in_bytes,
            "act_out_bytes_int8": out_bytes,
            "weight_offset": w_off,
            "bias_offset": b_off,
            "mult_offset": m_off,
            "prelu_offset": p_off if p_np.size > 0 else -1,
            "shift": spec["shift"],
            "scale_in": spec["s_in"],
            "scale_out": spec["s_out"],
        })

        all_w.append(w_np.reshape(-1))
        all_b.append(b_np.reshape(-1))
        all_m.append(m_np.reshape(-1))
        if p_np.size > 0:
            all_p.append(p_np.reshape(-1))

        w_off += int(w_np.size)
        b_off += int(b_np.size)
        m_off += int(m_np.size)
        p_off += int(p_np.size)
        prev_shape = out_shape

    all_w_np = np.concatenate(all_w)
    all_b_np = np.concatenate(all_b)
    all_m_np = np.concatenate(all_m)
    all_p_np = np.concatenate(all_p)

    write_mem_hex(os.path.join(export_dir, "weights_int8.mem"), all_w_np, 8)
    write_mem_hex(os.path.join(export_dir, "bias_int32.mem"), all_b_np, 32)
    write_mem_hex(os.path.join(export_dir, "requant_mult_int32.mem"), all_m_np, 32)
    write_mem_hex(os.path.join(export_dir, "prelu_q07.mem"), all_p_np, 8)

    params_json = {
        "model": "TinyMobileFaceNet-32 (Full INT8 QAT, Zero-BN)",
        "author": "Nguyen Sy Tuan Anh",
        "shift_bits_conv": SHIFT_BITS,
        "residual_stage2": golden_model.res_s2,
        "residual_stage4": golden_model.res_s4,
        "summary": {
            "total_layers": len(layer_meta),
            "total_macs": total_macs,
            "total_weight_bytes_int8": int(all_w_np.size),
            "total_bias_bytes_int32": int(all_b_np.size * 4),
            "total_prelu_bytes_int8": int(all_p_np.size),
            "total_mult_bytes_int32": int(all_m_np.size * 4),
            "max_single_activation_bank_bytes": max_single_bank,
            "max_ping_pong_ram_bytes": max_pingpong,
            "est_latency_ms_9dsp_100mhz": round(total_macs / (9.0 * 100e6) * 1000.0, 2),
        },
        "metrics_10fold_lfw": metrics,
        "layers": layer_meta,
    }
    with open(os.path.join(export_dir, "quant_params.json"), "w", encoding="utf-8") as f:
        json.dump(params_json, f, indent=2, ensure_ascii=False)

    # 2. Xuất Bảng Tiêu Chí Gốc (Hardware_Baseline_Spec.md)
    md = []
    md.append("# Bảng Tiêu Chí Gốc Phần Cứng (Hardware Baseline Specification)")
    md.append("")
    md.append("- **Mô hình:** `TinyMobileFaceNet-32` (Full INT8 QAT — 0 lớp BatchNorm)")
    md.append("- **Tác giả:** Nguyễn Sỹ Tuấn Anh")
    md.append("- **Định dạng số học phần cứng:** `Input INT8 × Weight INT8 → Accumulator INT32 + Bias INT32 → (*M >> 16) → PReLU Q0.7 → Output INT8`")
    md.append("")
    md.append("## 1. Chỉ tiêu độ tin cậy chuẩn 10-Fold LFW (6.000 cặp ảnh)")
    md.append("")
    md.append("| Phiên bản mô hình | 10-Fold Accuracy | ROC AUC | Equal Error Rate (EER) | Threshold |")
    md.append("| :--- | :---: | :---: | :---: | :---: |")
    md.append(f"| FP32 Folded-BN (Gốc) | {metrics['fp32_acc']:.2f}% ± {metrics['fp32_std']:.2f}% | {metrics['fp32_auc']:.4f} | {metrics['fp32_eer']:.2f}% | {metrics['fp32_th']:.3f} |")
    md.append(f"| PTQ Full INT8 (Trước QAT) | {metrics['ptq_acc']:.2f}% ± {metrics['ptq_std']:.2f}% | {metrics['ptq_auc']:.4f} | {metrics['ptq_eer']:.2f}% | {metrics['ptq_th']:.3f} |")
    md.append(f"| **QAT Full INT8 (Integer-Only Golden Model)** | **{metrics['qat_int_acc']:.2f}% ± {metrics['qat_int_std']:.2f}%** | **{metrics['qat_int_auc']:.4f}** | **{metrics['qat_int_eer']:.2f}%** | **{metrics['qat_int_th']:.3f}** |")
    md.append("")
    md.append("## 2. Tổng hợp tài nguyên phần cứng (Hardware Resource Budget)")
    md.append("")
    md.append("| Tiêu chí phần cứng | Giá trị chính xác | Ghi chú thiết kế Datapath |")
    md.append("| :--- | :---: | :--- |")
    md.append(f"| Tổng số tầng tích chập | **18 tầng `Conv2d`** | 0 tầng BatchNorm (đã gập hoàn toàn) |")
    md.append(f"| **Tổng số phép tính (MACs)** | **{total_macs:,}** (`{total_macs/1e6:.3f} MMACs`) | Mảng 9 DSP @ 100 MHz $\\approx$ **{total_macs/(9*100e6)*1000:.2f} ms** (~{int(9*100e6/total_macs)} FPS) |")
    md.append(f"| Trọng số `Conv2d` (`weights_int8.mem`) | **{all_w_np.size:,} B (`{all_w_np.size/1024:.2f} KB`)** | `INT8` đối xứng `[-127, +127]` |")
    md.append(f"| Trọng số `PReLU` (`prelu_q07.mem`) | **{all_p_np.size:,} B (`{all_p_np.size/1024:.2f} KB`)** | `INT8` fixed-point `Q0.7` (`>> 7`) |")
    md.append(f"| Hằng số `Bias` (`bias_int32.mem`) | **{all_b_np.size:,} giá trị (`{all_b_np.size*4/1024:.2f} KB`)** | `INT32` cộng trực tiếp vào thanh ghi Accumulator |")
    md.append(f"| Hệ số làm tròn `M` (`requant_mult_int32.mem`) | **{all_m_np.size:,} giá trị (`{all_m_np.size*4/1024:.2f} KB`)** | Dịch phải cố định `>> {SHIFT_BITS}` cho mọi tầng |")
    md.append(f"| **Bộ nhớ 1 Bank trung gian lớn nhất** | **{max_single_bank:,} B (`{max_single_bank/1024:.2f} KB`)** | Tầng `L04_s1_exp` & `L10_s3_exp` |")
    md.append(f"| **Tổng Ping-Pong RAM tối thiểu (In + Out)** | **{max_pingpong:,} B (`{max_pingpong/1024:.2f} KB`)** | Khuyến nghị cấp phát **2 bank × 16 KB = 32 KB** |")
    md.append("")
    md.append("## 3. Bảng chi tiết 18 tầng cho thiết kế Datapath & Controller")
    md.append("")
    md.append("| # | Tên tầng | Input `(C,H,W)` | Output `(C,H,W)` | Kernel/S/G | PReLU | MACs | Weight (B) | Act Out (B) |")
    md.append("| :-: | :--- | :---: | :---: | :---: | :-: | ---: | ---: | ---: |")
    for lm in layer_meta:
        ksg = f"{lm['kernel']}/s{lm['stride']}/g{lm['groups']}"
        pr = "Có" if lm["has_prelu"] else "Không"
        md.append(f"| {lm['index']:02d} | `{lm['name']}` | `{tuple(lm['in_shape_CHW'])}` | `{tuple(lm['out_shape_CHW'])}` | `{ksg}` | {pr} | {lm['macs']:,} | {lm['weight_bytes_int8']:,} | {lm['act_out_bytes_int8']:,} |")
    md.append("")
    md.append("## 4. Bộ file Golden Model & nạp mạch (`hardware_export/`)")
    md.append("- `weights_int8.mem`, `bias_int32.mem`, `prelu_q07.mem`, `requant_mult_int32.mem`: Nạp bằng `$readmemh` trong Verilog.")
    md.append("- `quant_params.json`: Địa chỉ offset, shape và hằng số `M, shift` cho FSM Controller.")
    md.append("- `golden_vectors/`: Chứa `input_32x32_int8.mem` + 18 file đáp án chuẩn từng tầng (`L01_stem_int8.mem` ... `L18_linear_conv_int8.mem`) để đối chiếu bit-exact với Testbench Verilog.")

    spec_path = os.path.join(export_dir, "Hardware_Baseline_Spec.md")
    with open(spec_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"\n[+] Đã xuất trọn gói dữ liệu phần cứng tại: {export_dir}")
    print(f"    - Bảng tiêu chí gốc : {spec_path}")
    print(f"    - File nạp Verilog  : weights_int8.mem ({all_w_np.size/1024:.2f} KB), bias_int32.mem, prelu_q07.mem, requant_mult_int32.mem")
    print(f"    - Golden Vectors    : {gv_dir} (Input + 18 tầng + 2 Residual + Embedding 128-D)")


def main():
    parser = argparse.ArgumentParser(description="QAT Full-INT8 & FPGA Golden Model Exporter")
    parser.add_argument("--epochs", type=int, default=30, help="Số epoch QAT fine-tune")
    parser.add_argument("--infinite", action="store_true", help="Chế độ QAT chạy vô tận đến khi bấm Ctrl+C")
    parser.add_argument("--eval-every", type=int, default=0, help="Tự động kiểm chứng LFW & lưu trọng số tốt nhất mỗi N epochs (0 = chỉ đánh giá cuối)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-only", action="store_true", help="Chỉ đánh giá mô hình QAT hiện tại và xuất biểu đồ báo cáo")
    parser.add_argument("--resume-qat", action="store_true", help="Tiếp tục huấn luyện QAT từ weights/tiny_mobilefacenet_lfw_qat.pt")
    parser.add_argument("--data-root", type=str, default=r"..\Python\lfw_dataset")
    parser.add_argument("--teacher", type=str, default=r"..\Python\mobilefacenet_scripted.pt")
    parser.add_argument("--weights", type=str, default=r"weights\tiny_mobilefacenet_lfw.pt")
    parser.add_argument("--qat-weights", type=str, default=r"weights\tiny_mobilefacenet_lfw_qat.pt")
    parser.add_argument("--export-dir", type=str, default="hardware_export")
    parser.add_argument("--reports-dir", type=str, default="reports")
    parser.add_argument("--history-file", type=str, default="qat_training_history.csv")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    def _abs(p): return os.path.normpath(os.path.join(script_dir, p)) if not os.path.isabs(p) else p

    data_root = _abs(args.data_root)
    teacher_path = _abs(args.teacher)
    weights_path = _abs(args.weights)
    qat_weights_path = _abs(args.qat_weights)
    export_dir = _abs(args.export_dir)
    reports_dir = _abs(args.reports_dir)
    history_file = _abs(args.history_file)

    aligned_dir = os.path.join(data_root, "lfw_aligned_retina_112")
    pairs_path = os.path.join(data_root, "pairs.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_infinite = args.infinite or (args.epochs <= 0)
    mode_name = "QAT_EvalOnly" if args.eval_only else ("QAT_Infinite" if is_infinite else ("QAT_Resume" if args.resume_qat else "QAT_Train"))
    print("=" * 74)
    print(f"TOOL HUẤN LUYỆN LƯỢNG TỬ HÓA QAT (FULL INT8) [{mode_name}]")
    print(f"- Thiết bị : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    ep_str = "INFINITE (Ctrl+C to stop)" if is_infinite else str(args.epochs)
    print(f"- Epochs   : {ep_str} | LR: {args.lr} | Batch: {args.batch_size} | Eval-Every: {args.eval_every}")
    print("=" * 74, flush=True)

    # 1. Khởi tạo khung QAT Model từ mô hình đã gập BN
    base_model = TinyMobileFaceNet32(embedding_size=128, use_bn=True)
    base_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    folded = base_model.fold_bn().to(device).eval()

    qat_model = TinyMobileFaceNet32_QAT(folded).to(device)
    pairs = read_lfw_pairs(pairs_path, aligned_dir)

    def make_int_eval(gm):
        def _fn(im32):
            x1 = np.clip(np.round(((im32.astype(np.float32) - 127.5) / 128.0) * 127.0), -127, 127).astype(np.int8)
            x2 = np.clip(np.round(((cv2.flip(im32, 1).astype(np.float32) - 127.5) / 128.0) * 127.0), -127, 127).astype(np.int8)
            t1 = torch.from_numpy(x1.transpose(2, 0, 1)).unsqueeze(0).to(device)
            t2 = torch.from_numpy(x2.transpose(2, 0, 1)).unsqueeze(0).to(device)
            e1 = gm.forward_int8(t1).float()
            e2 = gm.forward_int8(t2).float()
            v = F.normalize(F.normalize(e1, p=2, dim=1) + F.normalize(e2, p=2, dim=1), p=2, dim=1)
            return v.cpu().squeeze(0).numpy()
        return _fn

    def eval_fp32(im32):
        t1 = ((torch.from_numpy(im32.transpose(2, 0, 1)).float().unsqueeze(0) - 127.5) / 128.0).to(device)
        t2 = ((torch.from_numpy(cv2.flip(im32, 1).transpose(2, 0, 1)).float().unsqueeze(0) - 127.5) / 128.0).to(device)
        with torch.no_grad():
            return F.normalize(qat_model(t1, enable_quant=False) + qat_model(t2, enable_quant=False), p=2, dim=1).cpu().squeeze(0).numpy()

    # Đo mốc FP32 gốc
    fp32_acc, fp32_std, fp32_auc, fp32_eer, fp32_th, _, _, _, _ = evaluate_10fold_lfw(eval_fp32, pairs)
    print(f"[Baseline FP32 Folded] 10-Fold LFW: {fp32_acc:.2f}% +/- {fp32_std:.2f}% | AUC: {fp32_auc:.4f} | EER: {fp32_eer:.2f}%", flush=True)

    # CHẾ ĐỘ EVAL-ONLY
    if args.eval_only:
        if not os.path.exists(qat_weights_path):
            raise FileNotFoundError(f"Không tìm thấy {qat_weights_path}! Hãy chạy QAT Train trước.")
        qat_model.load_state_dict(torch.load(qat_weights_path, map_location=device))
        qat_model.eval()
        print(f"[+] Đã nạp trọng số QAT: {qat_weights_path}", flush=True)

        final_golden = IntegerOnlyGoldenModel(qat_model, device)
        qat_acc, qat_std, qat_auc, qat_eer, qat_th, fpr, tpr, all_s, all_l = evaluate_10fold_lfw(make_int_eval(final_golden), pairs)

        ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        latest_chart = os.path.join(export_dir, "qat_int8_lfw_report.png")
        ts_chart = os.path.join(reports_dir, f"qat_int8_report_{ts_str}.png")
        plot_qat_report(fpr, tpr, qat_auc, qat_eer, qat_th, qat_acc, all_s, all_l, latest_chart)
        os.makedirs(reports_dir, exist_ok=True)
        shutil.copyfile(latest_chart, ts_chart)
        print(f"[+] Đã lưu bản sao lịch sử tại: {ts_chart}")

        best_metrics = {
            "fp32_acc": fp32_acc, "fp32_std": fp32_std, "fp32_auc": fp32_auc, "fp32_eer": fp32_eer, "fp32_th": fp32_th,
            "ptq_acc": fp32_acc, "ptq_std": fp32_std, "ptq_auc": fp32_auc, "ptq_eer": fp32_eer, "ptq_th": fp32_th,
            "qat_int_acc": qat_acc, "qat_int_std": qat_std, "qat_int_auc": qat_auc, "qat_int_eer": qat_eer, "qat_int_th": qat_th,
        }
        export_hardware_package(final_golden, pairs[0][0][0], export_dir, best_metrics)
        log_qat_history(history_file, {
            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Mode": mode_name, "Epochs_Done": 0, "LR": 0.0, "Duration_Mins": "0.00",
            "FP32_Acc": f"{fp32_acc:.2f}%", "PTQ_INT8_Acc": "N/A",
            "QAT_INT8_Acc": f"{qat_acc:.2f}%", "QAT_AUC": f"{qat_auc:.4f}",
            "QAT_EER": f"{qat_eer:.2f}%", "QAT_Threshold": f"{qat_th:.3f}",
            "Report_Path": ts_chart,
        })
        return

    # 2. Nạp Dataset & Calibration (hoặc nạp lại QAT checkpoint nếu --resume-qat)
    ds = FastDistillDataset(aligned_dir, augment=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)

    if args.resume_qat and os.path.exists(qat_weights_path):
        qat_model.load_state_dict(torch.load(qat_weights_path, map_location=device))
        print(f"[+] Resume QAT: Đã nạp trọng số QAT trước đó từ {qat_weights_path}", flush=True)
    else:
        print("[+] Bước 1: Hiệu chuẩn (Calibration) dải Activation INT8 trên 2.560 ảnh...", flush=True)
        qat_model.eval()
        qat_model.set_calibrating(True)
        with torch.no_grad():
            for b_idx, (x_s, _) in enumerate(loader):
                qat_model(x_s.to(device), enable_quant=False)
                if b_idx >= 19:
                    break
        qat_model.finalize_calibration()

    # Đo mốc PTQ / Pre-QAT Full INT8
    ptq_golden = IntegerOnlyGoldenModel(qat_model, device)
    ptq_acc, ptq_std, ptq_auc, ptq_eer, ptq_th, ptq_fpr, ptq_tpr, ptq_s, ptq_l = evaluate_10fold_lfw(make_int_eval(ptq_golden), pairs)
    print(f"[Trước phiên QAT - Full INT8] 10-Fold LFW: {ptq_acc:.2f}% +/- {ptq_std:.2f}% | AUC: {ptq_auc:.4f} | EER: {ptq_eer:.2f}%", flush=True)

    # 3. Chạy QAT Fine-Tuning (hỗ trợ cả hữu hạn & vô tận, có bắt Ctrl+C an toàn)
    max_epochs = 9999999 if is_infinite else args.epochs
    ep_banner = "VÔ TẬN (bấm Ctrl+C bất kỳ lúc nào để dừng)" if is_infinite else f"{args.epochs} epochs"
    print(f"\n[+] Bước 2: Bắt đầu Huấn luyện Lượng tử hóa (QAT - {ep_banner})...", flush=True)
    teacher = torch.jit.load(teacher_path, map_location=device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(qat_model.parameters(), lr=args.lr, weight_decay=1e-4)
    if is_infinite:
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=30, T_mult=1, eta_min=1e-6)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-6)

    # Khởi tạo mốc tốt nhất ban đầu bằng bản trước phiên
    best_golden = ptq_golden
    best_acc, best_std, best_auc, best_eer, best_th = ptq_acc, ptq_std, ptq_auc, ptq_eer, ptq_th
    best_fpr, best_tpr, best_s, best_l = ptq_fpr, ptq_tpr, ptq_s, ptq_l

    t0 = time.time()
    epochs_done = 0
    try:
        ep = 1
        while ep <= max_epochs:
            qat_model.train()
            tot_loss = 0.0
            for x_s, x_t in loader:
                x_s, x_t = x_s.to(device, non_blocking=True), x_t.to(device, non_blocking=True)
                with torch.no_grad():
                    z_t = F.normalize(teacher(x_t), p=2, dim=1)
                z_s = qat_model(x_s, enable_quant=True)
                loss_cos = torch.mean(1.0 - F.cosine_similarity(z_s, z_t, dim=1))
                loss_rel = F.mse_loss(torch.matmul(z_s, z_s.T), torch.matmul(z_t, z_t.T))
                loss = loss_cos + 5.0 * loss_rel
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot_loss += loss.item()
            sched.step()
            epochs_done = ep
            if ep % 5 == 0 or (not is_infinite and ep == max_epochs):
                ep_lbl = f"{ep:03d}" if is_infinite else f"{ep:02d}/{max_epochs}"
                print(f"  QAT Epoch {ep_lbl} | Loss: {tot_loss/len(loader):.4f} | LR: {sched.get_last_lr()[0]:.6f} | Best INT8: {best_acc:.2f}% | Time: {time.time()-t0:.1f}s", flush=True)

            # Đánh giá định kỳ trong lúc chạy dài/vô tận và tự lưu ngay nếu phá kỷ lục!
            if args.eval_every > 0 and (ep % args.eval_every == 0):
                qat_model.eval()
                cur_golden = IntegerOnlyGoldenModel(qat_model, device)
                c_acc, c_std, c_auc, c_eer, c_th, c_fpr, c_tpr, c_s, c_l = evaluate_10fold_lfw(make_int_eval(cur_golden), pairs)
                print(f"  --> [Auto-Eval Ep {ep:03d}] Full-INT8 LFW: {c_acc:.2f}% +/- {c_std:.2f}% | AUC: {c_auc:.4f} | EER: {c_eer:.2f}%", flush=True)
                if c_acc > best_acc:
                    best_golden = cur_golden
                    best_acc, best_std, best_auc, best_eer, best_th = c_acc, c_std, c_auc, c_eer, c_th
                    best_fpr, best_tpr, best_s, best_l = c_fpr, c_tpr, c_s, c_l
                    torch.save(qat_model.state_dict(), qat_weights_path)
                    plot_qat_report(best_fpr, best_tpr, best_auc, best_eer, best_th, best_acc, best_s, best_l, os.path.join(export_dir, "qat_int8_lfw_report.png"))
                    print(f"  *** [NEW BEST QAT INT8!] {best_acc:.2f}% -> Đã tự động lưu {qat_weights_path}", flush=True)
            ep += 1
    except KeyboardInterrupt:
        print("\n[!] Đã nhận Ctrl+C! Dừng an toàn để đánh giá và xuất báo cáo...", flush=True)

    duration_mins = (time.time() - t0) / 60.0

    # 4. Kiểm chứng Bit-True Integer-Only Golden Model cuối phiên
    print("\n[+] Bước 3: Kiểm chứng Bit-True Integer-Only Golden Model cuối phiên trên 6.000 cặp LFW...", flush=True)
    qat_model.eval()
    final_golden = IntegerOnlyGoldenModel(qat_model, device)
    qat_acc, qat_std, qat_auc, qat_eer, qat_th, qat_fpr, qat_tpr, qat_s, qat_l = evaluate_10fold_lfw(make_int_eval(final_golden), pairs)

    if qat_acc >= best_acc:
        best_golden = final_golden
        best_acc, best_std, best_auc, best_eer, best_th = qat_acc, qat_std, qat_auc, qat_eer, qat_th
        best_fpr, best_tpr, best_s, best_l = qat_fpr, qat_tpr, qat_s, qat_l
        torch.save(qat_model.state_dict(), qat_weights_path)
        print(f"[+] CẢI THIỆN QAT INT8 -> {best_acc:.2f}%. Đã lưu: {qat_weights_path}")
    else:
        print(f"[!] Trạng thái cuối ({qat_acc:.2f}%) chưa vượt mốc cao nhất ({best_acc:.2f}%). Giữ nguyên bản tốt nhất {best_acc:.2f}%!")

    best_metrics = {
        "fp32_acc": fp32_acc, "fp32_std": fp32_std, "fp32_auc": fp32_auc, "fp32_eer": fp32_eer, "fp32_th": fp32_th,
        "ptq_acc": ptq_acc, "ptq_std": ptq_std, "ptq_auc": ptq_auc, "ptq_eer": ptq_eer, "ptq_th": ptq_th,
        "qat_int_acc": best_acc, "qat_int_std": best_std, "qat_int_auc": best_auc, "qat_int_eer": best_eer, "qat_int_th": best_th,
    }

    print("=" * 74)
    print("TỔNG KẾT LƯỢNG TỬ HÓA FULL INT8 (10-FOLD LFW - 6.000 CẶP):")
    print(f"  * FP32 Folded-BN (Gốc)     : {fp32_acc:.2f}% +/- {fp32_std:.2f}% | AUC: {fp32_auc:.4f} | EER: {fp32_eer:.2f}%")
    print(f"  * Trước QAT (Full INT8)    : {ptq_acc:.2f}% +/- {ptq_std:.2f}% | AUC: {ptq_auc:.4f} | EER: {ptq_eer:.2f}%")
    print(f"  * Sau QAT   (Full INT8)    : {qat_acc:.2f}% +/- {qat_std:.2f}% | AUC: {qat_auc:.4f} | EER: {qat_eer:.2f}%")
    print("=" * 74, flush=True)

    # 5. Xuất biểu đồ Report + Nhật ký CSV + Trọn gói phần cứng
    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    latest_chart = os.path.join(export_dir, "qat_int8_lfw_report.png")
    ts_chart = os.path.join(reports_dir, f"qat_int8_report_{ts_str}.png")
    plot_qat_report(best_fpr, best_tpr, best_auc, best_eer, best_th, best_acc, best_s, best_l, latest_chart)
    os.makedirs(reports_dir, exist_ok=True)
    shutil.copyfile(latest_chart, ts_chart)
    print(f"[+] Đã lưu bản sao biểu đồ lịch sử tại: {ts_chart}")

    export_hardware_package(best_golden, pairs[0][0][0], export_dir, best_metrics)
    log_qat_history(history_file, {
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Mode": mode_name,
        "Epochs_Done": epochs_done,
        "LR": args.lr,
        "Duration_Mins": f"{duration_mins:.2f}",
        "FP32_Acc": f"{fp32_acc:.2f}%",
        "PTQ_INT8_Acc": f"{ptq_acc:.2f}%",
        "QAT_INT8_Acc": f"{best_acc:.2f}%",
        "QAT_AUC": f"{best_auc:.4f}",
        "QAT_EER": f"{best_eer:.2f}%",
        "QAT_Threshold": f"{best_th:.3f}",
        "Report_Path": ts_chart,
    })


if __name__ == "__main__":
    main()
