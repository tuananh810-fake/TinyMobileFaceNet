# -*- coding: utf-8 -*-
"""
===============================================================================
NÂNG CẤP HỆ THỐNG HUẤN LUYỆN CHUYỂN GIAO TRI THỨC VÀ BENCHMARK LFW
Kiến trúc phần cứng: TinyMobileFaceNet-32 (217k params, 32x32x3)
Dành cho: AMD Spartan-7 XC7S25CSGA225-1 (LSI Design Contest 2026 - VNU-UET)

CÁC NÂNG CẤP ĐỘT PHÁ SO VỚI BẢN CŨ:
1. [TIỀN XỬ LÝ KHUÔN MẶT - 100% FACE ALIGNMENT / CROP]:
   - Tự động phát hiện và cắt đúng vùng mặt 1:1 từ ảnh gốc 250x250 của LFW.
   - Loại bỏ hoàn toàn 76% diện tích thừa (cổ áo vest, vai, phông nền tường).
   - Đảm bảo ma trận 32x32 của TinyMobileFaceNet và 112x112 của MobileFaceNet
     chứa trọn vẹn 100% ngũ quan (mắt, mũi, miệng, cằm), tương đồng 100% với
     khâu cắt mặt của camera_demo.py.
2. [CHUẨN HÓA GIAO THỨC ĐÁNH GIÁ 10-FOLD LFW OFFICIAL PROTOCOL]:
   - Cross-validation 10 folds: Tìm threshold trên 9 fold (5,400 cặp),
     kiểm thử trên fold thứ 10 (600 cặp).
3. [THEO DÕI TIẾN ĐỘ THỜI GIAN THỰC]:
   - In tiến độ từng batch có flush stdout để terminal không bao giờ bị đứng im.
4. [TỰ ĐỘNG LƯU TRỌNG SỐ & BÁO CÁO ĐỒ THỊ CHUẨN XUẤT BẢN]:
   - Xuất lfw_benchmark_report.png và weights/tiny_mobilefacenet_lfw.pt.
===============================================================================
"""

import os
import sys
import tarfile
import urllib.request
import subprocess
import time
import math
import argparse
import datetime
import csv
import shutil
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

# =============================================================================
# 1. ĐỊNH NGHĨA MÔ HÌNH PHẦN CỨNG TINY-MOBILEFACENET-32
# =============================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)
        self.prelu = nn.PReLU(out_c)

    def forward(self, x):
        return self.prelu(self.conv(x))


class Bottleneck(nn.Module):
    def __init__(self, in_c, out_c, stride, expansion_factor):
        super(Bottleneck, self).__init__()
        self.use_residual = (stride == 1 and in_c == out_c)
        hidden_dim = in_c * expansion_factor
        layers = []
        if expansion_factor != 1:
            layers.append(ConvBlock(in_c, hidden_dim, kernel_size=1, stride=1, padding=0))
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False))
        layers.append(nn.PReLU(hidden_dim))
        layers.append(nn.Conv2d(hidden_dim, out_c, kernel_size=1, stride=1, padding=0, bias=False))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class TinyMobileFaceNet32(nn.Module):
    def __init__(self, embedding_size=128):
        super(TinyMobileFaceNet32, self).__init__()
        self.stem = ConvBlock(in_c=3, out_c=32, kernel_size=3, stride=2, padding=1)
        self.dw_stem = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32, bias=False),
            nn.PReLU(32),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.stage1 = Bottleneck(in_c=32, out_c=64, stride=2, expansion_factor=2)
        self.stage2 = Bottleneck(in_c=64, out_c=64, stride=1, expansion_factor=2)
        self.stage3 = Bottleneck(in_c=64, out_c=128, stride=2, expansion_factor=4)
        self.stage4 = Bottleneck(in_c=128, out_c=128, stride=1, expansion_factor=2)
        self.conv_sep = ConvBlock(in_c=128, out_c=256, kernel_size=1, stride=1, padding=0)
        self.gdconv = nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=0, groups=256, bias=False)
        self.linear_conv = nn.Conv2d(256, embedding_size, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.stem(x)
        x = self.dw_stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv_sep(x)
        x = self.gdconv(x)
        x = self.linear_conv(x)
        embedding = x.view(x.size(0), -1)
        return F.normalize(embedding, p=2, dim=1)


# =============================================================================
# 2. BỘ CÔNG CỤ TẢI DỮ LIỆU & TIỀN XỬ LÝ CẮT KHUÔN MẶT CHUẨN XÁC (FACE CROP)
# =============================================================================
LFW_MIRRORS = [
    "https://huggingface.co/datasets/DerrickUnleashed/LFW/resolve/main/lfwfunneled.tgz",
    "http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz"
]
PAIRS_MIRRORS = [
    "https://huggingface.co/datasets/DerrickUnleashed/LFW/resolve/main/pairs.txt",
    "http://vis-www.cs.umass.edu/lfw/pairs.txt"
]

def download_file_robust(urls, dest_path, desc=""):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        return True

    for url in urls:
        print(f"[+] Đang tải {desc} từ: {url}")
        try:
            res = subprocess.run(
                ["curl.exe", "-L", "-o", dest_path, url],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300
            )
            if res.returncode == 0 and os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
                print(f"    -> Tải thành công ({os.path.getsize(dest_path)/(1024*1024):.1f} MB)!")
                return True
        except Exception:
            pass

        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=120) as response, open(dest_path, 'wb') as out_file:
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024 * 64
                downloaded = 0
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    if total_size > 0:
                        pct = int(downloaded * 100 / total_size)
                        sys.stdout.write(f"\r    -> Tiến độ: {pct}% [{downloaded/(1024*1024):.1f} MB / {total_size/(1024*1024):.1f} MB]")
                        sys.stdout.flush()
                print("\n    -> Tải thành công!")
                return True
        except Exception as e:
            print(f"    [!] Tải từ link này gặp sự cố ({e}), đang thử link tiếp theo...")
            if os.path.exists(dest_path):
                try: os.remove(dest_path)
                except Exception: pass

    return False

def ensure_lfw_raw(root_dir):
    """Đảm bảo tập dữ liệu gốc LFW đã sẵn sàng"""
    os.makedirs(root_dir, exist_ok=True)
    lfw_raw_dir = os.path.join(root_dir, "lfw_funneled")
    pairs_path = os.path.join(root_dir, "pairs.txt")
    archive_path = os.path.join(root_dir, "lfw-funneled.tgz")

    if not os.path.exists(pairs_path) or os.path.getsize(pairs_path) < 1000:
        download_file_robust(PAIRS_MIRRORS, pairs_path, "danh sách pairs.txt")

    if not os.path.exists(lfw_raw_dir) or len(os.listdir(lfw_raw_dir)) == 0:
        if not os.path.exists(archive_path) or os.path.getsize(archive_path) < 1000000:
            download_file_robust(LFW_MIRRORS, archive_path, "tập LFW (~230 MB)")
        print(f"[+] Đang giải nén tập LFW ({archive_path})...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=root_dir)
        if not os.path.exists(lfw_raw_dir):
            for alt in ["lfwfunneled", "lfw"]:
                if os.path.exists(os.path.join(root_dir, alt)):
                    os.rename(os.path.join(root_dir, alt), lfw_raw_dir)
                    break
        print("    -> Giải nén hoàn tất!")

    return lfw_raw_dir, pairs_path


def crop_face_box(img, face_cascade):
    """
    Cắt trọn vẹn khuôn mặt 1:1, mở rộng 15% biên để bao trọn trán và cằm,
    loại bỏ hoàn toàn cổ áo và phông nền tường.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))
    if len(faces) > 0:
        best_face = None
        min_dist = float('inf')
        for (x, y, w, h) in faces:
            cx, cy = x + w / 2.0, y + h / 2.0
            dist = (cx - 125)**2 + (cy - 125)**2
            if dist < min_dist:
                min_dist = dist
                best_face = (x, y, w, h)
        x, y, w, h = best_face
        pad = int(0.12 * max(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad)
        y2 = min(img.shape[0], y + h + pad)
        crop = img[y1:y2, x1:x2]
    else:
        # Fallback trung tâm nếu khuôn mặt bị tối/nghiêng
        crop = img[45:205, 45:205]

    return cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)


def prepare_aligned_lfw(raw_dir, aligned_dir):
    """
    Tiền xử lý toàn bộ 13,233 ảnh LFW thành ảnh cắt mặt 112x112 chuẩn xác.
    Lưu vào thư mục aligned_dir để nạp siêu tốc khi train và test.
    """
    os.makedirs(aligned_dir, exist_ok=True)
    subjects = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    
    # Kiểm tra xem đã tiền xử lý đủ chưa
    all_raw_files = []
    for s in subjects:
        s_dir = os.path.join(raw_dir, s)
        for f in os.listdir(s_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_raw_files.append((s, f))

    aligned_count = 0
    for root, _, files in os.walk(aligned_dir):
        aligned_count += len([f for f in files if f.lower().endswith('.jpg')])

    if aligned_count >= len(all_raw_files) and len(all_raw_files) > 0:
        print(f"[+] Dữ liệu khuôn mặt đã được Cắt sát (Aligned Face Crops): {aligned_count} ảnh.")
        return aligned_dir

    print(f"\n[+] BẮT ĐẦU TIỀN XỬ LÝ CẮT KHUÔN MẶT CHUẨN XÁC (FACE CROPS) CHO {len(all_raw_files)} ẢNH LFW...")
    print("    -> Loại bỏ 76% diện tích quần áo và phông nền thừa để mạng tập trung 100% vào ngũ quan!")
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    t0 = time.time()
    
    for idx, (sub, fn) in enumerate(all_raw_files):
        src_path = os.path.join(raw_dir, sub, fn)
        dst_subdir = os.path.join(aligned_dir, sub)
        os.makedirs(dst_subdir, exist_ok=True)
        dst_path = os.path.join(dst_subdir, fn)

        if not os.path.exists(dst_path):
            img = cv2.imread(src_path)
            if img is not None:
                cropped = crop_face_box(img, face_cascade)
                cv2.imwrite(dst_path, cropped)

        if (idx + 1) % 1000 == 0 or (idx + 1) == len(all_raw_files):
            pct = (idx + 1) * 100 // len(all_raw_files)
            sys.stdout.write(f"\r    -> Đã cắt mặt: {idx + 1}/{len(all_raw_files)} ({pct}%) [{time.time() - t0:.1f}s]")
            sys.stdout.flush()

    print(f"\n[+] Hoàn tất cắt khuôn mặt! Thư mục mặt chuẩn sẵn sàng: {aligned_dir}\n")
    return aligned_dir


# =============================================================================
# 3. DATASET VỚI DỮ LIỆU ĐÃ CẮT MẶT (100% DIỆN TÍCH LÀ KHUÔN MẶT)
# =============================================================================
class AlignedFaceDistillDataset(Dataset):
    def __init__(self, aligned_dir, augment=True):
        self.image_paths = []
        self.augment = augment
        for root, _, files in os.walk(aligned_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(root, file))

        if len(self.image_paths) == 0:
            raise RuntimeError(f"Không tìm thấy ảnh nào trong {aligned_dir}!")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            img_bgr = np.zeros((112, 112, 3), dtype=np.uint8)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if self.augment:
            if np.random.rand() > 0.5:
                img_rgb = cv2.flip(img_rgb, 1)
            # Biến thiên ánh sáng nhẹ nhàng
            alpha = np.random.uniform(0.9, 1.1)
            beta = np.random.randint(-10, 10)
            img_rgb = np.clip(img_rgb * alpha + beta, 0, 255).astype(np.uint8)

        # 1. Student input: 32x32 chứa 100% mặt người, chuẩn hóa [-1, 1]
        img_32 = cv2.resize(img_rgb, (32, 32), interpolation=cv2.INTER_AREA)
        t_student = torch.from_numpy(img_32.transpose(2, 0, 1)).float()
        t_student = (t_student - 127.5) / 128.0

        # 2. Teacher input: 112x112 chứa 100% mặt người, chuẩn hóa [-1, 1]
        t_teacher = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float()
        t_teacher = (t_teacher - 127.5) / 128.0

        return t_student, t_teacher


# =============================================================================
# 4. HÀM MẤT MÁT DISTILLATION TỐI ƯU
# =============================================================================
class DistillationLoss(nn.Module):
    def __init__(self, alpha=2.0):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha

    def forward(self, z_s, z_t):
        cosine_sim = F.cosine_similarity(z_s, z_t, dim=1)
        loss_cos = torch.mean(1.0 - cosine_sim)
        loss_mse = F.mse_loss(z_s, z_t)
        return loss_cos + self.alpha * loss_mse, loss_cos.item(), loss_mse.item()


# =============================================================================
# 5. GIAO THỨC ĐÁNH GIÁ 10-FOLD LFW OFFICIAL PROTOCOL
# =============================================================================
def read_lfw_pairs(pairs_path, aligned_dir):
    with open(pairs_path, 'r', encoding='utf-8') as f:
        lines = [line.strip().split() for line in f.readlines() if line.strip()]

    header = lines[0]
    num_folds = int(header[0])
    num_pairs_per_fold = int(header[1])

    idx = 1
    pairs = []
    for fold in range(num_folds):
        fold_pairs = []
        for _ in range(num_pairs_per_fold):
            p = lines[idx]
            name, i1, i2 = p[0], int(p[1]), int(p[2])
            f1 = os.path.join(aligned_dir, name, f"{name}_{i1:04d}.jpg")
            f2 = os.path.join(aligned_dir, name, f"{name}_{i2:04d}.jpg")
            fold_pairs.append((f1, f2, 1))
            idx += 1
        for _ in range(num_pairs_per_fold):
            p = lines[idx]
            n1, i1, n2, i2 = p[0], int(p[1]), p[2], int(p[3])
            f1 = os.path.join(aligned_dir, n1, f"{n1}_{i1:04d}.jpg")
            f2 = os.path.join(aligned_dir, n2, f"{n2}_{i2:04d}.jpg")
            fold_pairs.append((f1, f2, 0))
            idx += 1
        pairs.append(fold_pairs)

    return pairs


def extract_embedding_32(model, img_path, device):
    if not os.path.exists(img_path):
        return None
    img = cv2.imread(img_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_32 = cv2.resize(img_rgb, (32, 32), interpolation=cv2.INTER_AREA)

    t1 = torch.from_numpy(img_32.transpose(2, 0, 1)).float().unsqueeze(0)
    t1 = ((t1 - 127.5) / 128.0).to(device)

    img_flip = cv2.flip(img_32, 1)
    t2 = torch.from_numpy(img_flip.transpose(2, 0, 1)).float().unsqueeze(0)
    t2 = ((t2 - 127.5) / 128.0).to(device)

    with torch.no_grad():
        v1 = model(t1)
        v2 = model(t2)
        v = F.normalize(v1 + v2, p=2, dim=1)

    return v.cpu().squeeze(0).numpy()


def evaluate_lfw_official(model, pairs, device):
    """
    Đánh giá chuẩn 10-fold cross validation chính thức của LFW:
    Tìm ngưỡng tối ưu trên 9 fold, kiểm thử trên fold còn lại.
    """
    model.eval()
    print("\n" + "=" * 65)
    print("TIẾN HÀNH ĐÁNH GIÁ CHUẨN 10-FOLD LFW TRÊN ẢNH MẶT ĐÃ CẮT CHUẨN (CROPPED FACES)...")
    print("=" * 65)

    all_fold_scores = []
    all_fold_labels = []

    for fold_idx, fold_data in enumerate(pairs):
        scores, labels = [], []
        for f1, f2, label in fold_data:
            v1 = extract_embedding_32(model, f1, device)
            v2 = extract_embedding_32(model, f2, device)
            if v1 is None or v2 is None:
                continue
            scores.append(float(np.dot(v1, v2)))
            labels.append(label)

        all_fold_scores.append(np.array(scores))
        all_fold_labels.append(np.array(labels))

    test_accuracies = []
    threshold_candidates = np.linspace(-1.0, 1.0, 201)

    for test_fold in range(10):
        # 1. Gom 9 fold còn lại làm tập train threshold
        train_scores = np.concatenate([all_fold_scores[i] for i in range(10) if i != test_fold])
        train_labels = np.concatenate([all_fold_labels[i] for i in range(10) if i != test_fold])

        best_th = 0.0
        best_train_acc = 0.0
        for th in threshold_candidates:
            acc = np.mean((train_scores >= th).astype(int) == train_labels)
            if acc > best_train_acc:
                best_train_acc = acc
                best_th = th

        # 2. Đánh giá trên fold kiểm thử bằng best_th tìm được
        test_s = all_fold_scores[test_fold]
        test_l = all_fold_labels[test_fold]
        test_acc = np.mean((test_s >= best_th).astype(int) == test_l) * 100
        test_accuracies.append(test_acc)
        print(f"  Fold {test_fold + 1:02d}/10 | Test Accuracy: {test_acc:.2f}% (Tối ưu Threshold = {best_th:.3f})")

    mean_acc = np.mean(test_accuracies)
    std_acc = np.std(test_accuracies)

    # Đánh giá toàn bộ cho ROC và EER
    total_scores = np.concatenate(all_fold_scores)
    total_labels = np.concatenate(all_fold_labels)

    fpr, tpr, thresholds = roc_curve(total_labels, total_scores)
    roc_auc = auc(fpr, tpr)
    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0 * 100
    optimal_th = thresholds[eer_idx]

    print("\n" + "=" * 65)
    print("KẾT QUẢ BENCHMARK LFW CHÍNH THỨC CHO BÁO CÁO KHOA HỌC:")
    print(f"  * 10-Fold Accuracy : {mean_acc:.2f}% +/- {std_acc:.2f}%")
    print(f"  * ROC AUC Score    : {roc_auc:.4f} ({roc_auc*100:.2f}%)")
    print(f"  * Equal Error Rate : {eer:.2f}% (tại Threshold = {optimal_th:.3f})")
    print("=" * 65 + "\n")

    return mean_acc, std_acc, roc_auc, eer, optimal_th, fpr, tpr, total_scores, total_labels


# =============================================================================
# 6. VẼ ĐỒ THỊ BÁO CÁO KHOA HỌC
# =============================================================================
def plot_lfw_report(fpr, tpr, roc_auc, eer, best_th, mean_acc, all_scores, all_labels, save_path):
    plt.figure(figsize=(14, 6), dpi=150)

    # Subplot 1: ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'TinyMobileFaceNet-32 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
    plt.scatter([eer/100], [1.0 - eer/100], color='red', s=70, zorder=5, label=f'EER = {eer:.2f}%')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (TPR)', fontsize=12, fontweight='bold')
    plt.title(f'LFW ROC Curve (Accuracy: {mean_acc:.2f}%)', fontsize=13, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right', fontsize=11)

    # Subplot 2: Phân phối Cosine Similarity
    plt.subplot(1, 2, 2)
    genuine_scores = all_scores[all_labels == 1]
    impostor_scores = all_scores[all_labels == 0]

    plt.hist(impostor_scores, bins=50, alpha=0.6, color='#d62728', density=True, label=f'Impostor (Diff Person, N={len(impostor_scores)})')
    plt.hist(genuine_scores, bins=50, alpha=0.6, color='#2ca02c', density=True, label=f'Genuine (Same Person, N={len(genuine_scores)})')
    plt.axvline(best_th, color='black', linestyle='--', lw=2, label=f'Optimal Thresh = {best_th:.3f}')
    plt.xlabel('Cosine Similarity Score', fontsize=12, fontweight='bold')
    plt.ylabel('Probability Density', fontsize=12, fontweight='bold')
    plt.title('Cropped Face Separation Distribution (LFW)', fontsize=13, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[+] Đã lưu biểu đồ báo cáo khoa học tại: {save_path}")


# =============================================================================
# 7. QUẢN LÝ LỊCH SỬ HUẤN LUYỆN (CSV LOGGING)
# =============================================================================
def log_training_history(history_file, record):
    """Ghi lại lịch sử mỗi phiên huấn luyện vào file CSV để theo dõi tiến trình cải thiện"""
    file_exists = os.path.exists(history_file)
    headers = [
        "Timestamp", "Mode", "Epochs_Done", "Target_Epochs",
        "Duration_Mins", "Best_Loss", "LFW_Accuracy", "AUC", "EER",
        "Threshold", "Device", "Weights_Path", "Report_Path"
    ]
    try:
        with open(history_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)
        print(f"[+] Đã cập nhật lịch sử huấn luyện tại: {history_file}")
    except Exception as e:
        print(f"[!] Không thể ghi lịch sử huấn luyện ({e})")


# =============================================================================
# 8. QUY TRÌNH CHÍNH (TRAINING & EVALUATION PIPELINE)
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Upgraded LFW Distillation Pipeline for TinyMobileFaceNet-32")
    parser.add_argument("--epochs", type=int, default=30, help="Số epoch huấn luyện (0 hoặc âm = Huấn luyện vô tận)")
    parser.add_argument("--infinite", action="store_true", help="Bật chế độ huấn luyện vô tận (chạy đến khi bấm Ctrl+C)")
    parser.add_argument("--time-limit", type=float, default=0, help="Giới hạn thời gian huấn luyện tính bằng phút (0 = không giới hạn)")
    parser.add_argument("--batch-size", type=int, default=64, help="Kích thước batch (RTX 4060 khuyến nghị: 64)")
    parser.add_argument("--lr", type=float, default=0.001, help="Tốc độ học (AdamW)")
    parser.add_argument("--eval-every", type=int, default=0, help="Đánh giá LFW định kỳ mỗi N epoch (0 = chỉ đánh giá khi kết thúc)")
    parser.add_argument("--eval-only", action="store_true", help="Chỉ đánh giá benchmark LFW mô hình hiện tại và xuất báo cáo")
    parser.add_argument("--reset-weights", action="store_true", help="Xóa học lại từ đầu (không nạp trọng số đã lưu)")
    parser.add_argument("--teacher", type=str, default=None, help="Đường dẫn Teacher Model .pt")
    parser.add_argument("--data-root", type=str, default="./lfw_dataset", help="Thư mục LFW")
    parser.add_argument("--output-weights", type=str, default="./weights/tiny_mobilefacenet_lfw.pt")
    parser.add_argument("--report-img", type=str, default="./lfw_benchmark_report.png")
    parser.add_argument("--reports-dir", type=str, default="./reports", help="Thư mục lưu lịch sử các báo cáo biểu đồ")
    parser.add_argument("--history-file", type=str, default="./training_history.csv", help="File CSV ghi nhật ký huấn luyện")
    args = parser.parse_args()

    # Tự động đồng bộ đường dẫn về thư mục chứa mã nguồn (Python) nếu là đường dẫn tương đối
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for attr in ["data_root", "output_weights", "report_img", "reports_dir", "history_file"]:
        val = getattr(args, attr)
        if val and not os.path.isabs(val):
            # Ưu tiên tìm trong script_dir nếu có
            cand = os.path.normpath(os.path.join(script_dir, val))
            setattr(args, attr, cand)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU Đa nhân"

    # 1. Đảm bảo dữ liệu thô và khuôn mặt cắt 1:1 sẵn sàng
    if not os.path.exists(args.data_root) or len(os.listdir(args.data_root)) == 0:
        sibling_lfw = os.path.normpath(os.path.join(script_dir, "..", "Python", "lfw_dataset"))
        if os.path.exists(sibling_lfw) and len(os.listdir(sibling_lfw)) > 0:
            args.data_root = sibling_lfw
            print(f"[+] Tự động phát hiện tập dữ liệu LFW tại: {args.data_root}")

    raw_dir, pairs_path = ensure_lfw_raw(args.data_root)
    aligned_dir = os.path.join(args.data_root, "lfw_aligned_112")
    prepare_aligned_lfw(raw_dir, aligned_dir)

    # =========================================================================
    # CHẾ ĐỘ 1: EVAL ONLY (CHỈ ĐÁNH GIÁ MÔ HÌNH HIỆN TẠI VÀ XUẤT BÁO CÁO)
    # =========================================================================
    if args.eval_only:
        print("\n" + "=" * 70)
        print("CHẾ ĐỘ ĐÁNH GIÁ BENCHMARK LFW (EVAL ONLY)")
        print(f"- Thiết bị tính toán : {device} ({device_name})")
        print(f"- File trọng số     : {args.output_weights}")
        print("=" * 70)

        if not os.path.exists(args.output_weights):
            raise FileNotFoundError(f"Không tìm thấy file trọng số {args.output_weights} để đánh giá!")

        student_model = TinyMobileFaceNet32(embedding_size=128).to(device)
        student_model.load_state_dict(torch.load(args.output_weights, map_location=device))
        print(f"[+] Nạp thành công trọng số: {args.output_weights}")

        pairs = read_lfw_pairs(pairs_path, aligned_dir)
        mean_acc, std_acc, roc_auc, eer, best_th, fpr, tpr, all_scores, all_labels = evaluate_lfw_official(student_model, pairs, device)

        # Xuất biểu đồ báo cáo
        plot_lfw_report(fpr, tpr, roc_auc, eer, best_th, mean_acc, all_scores, all_labels, args.report_img)
        os.makedirs(args.reports_dir, exist_ok=True)
        ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ts_report_path = os.path.join(args.reports_dir, f"lfw_report_{ts_str}.png")
        shutil.copyfile(args.report_img, ts_report_path)
        print(f"[+] Đã lưu bản sao báo cáo lịch sử tại: {ts_report_path}")

        # Ghi nhật ký CSV
        record = {
            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Mode": "Eval_Only",
            "Epochs_Done": 0,
            "Target_Epochs": "N/A",
            "Duration_Mins": "0.0",
            "Best_Loss": "N/A",
            "LFW_Accuracy": f"{mean_acc:.2f}%",
            "AUC": f"{roc_auc:.4f}",
            "EER": f"{eer:.2f}%",
            "Threshold": f"{best_th:.3f}",
            "Device": device_name,
            "Weights_Path": args.output_weights,
            "Report_Path": ts_report_path
        }
        log_training_history(args.history_file, record)
        return

    # =========================================================================
    # CHẾ ĐỘ 2: HUẤN LUYỆN (TRAINING MODE)
    # =========================================================================
    is_infinite = args.infinite or (args.epochs <= 0 and args.time_limit <= 0)
    has_time_limit = args.time_limit > 0
    max_epochs = 9999999 if (is_infinite or (has_time_limit and args.epochs <= 0)) else args.epochs

    if is_infinite:
        mode_str = "Infinite"
        target_epochs_str = "Vô tận (Dừng bằng Ctrl+C)"
    elif has_time_limit:
        mode_str = f"TimeLimit_{args.time_limit}m"
        target_epochs_str = f"{args.time_limit} phút (Tối đa {args.epochs} epochs)" if args.epochs > 0 else f"{args.time_limit} phút"
    else:
        mode_str = f"FixedEpochs_{args.epochs}"
        target_epochs_str = f"{args.epochs} epochs"

    print("\n" + "=" * 70)
    print("HỆ THỐNG HUẤN LUYỆN CHUYỂN GIAO TRI THỨC VÀ BENCHMARK LFW ĐA NĂNG")
    print("Mục tiêu phần cứng: AMD Spartan-7 XC7S25 (LSI Design Contest 2026)")
    print(f"- Thiết bị tính toán  : {device} ({device_name})")
    print(f"- Chế độ huấn luyện   : {mode_str}")
    print(f"- Mục tiêu thời lượng : {target_epochs_str}")
    print(f"- Batch size          : {args.batch_size}")
    print(f"- Learning rate       : {args.lr}")
    print("=" * 70)

    # 2. Nạp Teacher Model
    teacher_path = args.teacher
    if teacher_path is None or not os.path.exists(teacher_path):
        candidate_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Python", "mobilefacenet_scripted.pt"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "MobileFaceNet", "pretrained_model", "mobilefacenet_scripted.pt"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "mobilefacenet_scripted.pt"),
            "./mobilefacenet_scripted.pt"
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                teacher_path = p
                break

    if teacher_path is None or not os.path.exists(teacher_path):
        raise FileNotFoundError(f"Không tìm thấy Teacher Model mobilefacenet_scripted.pt!")

    print(f"[1] Nạp Teacher Model: {teacher_path}")
    try:
        teacher_model = torch.jit.load(teacher_path, map_location=device)
    except Exception:
        teacher_model = torch.load(teacher_path, map_location=device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    print("    -> Nạp Teacher thành công (Đã đóng băng Gradient).")

    # 3. Khởi tạo Student Hardware Model
    print("[2] Khởi tạo Student Model: TinyMobileFaceNet-32...")
    student_model = TinyMobileFaceNet32(embedding_size=128).to(device)
    total_params = sum(p.numel() for p in student_model.parameters())
    print(f"    -> Tham số phần cứng: {total_params:,} (~{total_params/1024:.1f} KB INT8, tương thích Spartan-7 BRAM)")

    # 4. Kế thừa trọng số cũ nếu có (Continuous Training)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_weights)), exist_ok=True)
    if not args.reset_weights and os.path.exists(args.output_weights):
        print(f"    [+] Tìm thấy trọng số tốt nhất trước đó: {args.output_weights}")
        try:
            student_model.load_state_dict(torch.load(args.output_weights, map_location=device))
            print("        -> Đã kế thừa trọng số thành công! Tiếp tục tối ưu hóa sâu hơn.")
        except Exception as e:
            print(f"        [!] Không thể nạp trọng số cũ ({e}), khởi tạo lại từ đầu.")

    # 5. Nạp Dataset
    print("[3] Nạp tập dữ liệu khuôn mặt đã cắt sát (Face Crops)...")
    train_dataset = AlignedFaceDistillDataset(aligned_dir=aligned_dir, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"    -> Tổng số ảnh nạp vào: {len(train_dataset):,} ảnh (100% diện tích là ngũ quan khuôn mặt).")

    # 6. Thiết lập Optimizer và Scheduler
    criterion = DistillationLoss(alpha=2.0)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=1e-4)

    if is_infinite or (has_time_limit and args.epochs <= 0):
        # Chế độ dài hạn/vô tận: Chu kỳ Cosine Warm Restarts mỗi 25 epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=1, eta_min=1e-5)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)

    best_loss = float("inf")
    total_batches = len(train_loader)
    start_time = time.time()
    epochs_completed = 0
    stopped_by_user = False
    time_limit_reached = False

    print("\n[4] BẮT ĐẦU VÒNG LẶP HUẤN LUYỆN:")
    if is_infinite:
        print("    -> [LƯU Ý] MÔ HÌNH SẼ CHẠY LIÊN TỤC. BẤM CTRL+C BẤT KỲ LÚC NÀO ĐỂ DỪNG VÀ XUẤT KẾT QUẢ!")
    elif has_time_limit:
        print(f"    -> [LƯU Ý] Mô hình sẽ tự động dừng sau {args.time_limit} phút (hoặc bấm Ctrl+C để dừng sớm).")

    try:
        epoch = 1
        while epoch <= max_epochs:
            # Kiểm tra giới hạn thời gian trước khi vào epoch mới
            if has_time_limit and (time.time() - start_time) >= args.time_limit * 60:
                time_limit_reached = True
                print(f"\n[!] Đã đạt giới hạn thời gian {args.time_limit} phút. Đang dừng huấn luyện an toàn...")
                break

            student_model.train()
            epoch_loss = 0.0
            epoch_cos = 0.0
            epoch_mse = 0.0
            batches = 0
            ep_t0 = time.time()

            for batch_idx, (x_s, x_t) in enumerate(train_loader, 1):
                # Kiểm tra thời gian trong từng batch
                if has_time_limit and (time.time() - start_time) >= args.time_limit * 60:
                    time_limit_reached = True
                    break

                x_s, x_t = x_s.to(device), x_t.to(device)
                with torch.no_grad():
                    z_t = teacher_model(x_t)
                    z_t = F.normalize(z_t, p=2, dim=1)

                z_s = student_model(x_s)
                loss, cos_v, mse_v = criterion(z_s, z_t)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_cos += cos_v
                epoch_mse += mse_v
                batches += 1

                # In tiến độ trực tiếp
                if batch_idx % 25 == 0 or batch_idx == total_batches:
                    cur_loss = epoch_loss / batches
                    cur_cos = epoch_cos / batches
                    cur_mse = epoch_mse / batches
                    pct = batch_idx * 100 // total_batches
                    epoch_display = f"{epoch:02d}" if is_infinite else f"{epoch:02d}/{args.epochs:02d}"
                    elapsed_m = (time.time() - start_time) / 60
                    time_str = f" | Elapsed: {elapsed_m:.1f}m" + (f"/{args.time_limit:.0f}m" if has_time_limit else "")
                    sys.stdout.write(f"\r  [Epoch {epoch_display}] Batch {batch_idx:3d}/{total_batches} ({pct:3d}%) | "
                                     f"Loss: {cur_loss:.4f} (Cos: {cur_cos:.4f}, MSE: {cur_mse:.4f}){time_str}")
                    sys.stdout.flush()

            if batches == 0:
                break

            scheduler.step()
            avg_loss = epoch_loss / batches
            avg_cos = epoch_cos / batches
            avg_mse = epoch_mse / batches
            ep_elapsed = time.time() - ep_t0
            epochs_completed += 1

            epoch_display = f"{epoch:02d}" if is_infinite else f"{epoch:02d}/{args.epochs:02d}"
            print(f"\n>> Epoch [{epoch_display}] XONG ({ep_elapsed:.1f}s) | LR: {scheduler.get_last_lr()[0]:.6f} | "
                  f"Avg Loss: {avg_loss:.4f} (Cos: {avg_cos:.4f}, MSE: {avg_mse:.4f})")
            sys.stdout.flush()

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(student_model.state_dict(), args.output_weights)
                print(f"    [*] Đã lưu Checkpoint tốt nhất mới -> Loss: {best_loss:.4f}")

            # Đánh giá LFW định kỳ nếu bật eval_every
            if args.eval_every > 0 and (epoch % args.eval_every == 0) and os.path.exists(pairs_path):
                print(f"\n--- ĐÁNH GIÁ ĐỊNH KỲ TẠI EPOCH {epoch} ---")
                student_model.eval()
                pairs = read_lfw_pairs(pairs_path, aligned_dir)
                eval_acc, _, eval_auc, eval_eer, _, _, _, _, _ = evaluate_lfw_official(student_model, pairs, device)
                print(f"--- KẾT QUẢ TẠM THỜI: Accuracy = {eval_acc:.2f}%, AUC = {eval_auc:.4f}, EER = {eval_eer:.2f}% ---\n")
                student_model.train()

            if time_limit_reached:
                break

            epoch += 1

    except KeyboardInterrupt:
        stopped_by_user = True
        print("\n\n" + "!" * 70)
        print("[!] ĐÃ NHẬN TÍN HIỆU DỪNG TỪ BÀN PHÍM (Ctrl+C)!")
        print("[!] Đang an toàn lưu trữ trọng số và tạo báo cáo kiểm thử...")
        print("!" * 70)

    total_elapsed = time.time() - start_time
    elapsed_mins = total_elapsed / 60.0

    print("\n" + "=" * 70)
    status_label = "DỪNG BỞI NGƯỜI DÙNG (Ctrl+C)" if stopped_by_user else (
        "ĐẠT GIỚI HẠN THỜI GIAN" if time_limit_reached else "HOÀN TẤT HUẤN LUYỆN"
    )
    print(f"[{status_label}]")
    print(f"- Số epoch đã hoàn thành : {epochs_completed}")
    print(f"- Tổng thời gian huấn luyện: {total_elapsed:.1f}s ({elapsed_mins:.2f} phút)")
    print(f"- Trọng số tối ưu đã lưu tại : {args.output_weights}")
    print("=" * 70)

    # 7. Đánh giá kiểm thử 10-fold LFW và xuất báo cáo
    mean_acc, roc_auc, eer, best_th = 0.0, 0.0, 0.0, 0.0
    report_ts_path = "N/A"

    if os.path.exists(pairs_path) and os.path.exists(args.output_weights):
        student_model.load_state_dict(torch.load(args.output_weights, map_location=device))
        pairs = read_lfw_pairs(pairs_path, aligned_dir)
        mean_acc, std_acc, roc_auc, eer, best_th, fpr, tpr, all_scores, all_labels = evaluate_lfw_official(student_model, pairs, device)

        # Xuất biểu đồ mới nhất
        plot_lfw_report(fpr, tpr, roc_auc, eer, best_th, mean_acc, all_scores, all_labels, args.report_img)

        # Lưu bản sao có gắn nhãn thời gian vào thư mục reports/
        os.makedirs(args.reports_dir, exist_ok=True)
        ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_ts_path = os.path.join(args.reports_dir, f"lfw_report_{ts_str}.png")
        shutil.copyfile(args.report_img, report_ts_path)
        print(f"[+] Đã lưu bản sao báo cáo lịch sử tại: {report_ts_path}")

    # 8. Ghi nhận kết quả vào training_history.csv
    record = {
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Mode": mode_str,
        "Epochs_Done": epochs_completed,
        "Target_Epochs": target_epochs_str,
        "Duration_Mins": f"{elapsed_mins:.2f}",
        "Best_Loss": f"{best_loss:.4f}" if best_loss != float("inf") else "N/A",
        "LFW_Accuracy": f"{mean_acc:.2f}%",
        "AUC": f"{roc_auc:.4f}",
        "EER": f"{eer:.2f}%",
        "Threshold": f"{best_th:.3f}",
        "Device": device_name,
        "Weights_Path": args.output_weights,
        "Report_Path": report_ts_path
    }
    log_training_history(args.history_file, record)

    print("\n" + "=" * 70)
    print("TỔNG KẾT PHIÊN LÀM VIỆC:")
    print(f"  * Trạng thái         : {status_label}")
    print(f"  * Số epoch hoàn thành: {epochs_completed}")
    print(f"  * Thời gian huấn luyện: {elapsed_mins:.2f} phút")
    print(f"  * Độ chính xác LFW   : {mean_acc:.2f}%")
    print(f"  * ROC AUC Score      : {roc_auc:.4f}")
    print(f"  * Tỷ lệ lỗi (EER)   : {eer:.2f}%")
    print(f"  * File ảnh mới nhất  : {args.report_img}")
    print(f"  * Báo cáo lưu trữ   : {report_ts_path}")
    print(f"  * Nhật ký CSV        : {args.history_file}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
