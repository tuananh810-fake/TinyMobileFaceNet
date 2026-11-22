# -*- coding: utf-8 -*-
"""
Chương trình Đánh giá Độ tin cậy (Reliability Evaluation & Benchmark)
cho mô hình phần cứng TinyMobileFaceNet-32 (Tương đương chuẩn LFW của MobileFaceNet)
Tác giả: Nguyễn Sỹ Tuấn Anh (VNU-UET)
Dành cho: LSI Design Contest 2026

Bao gồm:
1. Đọc tập cặp kiểm thử test_pairs.txt (Genuine & Impostor)
2. Trích xuất đặc trưng vector 128-D từ input 32x32x3
3. Tính toán Cosine Similarity và Phân phối Góc Theta (độ)
4. Tính các chỉ số khoa học: Accuracy, AUC, EER, FAR, FRR
5. Xuất báo cáo đồ thị 4 bảng chuẩn xuất bản (reliability_evaluation_report.png)
6. Xuất file angles.txt và evaluation_summary.json
"""

import os
import sys
import math
import time
import json
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns

# Thiết lập encoding UTF-8 cho console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from tiny_mobilefacenet import TinyMobileFaceNet32

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
PAIRS_FILE = os.path.join(DATASET_DIR, "test_pairs.txt")
DATA_OUT_DIR = os.path.join(BASE_DIR, "data")
ANGLES_FILE = os.path.join(DATA_OUT_DIR, "angles.txt")
SUMMARY_FILE = os.path.join(DATA_OUT_DIR, "evaluation_summary.json")
REPORT_IMAGE = os.path.join(BASE_DIR, "reliability_evaluation_report.png")
DEFAULT_WEIGHTS = os.path.join(BASE_DIR, "weights", "tiny_mobilefacenet_lfw.pt")


def preprocess_image_32(img_path, flip=False):
    """Đọc và chuẩn hóa ảnh 32x32 về định dạng tensor [-1, 1]"""
    if not os.path.isabs(img_path):
        img_path = os.path.join(BASE_DIR, img_path)

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Không thể đọc ảnh tại: {img_path}")

    if flip:
        img = cv2.flip(img, 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img_rgb.shape[:2] != (32, 32):
        img_rgb = cv2.resize(img_rgb, (32, 32), interpolation=cv2.INTER_AREA)

    tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float()
    tensor = (tensor - 127.5) / 128.0
    return tensor.unsqueeze(0)


def extract_feature(model, img_path, device, use_flip_enhancement=True):
    """
    Trích xuất vector 128 chiều kèm kỹ thuật Flip Enhancement
    f = (f_orig + f_flip) / ||f_orig + f_flip|| (tương tự như lfw_eval.py)
    """
    t1 = preprocess_image_32(img_path, flip=False).to(device)
    with torch.no_grad():
        v1 = model(t1)
        if use_flip_enhancement:
            t2 = preprocess_image_32(img_path, flip=True).to(device)
            v2 = model(t2)
            v = v1 + v2
            v = F.normalize(v, p=2, dim=1)
        else:
            v = v1
    return v.cpu().squeeze(0).numpy()


def run_benchmark(model_path=None, pairs_file=PAIRS_FILE, use_flip=True):
    """Thực thi kiểm thử toàn bộ cặp dữ liệu"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 70)
    print("BẮT ĐẦU ĐÁNH GIÁ ĐỘ TIN CẬY TINY-MOBILEFACENET-32")
    print(f"- Thiết bị tính toán: {device}")
    print(f"- File cặp kiểm thử : {pairs_file}")
    print("=" * 70)

    # 1. Khởi tạo mô hình
    model = TinyMobileFaceNet32(embedding_size=128).to(device)
    if model_path and os.path.exists(model_path):
        print(f"[+] Nạp trọng số từ: {model_path}")
        ckpt = torch.load(model_path, map_location=device)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict)
    else:
        print("[!] Không tìm thấy checkpoint tùy chỉnh. Sử dụng trọng số khởi tạo ban đầu.")

    model.eval()

    # 2. Đọc file cặp kiểm thử
    if not os.path.exists(pairs_file):
        raise FileNotFoundError(f"Không tìm thấy file cặp kiểm thử tại: {pairs_file}. "
                                f"Hãy chạy: python dataset_manager.py --generate-pairs")

    with open(pairs_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"[+] Tổng số cặp kiểm thử nạp được: {len(lines)}")

    os.makedirs(DATA_OUT_DIR, exist_ok=True)
    cosine_scores = []
    angle_scores = []
    labels = []
    angles_txt_lines = []

    total_time = 0.0

    print("[+] Đang trích xuất đặc trưng và tính toán tương đồng từng cặp...")
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        p1, p2, label_str = parts[0], parts[1], parts[2]
        label = int(label_str)

        start_t = time.time()
        f1 = extract_feature(model, p1, device, use_flip)
        f2 = extract_feature(model, p2, device, use_flip)
        total_time += (time.time() - start_t)

        # Tính Cosine Similarity
        cosine = float(np.dot(f1, f2))
        cosine_clipped = max(-1.0, min(1.0, cosine))

        # Tính Góc Theta (độ) theo chuẩn MobileFaceNet: theta = arccos(cosine) * 180 / pi
        theta = math.acos(cosine_clipped) * 180.0 / math.pi

        cosine_scores.append(cosine_clipped)
        angle_scores.append(theta)
        labels.append(label)
        angles_txt_lines.append(f"{theta:.4f} {label}\n")

    # Lưu angles.txt giống hệt MobileFaceNet
    with open(ANGLES_FILE, "w", encoding="utf-8") as f:
        f.writelines(angles_txt_lines)
    print(f"[+] Đã xuất file số liệu góc: {ANGLES_FILE}")

    cosine_scores = np.array(cosine_scores)
    angle_scores = np.array(angle_scores)
    labels = np.array(labels)

    # 3. Tính toán các chỉ số thống kê & độ tin cậy
    avg_latency_ms = (total_time / (len(lines) * 2)) * 1000.0

    # Quét tìm ngưỡng tối ưu cho Cosine và Góc Theta
    thresholds = np.linspace(0, 180, 1801) # Quét bước 0.1 độ
    best_acc = 0.0
    best_theta_th = 0.0

    ones_angles = angle_scores[labels == 1]
    zeros_angles = angle_scores[labels == 0]

    for th in thresholds:
        # Cùng người (label=1) thì góc phải nhỏ hơn hoặc bằng ngưỡng
        # Khác người (label=0) thì góc phải lớn hơn ngưỡng
        pred = (angle_scores <= th).astype(int)
        acc = np.mean(pred == labels)
        if acc > best_acc:
            best_acc = acc
            best_theta_th = th

    best_cosine_th = math.cos(best_theta_th * math.pi / 180.0)

    # Tính ROC và AUC (Dùng Cosine score)
    fpr, tpr, roc_thresholds = roc_curve(labels, cosine_scores)
    roc_auc = auc(fpr, tpr)

    # Tính EER (Equal Error Rate: Điểm mà FAR == FRR)
    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    eer_cosine_th = roc_thresholds[eer_idx]
    eer_theta_th = math.acos(max(-1.0, min(1.0, eer_cosine_th))) * 180.0 / math.pi

    # Dự đoán tại ngưỡng tối ưu
    y_pred_best = (angle_scores <= best_theta_th).astype(int)
    cm = confusion_matrix(labels, y_pred_best)

    # Thống kê phân phối góc Gaussian
    mu_0 = float(np.mean(zeros_angles)) if len(zeros_angles) > 0 else 90.0
    sigma_0 = float(np.std(zeros_angles)) if len(zeros_angles) > 0 else 10.0
    mu_1 = float(np.mean(ones_angles)) if len(ones_angles) > 0 else 40.0
    sigma_1 = float(np.std(ones_angles)) if len(ones_angles) > 0 else 10.0

    # In kết quả dạng bảng
    print("\n" + "=" * 70)
    print("KẾT QUẢ ĐO ĐẠC ĐỘ TIN CẬY (RELIABILITY BENCHMARK REPORT)")
    print("=" * 70)
    print(f"| Chỉ số kỹ thuật                     | Giá trị đạt được         |")
    print(f"|-------------------------------------|--------------------------|")
    print(f"| Tổng số cặp kiểm thử                | {len(labels):<24} |")
    print(f"| Số cặp Cùng người (Genuine)         | {len(ones_angles):<24} |")
    print(f"| Số cặp Khác người (Impostor)        | {len(zeros_angles):<24} |")
    print(f"| Độ chính xác cực đại (Accuracy)     | {best_acc * 100:.2f}%{'':<18} |")
    print(f"| Ngưỡng tối ưu Theta (deg)           | {best_theta_th:.2f} deg{'':<16} |")
    print(f"| Ngưỡng tối ưu Cosine                | {best_cosine_th:.4f}{'':<19} |")
    print(f"| Diện tích dưới đường ROC (AUC)      | {roc_auc:.4f}{'':<19} |")
    print(f"| Tỷ lệ lỗi cân bằng (EER)            | {eer * 100:.2f}%{'':<18} |")
    print(f"| Ngưỡng tại điểm EER                 | {eer_theta_th:.2f} deg{'':<16} |")
    print(f"| Phân phối Genuine (mu_1, sigma_1)   | ({mu_1:.2f}, {sigma_1:.2f}){'':<12} |")
    print(f"| Phân phối Impostor (mu_0, sigma_0)  | ({mu_0:.2f}, {sigma_0:.2f}){'':<12} |")
    print(f"| Thời gian suy luận trung bình/khuôn | {avg_latency_ms:.2f} ms{'':<17} |")
    print("=" * 70)

    # Lưu tóm tắt JSON
    summary_data = {
        "total_pairs": len(labels),
        "genuine_pairs": len(ones_angles),
        "impostor_pairs": len(zeros_angles),
        "best_accuracy_percent": round(best_acc * 100, 2),
        "optimal_theta_threshold_deg": round(best_theta_th, 2),
        "optimal_cosine_threshold": round(best_cosine_th, 4),
        "roc_auc": round(roc_auc, 4),
        "equal_error_rate_percent": round(eer * 100, 2),
        "eer_theta_threshold_deg": round(eer_theta_th, 2),
        "genuine_distribution": {"mean": round(mu_1, 2), "std": round(sigma_1, 2)},
        "impostor_distribution": {"mean": round(mu_0, 2), "std": round(sigma_0, 2)},
        "average_latency_ms": round(avg_latency_ms, 2)
    }
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)
    print(f"[+] Đã lưu bản tóm tắt chỉ số vào: {SUMMARY_FILE}")

    # 4. Vẽ 4 biểu đồ báo cáo khoa học
    draw_reliability_report(
        angle_scores, labels, ones_angles, zeros_angles,
        fpr, tpr, roc_auc, best_theta_th, eer, eer_theta_th,
        mu_0, sigma_0, mu_1, sigma_1, cm, best_acc
    )


def draw_reliability_report(angle_scores, labels, ones_angles, zeros_angles,
                            fpr, tpr, roc_auc, best_theta_th, eer, eer_theta_th,
                            mu_0, sigma_0, mu_1, sigma_1, cm, best_acc):
    """Vẽ và xuất ảnh 4 biểu đồ báo cáo khoa học đạt chuẩn bài báo quốc tế"""
    print(f"[+] Đang vẽ và xuất 4 biểu đồ báo cáo khoa học -> {REPORT_IMAGE}...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # --- BIỂU ĐỒ 1: HISTOGRAM PHÂN PHỐI GÓC THETA VÀ GAUSSIAN FIT (GIỐNG BÀI BÁO GỐC) ---
    ax1 = axes[0, 0]
    bins = np.linspace(0, 180, 91)
    if len(zeros_angles) > 0:
        ax1.hist(zeros_angles, bins=bins, density=True, alpha=0.5, color='red', label='Khác người (Impostor - 0)')
        y_0 = scipy.stats.norm.pdf(bins, mu_0, sigma_0)
        ax1.plot(bins, y_0, 'r--', lw=2, label=f'Fit Impostor (mu={mu_0:.1f}, sig={sigma_0:.1f})')

    if len(ones_angles) > 0:
        ax1.hist(ones_angles, bins=bins, density=True, alpha=0.5, color='blue', label='Cùng người (Genuine - 1)')
        y_1 = scipy.stats.norm.pdf(bins, mu_1, sigma_1)
        ax1.plot(bins, y_1, 'b--', lw=2, label=f'Fit Genuine (mu={mu_1:.1f}, sig={sigma_1:.1f})')

    ax1.axvline(best_theta_th, color='green', linestyle='-', linewidth=2.5, label=f'Ngưỡng tối ưu: {best_theta_th:.1f}°')
    ax1.set_xlabel('Góc Theta (Độ)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Mật độ xác suất (Density)', fontsize=11, fontweight='bold')
    ax1.set_title('1. PHÂN PHỐI GÓC THETA (HISTOGRAM & GAUSSIAN FIT)', fontsize=12, fontweight='bold', color='#1a365d')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim([0, 180])

    # --- BIỂU ĐỒ 2: ĐƯỜNG CONG ROC VÀ AUC ---
    ax2 = axes[0, 1]
    ax2.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Ngẫu nhiên (Chance)')
    # Điểm EER
    ax2.plot([eer], [1.0 - eer], 'ro', markersize=8, label=f'Điểm EER: {eer*100:.2f}%')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Tỷ lệ nhận nhầm - FAR (False Positive Rate)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Tỷ lệ nhận đúng - TAR (True Positive Rate)', fontsize=11, fontweight='bold')
    ax2.set_title('2. ĐƯỜNG CONG ĐỘ TIN CẬY ROC (LSI BENCHMARK)', fontsize=12, fontweight='bold', color='#1a365d')
    ax2.legend(loc="lower right", fontsize=10)

    # --- BIỂU ĐỒ 3: ĐỒ THỊ FAR VÀ FRR THEO NGƯỠNG THETA (TÌM EER TRỰC QUAN) ---
    ax3 = axes[1, 0]
    scan_th = np.linspace(10, 140, 131)
    far_list = []
    frr_list = []
    for th in scan_th:
        # FAR: Khác người mà góc <= th (nhận nhầm)
        far = np.mean(zeros_angles <= th) if len(zeros_angles) > 0 else 0
        # FRR: Cùng người mà góc > th (từ chối nhầm)
        frr = np.mean(ones_angles > th) if len(ones_angles) > 0 else 0
        far_list.append(far * 100)
        frr_list.append(frr * 100)

    ax3.plot(scan_th, far_list, color='red', lw=2, label='FAR (%) - Nhận nhầm')
    ax3.plot(scan_th, frr_list, color='blue', lw=2, label='FRR (%) - Từ chối nhầm')
    ax3.axvline(eer_theta_th, color='purple', linestyle='dashed', lw=2, label=f'EER = {eer*100:.2f}% (tại {eer_theta_th:.1f}°)')
    ax3.set_xlabel('Ngưỡng góc Theta (Độ)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Tỷ lệ sai số (%)', fontsize=11, fontweight='bold')
    ax3.set_title('3. ĐỒ THỊ CÂN BẰNG BẢO MẬT FAR & FRR', fontsize=12, fontweight='bold', color='#1a365d')
    ax3.legend(loc="upper center", fontsize=10)
    ax3.set_ylim([-2, 102])

    # --- BIỂU ĐỒ 4: MA TRẬN NHẦM LẪN (CONFUSION MATRIX) TẠI NGƯỠNG TỐI ƯU ---
    ax4 = axes[1, 1]
    cm_norm = cm.astype('float') / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1) * 100
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues', ax=ax4, cbar=False,
                xticklabels=['Khác người', 'Cùng người'],
                yticklabels=['Khác người', 'Cùng người'], annot_kws={"size": 13, "weight": "bold"})
    ax4.set_xlabel('Dự đoán của TinyMobileFaceNet (%)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Thực tế (Ground Truth) (%)', fontsize=11, fontweight='bold')
    ax4.set_title(f'4. MA TRẬN XÁC THỰC (ACCURACY = {best_acc*100:.1f}%)', fontsize=12, fontweight='bold', color='#1a365d')

    plt.tight_layout()
    plt.savefig(REPORT_IMAGE, dpi=300)
    plt.close()
    print(f"\n[XUẤT SẮC!] Đã lưu file ảnh báo cáo độ tin cậy tại:\n-> {os.path.abspath(REPORT_IMAGE)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá độ tin cậy mô hình TinyMobileFaceNet-32")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS, help="Đường dẫn file trọng số .pt")
    parser.add_argument("--pairs", type=str, default=PAIRS_FILE, help="Đường dẫn file test_pairs.txt")
    parser.add_argument("--no-flip", action="store_true", help="Không sử dụng kỹ thuật lật ảnh tăng cường")

    args = parser.parse_args()

    # Kiểm tra nếu chưa có cặp kiểm thử thì sinh dữ liệu mẫu trước
    if not os.path.exists(args.pairs):
        print("[!] Không tìm thấy file cặp kiểm thử. Đang tự động gọi dataset_manager để tạo tập dữ liệu mẫu...")
        from dataset_manager import create_sample_dataset
        create_sample_dataset()

    weights_to_use = args.weights if os.path.exists(args.weights) else None
    run_benchmark(model_path=weights_to_use, pairs_file=args.pairs, use_flip=not args.no_flip)
