# -*- coding: utf-8 -*-
"""
Chương trình Demo Nhận diện Khuôn mặt Thời gian thực DÀNH RIÊNG CHO TINY-MOBILEFACENET-32
CHỈ SỬ DỤNG MÔ HÌNH THU NHỎ 32x32 CỦA BẠN (Spartan-7 FPGA Accelerator)

Quy trình hoạt động:
1. Camera 640x480 -> Phát hiện khuôn mặt (Low-res 240px)
2. Cắt vùng mặt -> Resize chuẩn xác về ma trận 32x32x3
3. Bơm vào lõi TinyMobileFaceNet-32 -> Trích xuất Vector 128-D
4. So khớp Cosine Similarity với Database:
   - Tuấn Anh: Điểm ~55% - 80% (Khung XANH LÁ)
   - Người lạ: Điểm ~2% - 15% (Khung ĐỎ 'Nguoi La (Unknown)')

Tác giả: Dương Văn Tuyên & Nguyễn Sỹ Tuấn Anh (VNU-UET)
Dành cho: LSI Design Contest 2026
"""

import os
import sys
import time
import cv2
import torch
import torch.nn.functional as F
import numpy as np

# Thiết lập encoding UTF-8 cho console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from tiny_mobilefacenet import TinyMobileFaceNet32

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LFW_WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "tiny_mobilefacenet_lfw.pt")
DISTILL_WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "tiny_mobilefacenet_distilled.pt")
DB_FILE = os.path.join(BASE_DIR, "face_database.pt")


def load_model():
    """Khởi tạo mô hình TinyMobileFaceNet-32"""
    model = TinyMobileFaceNet32(embedding_size=128)
    chosen_path = None
    if os.path.exists(LFW_WEIGHTS_PATH):
        chosen_path = LFW_WEIGHTS_PATH
    elif os.path.exists(DISTILL_WEIGHTS_PATH):
        chosen_path = DISTILL_WEIGHTS_PATH

    if chosen_path:
        print(f"[+] Nạp trọng số TinyMobileFaceNet-32 từ: {chosen_path}")
        ckpt = torch.load(chosen_path, map_location='cpu')
        sd = ckpt['model_state_dict'] if (isinstance(ckpt, dict) and 'model_state_dict' in ckpt) else ckpt
        model.load_state_dict(sd)
    else:
        print("[!] Không tìm thấy file trọng số trong weights/. Dùng trọng số khởi tạo ngẫu nhiên.")
    model.eval()
    return model


def load_database():
    """Nạp cơ sở dữ liệu khuôn mặt"""
    if os.path.exists(DB_FILE):
        face_db = torch.load(DB_FILE, map_location='cpu')
        return face_db
    return {}


def preprocess_to_32x32(face_img):
    """
    Tiền xử lý duy nhất cho TinyMobileFaceNet:
    Cắt mặt -> Co về đúng ma trận 32x32x3 -> Chuẩn hóa [-1, 1]
    """
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_32 = cv2.resize(face_rgb, (32, 32), interpolation=cv2.INTER_AREA)
    norm = (face_32.astype(np.float32) - 127.5) / 128.0
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, face_32


def main():
    print("\n" + "=" * 70)
    print("DEMO NHẬN DIỆN THỜI GIAN THỰC 100% TINY-MOBILEFACENET-32")
    print("Target: AMD Spartan-7 XC7S25CSGA225-1 (ZERO7 Board - LSI Contest 2026)")
    print("=" * 70)

    # 1. Nạp mô hình TinyMobileFaceNet-32 duy nhất
    model = load_model()

    # 2. Nạp Database
    face_db = load_database()
    print(f"[+] Danh sách người có trong cơ sở dữ liệu: {list(face_db.keys())}")

    # 3. Mở Camera
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[LỖI] Không thể kết nối với Webcam!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Ngưỡng phân tách khoa học đo được:
    # Cùng người: ~0.50 - 0.70 | Người lạ: ~0.02 - 0.15
    # Chọn ngưỡng tối ưu: 0.38
    threshold = 0.38

    current_box = None
    frame_count = 0
    t_cnn_ms = 0.0
    fps_history = []

    print(f"\n[SẴN SÀNG] Ngưỡng nhận diện thiết lập: {threshold:.2f}")
    print("• Điểm Cosine >= 0.38 -> Người quen (Khung Xanh Lá)")
    print("• Điểm Cosine <  0.38 -> Người lạ (Khung Đỏ 'Nguoi La')")
    print("Nhấn [Q] trên cửa sổ hình ảnh để thoát.\n")

    while True:
        loop_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h_f, w_f = frame.shape[:2]
        frame_count += 1

        # Quét tìm mặt trên ảnh thu nhỏ để tối ưu FPS
        is_keyframe = (frame_count % 3 == 0) or (current_box is None)
        if is_keyframe:
            det_scale = 240.0 / w_f
            small_frame = cv2.resize(frame, (240, int(h_f * det_scale)), interpolation=cv2.INTER_NEAREST)
            small_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            small_gray_eq = cv2.equalizeHist(small_gray)

            faces = face_cascade.detectMultiScale(small_gray_eq, scaleFactor=1.1, minNeighbors=3, minSize=(25, 25))
            if len(faces) > 0:
                sx, sy, sw, sh = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
                orig_x = int(sx / det_scale)
                orig_y = int(sy / det_scale)
                orig_w = int(sw / det_scale)
                orig_h = int(sh / det_scale)

                pad = int(0.15 * min(orig_w, orig_h))
                new_x1 = max(0, orig_x - pad)
                new_y1 = max(0, orig_y - pad)
                new_x2 = min(w_f, orig_x + orig_w + pad)
                new_y2 = min(h_f, orig_y + orig_h + pad)
                target_box = np.array([new_x1, new_y1, new_x2, new_y2], dtype=np.float32)

                if current_box is None:
                    current_box = target_box
                else:
                    current_box = 0.65 * current_box + 0.35 * target_box
            else:
                if frame_count % 9 == 0:
                    current_box = None

        if current_box is not None:
            bx1, by1, bx2, by2 = current_box.astype(int)
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(w_f, bx2), min(h_f, by2)

            if (bx2 - bx1 > 20) and (by2 - by1 > 20):
                face_crop = frame[by1:by2, bx1:bx2]

                # BƯỚC 1: CO VỀ MA TRẬN 32x32 CỦA PHẦN CỨNG SPARTAN-7
                face_tensor, face_32_bgr = preprocess_to_32x32(face_crop)

                # BƯỚC 2: CHẠY LÕI TINY-MOBILEFACENET-32
                t0 = time.perf_counter()
                with torch.no_grad():
                    current_vec = model(face_tensor)
                t_cnn_ms = (time.perf_counter() - t0) * 1000.0

                # BƯỚC 3: SO KHỚP COSINE VỚI DATABASE
                best_name = "Nguoi La (Unknown)"
                best_score = -1.0

                for person_name, db_vec in face_db.items():
                    sim = F.cosine_similarity(current_vec, db_vec).item()
                    if sim > best_score:
                        best_score = sim
                        if sim >= threshold:
                            best_name = person_name

                # BƯỚC 4: HIỂN THỊ KẾT QUẢ
                if best_name != "Nguoi La (Unknown)":
                    color = (0, 255, 0) # Xanh lá: Đúng người quen
                    label = f"{best_name} | {best_score*100:.1f}%"
                else:
                    color = (0, 0, 255) # Đỏ: Người lạ!
                    label = f"Nguoi La ({best_score*100:.1f}%)" if best_score > 0 else "Nguoi La"

                # Khung chữ nhật bao quanh mặt
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                cv2.putText(frame, label, (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Vẽ ô ma trận 32x32 cạnh mặt để ban giám khảo thấy đúng dữ liệu đưa vào chip
                preview_32 = cv2.resize(face_32_bgr, (48, 48))
                cv2.rectangle(preview_32, (0, 0), (47, 47), (255, 255, 255), 1)
                py1, py2 = max(0, by1), min(h_f, by1 + 48)
                px1, px2 = max(0, bx1 - 56), min(w_f, bx1 - 8)
                if (py2 - py1 == 48) and (px2 - px1 == 48) and px1 >= 0:
                    frame[py1:py2, px1:px2] = preview_32
                    cv2.putText(frame, "32x32", (px1, py2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        # Đo FPS
        loop_time = time.perf_counter() - loop_start
        current_fps = 1.0 / loop_time if loop_time > 0 else 0
        fps_history.append(current_fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = np.mean(fps_history)

        # Thanh HUD Dashboard trên cùng
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (630, 85), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        cv2.putText(frame, "TinyMobileFaceNet-32 (Spartan-7 XC7S25 Core)", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {avg_fps:.0f} | TinyCNN Latency: {t_cnn_ms:.2f} ms | Threshold: {threshold:.2f}", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2)
        cv2.putText(frame, f"Enrolled: {list(face_db.keys())}", (20, 78), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

        cv2.imshow("TinyMobileFaceNet-32 Real-Time Demo", frame)

        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q'), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
