# -*- coding: utf-8 -*-
"""
Hệ thống Nhận diện Khuôn mặt Thời gian thực: Chạy trên Camera hoặc Video file
Tác giả: Dương Văn Tuyên & Nguyễn Sỹ Tuấn Anh
"""
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import time
import os
from tiny_mobilefacenet import TinyMobileFaceNet32

def run_face_system(source=0, db_file="face_database.pt", threshold=0.65):
    """
    source: 0 (webcam) hoặc đường dẫn file video, ví dụ "video_test.mp4"
    """
    # 1. Tải mô hình
    model = TinyMobileFaceNet32(embedding_size=128)
    weights_candidates = [
        os.path.join(os.path.dirname(__file__), "weights", "tiny_mobilefacenet_lfw.pt"),
        os.path.join(os.path.dirname(__file__), "weights", "tiny_mobilefacenet_distilled.pt")
    ]
    for w in weights_candidates:
        if os.path.exists(w):
            print(f"[INFO] Nạp trọng số mô hình từ: {w}")
            ckpt = torch.load(w, map_location='cpu')
            sd = ckpt['model_state_dict'] if (isinstance(ckpt, dict) and 'model_state_dict' in ckpt) else ckpt
            model.load_state_dict(sd)
            break
    model.eval()

    # 2. Tải database
    if not os.path.exists(db_file):
        print(f"[CẢNH BÁO] Chưa có file '{db_file}'! Hãy chạy build_database.py trước.")
        face_db = {}
    else:
        face_db = torch.load(db_file)
        print(f"[INFO] Đã tải Database gồm {len(face_db)} người: {list(face_db.keys())}")

    # 3. Mở nguồn Video hoặc Camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[LỖI] Không thể mở nguồn: {source}")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print("\n[ĐANG CHẠY...] Bấm [Q] trên cửa sổ video để THOÁT.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Đã kết thúc video.")
            break

        # Nếu là webcam (source=0) thì lật gương, nếu là video file thì không cần
        if source == 0:
            frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            
            # Preprocess về 32x32
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (32, 32))
            face_norm = (face_resized.astype(np.float32) - 127.5) / 128.0
            face_tensor = torch.tensor(face_norm).permute(2, 0, 1).unsqueeze(0)

            # ĐO TỐC ĐỘ SUY LUẬN
            t0 = time.perf_counter()
            with torch.no_grad():
                current_vec = model(face_tensor)
            t1 = time.perf_counter()
            latency = (t1 - t0) * 1000
            fps = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0

            # SO KHỚP VỚI CƠ SỞ DỮ LIỆU (1:N Matching)
            best_name = "Unknown"
            best_score = -1.0

            for name, db_vec in face_db.items():
                score = F.cosine_similarity(current_vec, db_vec).item()
                if score > best_score:
                    best_score = score
                    if score >= threshold:
                        best_name = name

            # Hiển thị
            color = (0, 255, 0) if best_name != "Unknown" else (0, 0, 255)
            label = f"{best_name} ({best_score*100:.1f}%)"

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"CNN: {latency:.2f}ms ({fps:.0f} FPS)", (x, y+h+20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.imshow("Face Recognition Demo (LSI Contest)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # ĐỂ CHẠY WEBCAM: Dùng source = 0
    # ĐỂ CHẠY VIDEO FILE: Thay bằng đường dẫn file video, ví dụ source = "video_test.mp4"
    run_face_system(source=0, threshold=0.65)