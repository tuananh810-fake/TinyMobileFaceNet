# -*- coding: utf-8 -*-
"""
Công cụ Quản lý và Tiền xử lý Dữ liệu Khuôn mặt cho TinyMobileFaceNet-32
Tác giả: Nguyễn Sỹ Tuấn Anh (VNU-UET)
Dành cho: LSI Design Contest 2026

Các chức năng chính:
1. Thu thập ảnh khuôn mặt từ Webcam (--collect --name <tên>)
2. Cắt và chuẩn hóa dữ liệu khuôn mặt về 32x32 (--process)
3. Tự động sinh danh sách cặp kiểm thử Genuine / Impostor (--generate-pairs)
4. Tự tạo dữ liệu mẫu để chạy thử nghiệm benchmark tức thì (--create-sample)
"""

import os
import sys
import cv2
import random
import argparse
import numpy as np

# Thiết lập encoding UTF-8 cho console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RAW_DIR = os.path.join(DATASET_DIR, "raw")
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed_32x32")
PAIRS_FILE = os.path.join(DATASET_DIR, "test_pairs.txt")


def get_face_detector():
    """Khởi tạo Haar Cascade Face Detector từ OpenCV"""
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    detector = cv2.CascadeClassifier(cascade_path)
    return detector


def collect_from_webcam(person_name, max_samples=30):
    """
    Thu thập ảnh khuôn mặt người dùng từ Webcam và tự động lưu vào dataset/raw/<tên>
    """
    person_dir = os.path.join(RAW_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)
    
    detector = get_face_detector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print(f"[LỖI] Không thể kết nối với Webcam. Hãy kiểm tra lại camera!")
        return

    print("=" * 65)
    print(f"BẮT ĐẦU THU THẬP DỮ LIỆU CHO: {person_name}")
    print(f"- Nhấn [SPACE] để chụp ảnh (khuyên dùng: 15-30 ảnh các góc)")
    print(f"- Thay đổi biểu cảm: thẳng, hơi nghiêng trái/phải, cười, nghiêm túc")
    print(f"- Nhấn [Q] hoặc [ESC] để kết thúc thu thập")
    print("=" * 65)

    count = 0
    while count < max_samples:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Lật ảnh gương cho người dùng dễ nhìn
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        h_f, w_f = frame.shape[:2]

        # Tiền xử lý tăng cường độ tương phản giúp bắt mặt nhạy gấp 3 lần
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        faces = detector.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))

        target_face_crop = None
        if len(faces) > 0:
            # Sắp xếp lấy khuôn mặt to nhất trong khung hình
            x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
            # Thêm lề (padding) 15% để lấy trọn vẹn khuôn mặt
            pad_h = int(0.15 * h)
            pad_w = int(0.15 * w)
            y1 = max(0, y - pad_h)
            y2 = min(frame.shape[0], y + h + pad_h)
            x1 = max(0, x - pad_w)
            x2 = min(frame.shape[1], x + w + pad_w)
            target_face_crop = frame[y1:y2, x1:x2]

            # Vẽ khung xanh lá báo hiệu đã bắt được mặt
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, f"DA BAT MAT ({w}x{h} px)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        else:
            # Khung dẫn hướng ở trung tâm màn hình nếu Haar cascade trượt
            cw, ch = int(w_f * 0.4), int(h_f * 0.55)
            cx1, cy1 = (w_f - cw) // 2, (h_f - ch) // 2
            cx2, cy2 = cx1 + cw, cy1 + ch
            target_face_crop = frame[cy1:cy2, cx1:cx2]
            cv2.rectangle(display_frame, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)
            cv2.putText(display_frame, "Dua mat vao khung vang & bam SPACE", (cx1, cy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        info_text = f"Nguoi: {person_name} | Da chup: {count}/{max_samples} [SPACE: Chup, Q: Thoat]"
        cv2.putText(display_frame, info_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Thu thap du lieu khuon mat TinyMobileFaceNet", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 32: # Phím Space
            if target_face_crop is not None and target_face_crop.shape[0] > 10 and target_face_crop.shape[1] > 10:
                count += 1
                save_path = os.path.join(person_dir, f"{person_name}_{count:03d}.jpg")
                cv2.imwrite(save_path, target_face_crop)
                print(f"[OK] Đã lưu ảnh {count}/{max_samples} (Kích thước gốc: {target_face_crop.shape[1]}x{target_face_crop.shape[0]}): {save_path}")
            else:
                print("[!] Vui lòng nhìn vào khung hình để chụp!")
        elif key in [ord('q'), 27]: # Phím Q hoặc ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[HOÀN TẤT] Đã lưu tổng cộng {count} ảnh cho danh tính '{person_name}'.")


def process_raw_dataset(target_size=(32, 32)):
    """
    Quét toàn bộ thư mục dataset/raw/, cắt khuôn mặt và resize về chuẩn (32, 32)
    Lưu vào dataset/processed_32x32/
    """
    if not os.path.exists(RAW_DIR):
        print(f"[!] Thư mục '{RAW_DIR}' chưa tồn tại. Hãy thêm ảnh raw trước.")
        return

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    detector = get_face_detector()
    
    total_processed = 0
    print("\n" + "=" * 65)
    print(f"TIỀN XỬ LÝ KHUÔN MẶT SANG ĐẦU VÀO CHUẨN {target_size[0]}x{target_size[1]}")
    print("=" * 65)

    identities = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]
    if not identities:
        print("[!] Không tìm thấy thư mục người dùng nào trong dataset/raw/!")
        return

    for person_name in identities:
        src_person_dir = os.path.join(RAW_DIR, person_name)
        dst_person_dir = os.path.join(PROCESSED_DIR, person_name)
        os.makedirs(dst_person_dir, exist_ok=True)

        img_files = [f for f in os.listdir(src_person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        saved_count = 0

        for img_file in img_files:
            img_path = os.path.join(src_person_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            h_img, w_img = img.shape[:2]
            aspect_ratio = min(h_img, w_img) / max(h_img, w_img)

            # Trường hợp 1: Ảnh đã là ảnh khuôn mặt được cắt sẵn (từ webcam collect hoặc dataset chuẩn)
            # Dấu hiệu: Tỷ lệ gần vuông (aspect_ratio > 0.7)
            if aspect_ratio > 0.7:
                face_crop = img
            else:
                # Trường hợp 2: Ảnh chụp toàn thân/chưa cắt, cần dùng detector tìm vị trí mặt
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_eq = cv2.equalizeHist(gray)
                faces = detector.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

                if len(faces) > 0:
                    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
                    # Thêm lề 10%
                    pad = int(0.1 * min(w, h))
                    y1 = max(0, y - pad)
                    y2 = min(h_img, y + h + pad)
                    x1 = max(0, x - pad)
                    x2 = min(w_img, x + w + pad)
                    face_crop = img[y1:y2, x1:x2]
                else:
                    # Nếu ảnh lớn mà không phát hiện được mặt, crop vùng giữa thay vì bóp méo cả ảnh
                    cy, cx = h_img // 2, w_img // 2
                    crop_sz = min(h_img, w_img) // 2
                    face_crop = img[max(0, cy - crop_sz):min(h_img, cy + crop_sz),
                                    max(0, cx - crop_sz):min(w_img, cx + crop_sz)]
                    print(f"  [Chú ý] '{img_file}' không bắt được mặt tự động -> Tự động cắt vùng trung tâm.")

            # Co khuôn mặt (dù to cỡ nào) về đúng kích thước chuẩn 32x32 của phần cứng
            face_32 = cv2.resize(face_crop, target_size, interpolation=cv2.INTER_AREA)
            
            dst_path = os.path.join(dst_person_dir, f"crop_{img_file}")
            cv2.imwrite(dst_path, face_32)
            saved_count += 1
            total_processed += 1

        print(f"- [{person_name}]: Đã xử lý {saved_count}/{len(img_files)} ảnh -> {dst_person_dir}")

    print(f"[XONG] Tổng số ảnh chuẩn {target_size[0]}x{target_size[1]} được tạo: {total_processed}")


def generate_test_pairs(num_genuine_per_id=15, num_impostor_pairs=100):
    """
    Sinh file test_pairs.txt theo chuẩn LFW:
    <đường_dẫn_ảnh_1> <đường_dẫn_ảnh_2> <nhãn: 1=Cùng người, 0=Khác người>
    """
    if not os.path.exists(PROCESSED_DIR):
        print(f"[!] Thư mục '{PROCESSED_DIR}' chưa tồn tại. Hãy chạy --process trước.")
        return

    identities = [d for d in os.listdir(PROCESSED_DIR) if os.path.isdir(os.path.join(PROCESSED_DIR, d))]
    if len(identities) < 2:
        print("[!] Cần ít nhất 2 người khác nhau trong dataset để tạo cặp kiểm thử!")
        return

    person_images = {}
    for person in identities:
        p_dir = os.path.join(PROCESSED_DIR, person)
        imgs = [os.path.join(p_dir, f) for f in os.listdir(p_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(imgs) >= 2:
            person_images[person] = imgs

    valid_persons = list(person_images.keys())
    if len(valid_persons) < 2:
        print("[!] Cần ít nhất 2 người, mỗi người có từ 2 ảnh trở lên!")
        return

    pairs = []

    # 1. Sinh các cặp CÙNG NGƯỜI (Genuine Pairs - Nhãn 1)
    for person, imgs in person_images.items():
        n_imgs = len(imgs)
        # Lấy tối đa số cặp tổ hợp có thể
        possible_pairs = []
        for i in range(n_imgs):
            for j in range(i + 1, n_imgs):
                possible_pairs.append((imgs[i], imgs[j], 1))
        
        sample_k = min(len(possible_pairs), num_genuine_per_id)
        selected_genuine = random.sample(possible_pairs, sample_k)
        pairs.extend(selected_genuine)

    num_genuine_total = len(pairs)

    # 2. Sinh các cặp KHÁC NGƯỜI (Impostor Pairs - Nhãn 0) cân bằng với số cặp Genuine
    target_impostor = max(num_impostor_pairs, num_genuine_total)
    impostor_pairs = set()

    attempts = 0
    while len(impostor_pairs) < target_impostor and attempts < target_impostor * 10:
        attempts += 1
        p1, p2 = random.sample(valid_persons, 2)
        img1 = random.choice(person_images[p1])
        img2 = random.choice(person_images[p2])
        pair = (img1, img2, 0)
        impostor_pairs.add(pair)

    pairs.extend(list(impostor_pairs))
    random.shuffle(pairs)

    # Ghi file test_pairs.txt
    with open(PAIRS_FILE, "w", encoding="utf-8") as f:
        for p1, p2, label in pairs:
            # Sử dụng đường dẫn tương đối để dễ chuyển đổi qua lại giữa các máy
            rel_p1 = os.path.relpath(p1, BASE_DIR).replace("\\", "/")
            rel_p2 = os.path.relpath(p2, BASE_DIR).replace("\\", "/")
            f.write(f"{rel_p1} {rel_p2} {label}\n")

    print("\n" + "=" * 65)
    print(f"ĐÃ TẠO TẬP CẶP KIỂM THỬ: {PAIRS_FILE}")
    print(f"- Số cặp CÙNG một người (Genuine): {num_genuine_total}")
    print(f"- Số cặp KHÁC người (Impostor)  : {len(impostor_pairs)}")
    print(f"- Tổng số cặp kiểm thử          : {len(pairs)}")
    print("=" * 65)


def create_sample_dataset(num_identities=5, images_per_id=10):
    """
    Tự động tạo tập dữ liệu mẫu chuẩn (kèm nhiễu ánh sáng, mờ, góc nghiêng)
    để người dùng có thể chạy thử nghiệm đo đạc ngay lập tức mà chưa cần camera.
    """
    print(f"\n[+] Đang tự động tạo tập dữ liệu mẫu mô phỏng ({num_identities} người, {images_per_id} ảnh/người)...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    np.random.seed(42)
    names = ["Tuyen_Duong", "Tuan_Anh", "Minh_Hoang", "Lan_Huong", "Van_Duc"]
    
    for i in range(min(num_identities, len(names))):
        name = names[i]
        person_dir = os.path.join(PROCESSED_DIR, name)
        os.makedirs(person_dir, exist_ok=True)
        
        # Tạo vector cấu trúc khuôn mặt cơ sở cho danh tính này
        base_canvas = np.zeros((32, 32, 3), dtype=np.uint8)
        # Tạo hình elip đại diện cho khuôn mặt
        cv2.ellipse(base_canvas, (16, 16), (10, 13), 0, 0, 360, (180 + i*15, 140 + i*10, 120 + i*10), -1)
        # Vẽ 2 mắt
        cv2.circle(base_canvas, (12, 12), 2, (40, 30, 20), -1)
        cv2.circle(base_canvas, (20, 12), 2, (40, 30, 20), -1)
        # Vẽ miệng
        cv2.ellipse(base_canvas, (16, 23), (4, 2), 0, 0, 180, (50, 40, 160), -1)

        for img_idx in range(images_per_id):
            img = base_canvas.copy()
            # Thêm biến thể thực tế:
            # 1. Ánh sáng / tương phản
            alpha = np.random.uniform(0.7, 1.3)
            beta = np.random.randint(-20, 20)
            img = np.clip(img * alpha + beta, 0, 255).astype(np.uint8)
            
            # 2. Nhiễu hạt (sensor noise)
            noise = np.random.normal(0, 5, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # 3. Dịch chuyển nhẹ mô phỏng góc mặt
            dx = np.random.randint(-1, 2)
            dy = np.random.randint(-1, 2)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            img = cv2.warpAffine(img, M, (32, 32))
            
            img_path = os.path.join(person_dir, f"sample_{img_idx:03d}.jpg")
            cv2.imwrite(img_path, img)

    print(f"[OK] Đã tạo thành công dữ liệu mẫu tại: {PROCESSED_DIR}")
    # Tự động tạo luôn cặp kiểm thử
    generate_test_pairs(num_genuine_per_id=12, num_impostor_pairs=60)


def export_database(model_weights=None, save_file="face_database.pt"):
    """
    Trích xuất vector trung bình cho từng người trong dataset/processed_32x32/
    và lưu thành face_database.pt phục vụ face_system.py nhận diện thời gian thực.
    """
    import torch
    import torch.nn.functional as F
    from tiny_mobilefacenet import TinyMobileFaceNet32

    default_distilled = os.path.join(BASE_DIR, "weights", "tiny_mobilefacenet_distilled.pt")
    weights_path = model_weights if (model_weights and os.path.exists(model_weights)) else (default_distilled if os.path.exists(default_distilled) else None)

    model = TinyMobileFaceNet32(embedding_size=128)
    if weights_path:
        print(f"[+] Sử dụng trọng số: {weights_path}")
        ckpt = torch.load(weights_path, map_location='cpu')
        sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(sd)
    model.eval()

    face_db = {}
    if not os.path.exists(PROCESSED_DIR):
        print(f"[!] Thư mục '{PROCESSED_DIR}' chưa tồn tại.")
        return

    print("\n" + "=" * 65)
    print("XUẤT CƠ SỞ DỮ LIỆU ĐẶC TRƯNG KHUÔN MẶT (FACE DATABASE)")
    print("=" * 65)

    for person_name in os.listdir(PROCESSED_DIR):
        p_dir = os.path.join(PROCESSED_DIR, person_name)
        if not os.path.isdir(p_dir):
            continue
        vectors = []
        for img_name in os.listdir(p_dir):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(p_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img_rgb.shape[:2] != (32, 32):
                img_rgb = cv2.resize(img_rgb, (32, 32))
            norm = (img_rgb.astype(np.float32) - 127.5) / 128.0
            tensor = torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0)
            with torch.no_grad():
                v = model(tensor)
            vectors.append(v)

        if len(vectors) > 0:
            # Lấy trung bình vector để tăng độ tin cậy
            avg_vec = torch.mean(torch.cat(vectors, dim=0), dim=0, keepdim=True)
            avg_vec = F.normalize(avg_vec, p=2, dim=1)
            face_db[person_name] = avg_vec
            print(f"- [{person_name}]: Tổng hợp từ {len(vectors)} ảnh")

    save_path = os.path.join(BASE_DIR, save_file)
    torch.save(face_db, save_path)
    print("=" * 65)
    print(f"[THÀNH CÔNG] Đã lưu cơ sở dữ liệu {len(face_db)} danh tính vào:\n-> {save_path}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quản lý tập dữ liệu cho TinyMobileFaceNet-32")
    parser.add_argument("--collect", action="store_true", help="Thu thập ảnh từ Webcam")
    parser.add_argument("--name", type=str, default="User", help="Tên danh tính cần thu thập")
    parser.add_argument("--count", type=int, default=25, help="Số lượng ảnh cần chụp")
    parser.add_argument("--process", action="store_true", help="Cắt và chuẩn hóa ảnh về 32x32")
    parser.add_argument("--generate-pairs", action="store_true", help="Sinh file cặp kiểm thử test_pairs.txt")
    parser.add_argument("--create-sample", action="store_true", help="Tự động tạo dữ liệu mẫu thử nghiệm")
    parser.add_argument("--export-db", action="store_true", help="Xuất file face_database.pt từ dữ liệu 32x32")

    args = parser.parse_args()

    if args.collect:
        collect_from_webcam(args.name, max_samples=args.count)
    elif args.process:
        process_raw_dataset()
        generate_test_pairs()
    elif args.generate_pairs:
        generate_test_pairs()
    elif args.create_sample:
        create_sample_dataset()
    elif args.export_db:
        export_database()
    else:
        print("\n=== HƯỚNG DẪN SỬ DỤNG DATASET_MANAGER.PY ===")
        print("1. Thu thập ảnh từ Webcam:")
        print("   python dataset_manager.py --collect --name Tuyen_Duong --count 25")
        print("2. Cắt và chuẩn hóa ảnh thư mục raw/ về 32x32:")
        print("   python dataset_manager.py --process")
        print("3. Tự sinh lại file cặp kiểm thử test_pairs.txt:")
        print("   python dataset_manager.py --generate-pairs")
        print("4. Tạo nhanh dữ liệu mẫu để thử nghiệm tức thì:")
        print("   python dataset_manager.py --create-sample")
        print("5. Xuất cơ sở dữ liệu face_database.pt phục vụ face_system.py:")
        print("   python dataset_manager.py --export-db\n")
