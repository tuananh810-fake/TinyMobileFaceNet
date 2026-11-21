# LFW Dataset (Labeled Faces in the Wild)

Mô hình **TinyMobileFaceNet** được huấn luyện và đánh giá trên bộ dữ liệu chuẩn **LFW** (Labeled Faces in the Wild):

- **Nguồn dữ liệu chính thức:** Đại học Massachusetts Amherst (UMass)
- **Trang chủ:** [http://vis-www.cs.umass.edu/lfw/](http://vis-www.cs.umass.edu/lfw/)
- **Tải ảnh LFW trực tiếp:**
  - [lfw.tgz (173 MB)](http://vis-www.cs.umass.edu/lfw/lfw.tgz) hoặc [lfw-funneled.tgz (233 MB)](http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz)

---

## 📁 Cấu trúc thư mục sau khi tải và giải nén

Để chạy kiểm thử benchmark trên máy cục bộ, bạn hãy giải nén ảnh khuôn mặt LFW vào thư mục `dataset/` theo cấu trúc sau:

```text
dataset/
├── pairs.txt                  # 6,000 cặp kiểm thử chuẩn (đã có sẵn)
├── README.md                  # Hướng dẫn này
└── lfw/                       # Thư mục ảnh khuôn mặt tải về
    ├── George_W_Bush/
    │   ├── George_W_Bush_0001.jpg
    │   └── ...
    ├── Colin_Powell/
    │   ├── Colin_Powell_0001.jpg
    │   └── ...
    └── ...
```

---

## ⚙️ Tiền xử lý (Preprocessing)

Trong quá trình huấn luyện và đánh giá:
1. Mỗi khuôn mặt trong tập LFW được phát hiện và cắt theo khung hình vuông tỷ lệ **1:1** ôm sát khuôn mặt (Face Crop).
2. Ảnh sau khi cắt được co về ma trận kích thước **`32 x 32 x 3`** bằng phép nội suy diện tích (`cv2.INTER_AREA`).
3. Chuẩn hóa giá trị điểm ảnh về dải đối xứng `[-1, 1]` theo công thức:
   $$x_{\text{norm}} = \frac{x - 127.5}{128.0}$$
4. Đưa vào mô hình `TinyMobileFaceNet-32` để trích xuất vector đặc trưng 128 chiều (`[B, 128]`).
