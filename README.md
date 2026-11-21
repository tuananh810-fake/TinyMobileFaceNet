# TinyMobileFaceNet: Lightweight Face Recognition Model

Tác giả: **Nguyễn Sỹ Tuấn Anh** ([@tuananh810-fake](https://github.com/tuananh810-fake))

TinyMobileFaceNet là một kiến trúc mạng nơ-ron nhận diện khuôn mặt siêu nhẹ, được kế thừa và tinh gọn từ mô hình **MobileFaceNet (2018)** nhằm phục vụ cho các hệ thống nhúng và vi xử lý hạn chế tài nguyên bộ nhớ.

## 🌟 Đặc điểm nổi bật
- **Kích thước đầu vào:** $32 \times 32 \times 3$ (giảm 91.8% dung lượng đệm so với chuẩn $112 \times 112$).
- **Số lượng tham số:** **217,088 parameters** (~212 KB ở định dạng INT8, ~848 KB ở FP32).
- **Phương pháp huấn luyện:** Áp dụng **Knowledge Distillation (Chưng cất tri thức)** từ mô hình giáo viên MobileFaceNet gốc $112 \times 112$.
- **Độ tin cậy cao trên chuẩn LFW:**
  - **Độ chính xác (Accuracy):** **90.13%** (Peak: 90.30%)
  - **Diện tích dưới đường cong ROC (AUC):** **0.9601**
  - **Tỷ lệ lỗi cân bằng (Equal Error Rate - EER):** **9.97%**
  - **Ngưỡng khoảng cách tối ưu (Optimal Threshold):** **0.273**

![LFW Benchmark](docs/lfw_benchmark_report.png)

## 🚀 Cài đặt & Sử dụng

### 1. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 2. Đánh giá độ tin cậy trên tập kiểm thử LFW
```bash
python evaluate_benchmark.py
```

### 3. Chạy ứng dụng nhận diện thời gian thực qua Webcam
```bash
python camera_demo.py
```

## 📄 Báo cáo kỹ thuật
Báo cáo kỹ thuật chi tiết có thể xem tại: [docs/Kiem_Tra_Tinh_Tin_Cay_TinyMobileFaceNet.pdf](docs/Kiem_Tra_Tinh_Tin_Cay_TinyMobileFaceNet.pdf)

## 📜 Acknowledgements
Dự án kế thừa và phát triển dựa trên kiến trúc [MobileFaceNet](https://arxiv.org/abs/1804.07573) (Sheng Chen et al., 2018).