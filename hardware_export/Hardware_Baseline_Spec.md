# Bảng Tiêu Chí Gốc Phần Cứng (Hardware Baseline Specification)

- **Mô hình:** `TinyMobileFaceNet-32` (Full INT8 QAT — 0 lớp BatchNorm)
- **Tác giả:** Nguyễn Sỹ Tuấn Anh
- **Định dạng số học phần cứng:** `Input INT8 × Weight INT8 → Accumulator INT32 + Bias INT32 → (*M >> 16) → PReLU Q0.7 → Output INT8`

## 1. Chỉ tiêu độ tin cậy chuẩn 10-Fold LFW (6.000 cặp ảnh)

| Phiên bản mô hình | 10-Fold Accuracy | ROC AUC | Equal Error Rate (EER) | Threshold |
| :--- | :---: | :---: | :---: | :---: |
| FP32 Folded-BN (Gốc) | 96.68% ± 0.83% | 0.9945 | 3.22% | 0.243 |
| PTQ Full INT8 (Trước QAT) | 96.72% ± 0.61% | 0.9941 | 3.22% | 0.238 |
| **QAT Full INT8 (Integer-Only Golden Model)** | **97.07% ± 0.78%** | **0.9941** | **3.27%** | **0.240** |

## 2. Tổng hợp tài nguyên phần cứng (Hardware Resource Budget)

| Tiêu chí phần cứng | Giá trị chính xác | Ghi chú thiết kế Datapath |
| :--- | :---: | :--- |
| Tổng số tầng tích chập | **18 tầng `Conv2d`** | 0 tầng BatchNorm (đã gập hoàn toàn) |
| **Tổng số phép tính (MACs)** | **5,758,976** (`5.759 MMACs`) | Mảng 9 DSP @ 100 MHz $\approx$ **6.40 ms** (~156 FPS) |
| Trọng số `Conv2d` (`weights_int8.mem`) | **215,360 B (`210.31 KB`)** | `INT8` đối xứng `[-127, +127]` |
| Trọng số `PReLU` (`prelu_q07.mem`) | **1,728 B (`1.69 KB`)** | `INT8` fixed-point `Q0.7` (`>> 7`) |
| Hằng số `Bias` (`bias_int32.mem`) | **2,528 giá trị (`9.88 KB`)** | `INT32` cộng trực tiếp vào thanh ghi Accumulator |
| Hệ số làm tròn `M` (`requant_mult_int32.mem`) | **2,528 giá trị (`9.88 KB`)** | Dịch phải cố định `>> 16` cho mọi tầng |
| **Bộ nhớ 1 Bank trung gian lớn nhất** | **16,384 B (`16.00 KB`)** | Tầng `L04_s1_exp` & `L10_s3_exp` |
| **Tổng Ping-Pong RAM tối thiểu (In + Out)** | **24,576 B (`24.00 KB`)** | Khuyến nghị cấp phát **2 bank × 16 KB = 32 KB** |

## 3. Bảng chi tiết 18 tầng cho thiết kế Datapath & Controller

| # | Tên tầng | Input `(C,H,W)` | Output `(C,H,W)` | Kernel/S/G | PReLU | MACs | Weight (B) | Act Out (B) |
| :-: | :--- | :---: | :---: | :---: | :-: | ---: | ---: | ---: |
| 01 | `L01_stem` | `(3, 32, 32)` | `(32, 16, 16)` | `3x3/s2/g1` | Có | 221,184 | 864 | 8,192 |
| 02 | `L02_dw_stem_dw` | `(32, 16, 16)` | `(32, 16, 16)` | `3x3/s1/g32` | Có | 73,728 | 288 | 8,192 |
| 03 | `L03_dw_stem_pw` | `(32, 16, 16)` | `(32, 16, 16)` | `1x1/s1/g1` | Không | 262,144 | 1,024 | 8,192 |
| 04 | `L04_s1_exp` | `(32, 16, 16)` | `(64, 16, 16)` | `1x1/s1/g1` | Có | 524,288 | 2,048 | 16,384 |
| 05 | `L05_s1_dw` | `(64, 16, 16)` | `(64, 8, 8)` | `3x3/s2/g64` | Có | 36,864 | 576 | 4,096 |
| 06 | `L06_s1_pw` | `(64, 8, 8)` | `(64, 8, 8)` | `1x1/s1/g1` | Không | 262,144 | 4,096 | 4,096 |
| 07 | `L07_s2_exp` | `(64, 8, 8)` | `(128, 8, 8)` | `1x1/s1/g1` | Có | 524,288 | 8,192 | 8,192 |
| 08 | `L08_s2_dw` | `(128, 8, 8)` | `(128, 8, 8)` | `3x3/s1/g128` | Có | 73,728 | 1,152 | 8,192 |
| 09 | `L09_s2_pw` | `(128, 8, 8)` | `(64, 8, 8)` | `1x1/s1/g1` | Không | 524,288 | 8,192 | 4,096 |
| 10 | `L10_s3_exp` | `(64, 8, 8)` | `(256, 8, 8)` | `1x1/s1/g1` | Có | 1,048,576 | 16,384 | 16,384 |
| 11 | `L11_s3_dw` | `(256, 8, 8)` | `(256, 4, 4)` | `3x3/s2/g256` | Có | 36,864 | 2,304 | 4,096 |
| 12 | `L12_s3_pw` | `(256, 4, 4)` | `(128, 4, 4)` | `1x1/s1/g1` | Không | 524,288 | 32,768 | 2,048 |
| 13 | `L13_s4_exp` | `(128, 4, 4)` | `(256, 4, 4)` | `1x1/s1/g1` | Có | 524,288 | 32,768 | 4,096 |
| 14 | `L14_s4_dw` | `(256, 4, 4)` | `(256, 4, 4)` | `3x3/s1/g256` | Có | 36,864 | 2,304 | 4,096 |
| 15 | `L15_s4_pw` | `(256, 4, 4)` | `(128, 4, 4)` | `1x1/s1/g1` | Không | 524,288 | 32,768 | 2,048 |
| 16 | `L16_conv_sep` | `(128, 4, 4)` | `(256, 4, 4)` | `1x1/s1/g1` | Có | 524,288 | 32,768 | 4,096 |
| 17 | `L17_gdconv` | `(256, 4, 4)` | `(256, 1, 1)` | `4x4/s1/g256` | Không | 4,096 | 4,096 | 256 |
| 18 | `L18_linear_conv` | `(256, 1, 1)` | `(128, 1, 1)` | `1x1/s1/g1` | Không | 32,768 | 32,768 | 128 |

## 4. Bộ file Golden Model & nạp mạch (`hardware_export/`)
- `weights_int8.mem`, `bias_int32.mem`, `prelu_q07.mem`, `requant_mult_int32.mem`: Nạp bằng `$readmemh` trong Verilog.
- `quant_params.json`: Địa chỉ offset, shape và hằng số `M, shift` cho FSM Controller.
- `golden_vectors/`: Chứa `input_32x32_int8.mem` + 18 file đáp án chuẩn từng tầng (`L01_stem_int8.mem` ... `L18_linear_conv_int8.mem`) để đối chiếu bit-exact với Testbench Verilog.
