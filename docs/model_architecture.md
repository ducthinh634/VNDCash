# KIẾN TRÚC MÔ HÌNH VÀ CHIẾN LƯỢC HUẤN LUYỆN (MODEL ARCHITECTURE & TRAINING PROTOCOL)
**Dự án**: XÂY DỰNG ỨNG DỤNG NHẬN DIỆN TIỀN GIẤY VIỆT NAM BẰNG CNN (VNDCash)  
**Mã công việc**: Stack02 - Custom CNN & Training Engine  
**Thành viên phụ trách**: Tô Quang Dũng, Lê Quang Minh Nhật, Nguyễn Trường Tấn  
**Mã nghiệm thu**: `Stack201` -> `Stack210`  
**Ngày cập nhật**: 05/11/2026  

---

## 1. TỔNG QUAN THIẾT KẾ KIẾN TRÚC BASELINE CNN (`Stack201`, `Stack204`)

Mô hình **Baseline CNN** được thiết kế tùy biến riêng cho bài toán nhận diện 09 mệnh giá tiền giấy Việt Nam, ưu tiên tính gọn nhẹ, tốc độ suy luận nhanh trên CPU/thiết bị di động, hạn chế tối đa rủi ro quá khớp (overfitting) với tập dữ liệu thực nghiệm ~2.000 ảnh.

### 1.1. Bảng đặc tả chi tiết từng tầng (Layer Breakdown)

| Tầng (Layer) | Loại tầng (Type) | Kích thước đầu vào | Cấu hình Kernel / Stride / Pad | Kích thước đầu ra | Kích hoạt & Điều chuẩn | Số tham số |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Input** | Tensor hình ảnh | - | - | `[B, 3, 224, 224]` | Chuẩn hóa ImageNet | 0 |
| **Block 1** | Conv2d | `[B, 3, 224, 224]` | $3 \times 3$, stride 1, pad 1 | `[B, 32, 224, 224]` | BatchNorm2d + ReLU | 864 |
| | MaxPool2d | `[B, 32, 224, 224]` | $2 \times 2$, stride 2 | `[B, 32, 112, 112]` | - | 0 |
| **Block 2** | Conv2d | `[B, 32, 112, 112]` | $3 \times 3$, stride 1, pad 1 | `[B, 64, 112, 112]` | BatchNorm2d + ReLU | 18.432 |
| | MaxPool2d | `[B, 64, 112, 112]` | $2 \times 2$, stride 2 | `[B, 64, 56, 56]` | - | 0 |
| **Block 3** | Conv2d | `[B, 64, 56, 56]` | $3 \times 3$, stride 1, pad 1 | `[B, 128, 56, 56]` | BatchNorm2d + ReLU | 73.728 |
| | MaxPool2d | `[B, 64, 56, 56]` | $2 \times 2$, stride 2 | `[B, 128, 28, 28]` | - | 0 |
| **Block 4** | Conv2d | `[B, 128, 28, 28]` | $3 \times 3$, stride 1, pad 1 | `[B, 256, 28, 28]` | BatchNorm2d + ReLU | 294.912 |
| | MaxPool2d | `[B, 128, 28, 28]` | $2 \times 2$, stride 2 | `[B, 256, 14, 14]` | - | 0 |
| **Pooling** | AdaptiveAvgPool2d | `[B, 256, 14, 14]` | target $(4, 4)$ | `[B, 256, 4, 4]` | Giữ không gian đặc trưng | 0 |
| **Flatten** | Flatten | `[B, 256, 4, 4]` | flatten sang vector | `[B, 4096]` | - | 0 |
| **FC 1** | Linear + BN | `[B, 4096]` | in: 4096, out: 256 | `[B, 256]` | BatchNorm1d + ReLU + Dropout(0.3) | 1.049.088 |
| **Output Head**| Linear | `[B, 256]` | in: 256, out: 9 | `[B, 9]` | Logits của 09 mệnh giá | 2.313 |
| **TỔNG CỘNG**| | | | | | **~1.439.337 tham số (~5.5 MB)** |

---

## 2. QUY CHUẨN ĐẦU RA 09 MỆNH GIÁ (`Stack203`)

Mô hình đảm bảo đúng 9 đầu ra (logits) tương ứng với 9 chỉ số lớp:
- `0`: 1.000 VND (Cotton)
- `1`: 2.000 VND (Cotton)
- `2`: 5.000 VND (Cotton)
- `3`: 10.000 VND (Polymer)
- `4`: 20.000 VND (Polymer)
- `5`: 50.000 VND (Polymer)
- `6`: 100.000 VND (Polymer)
- `7`: 200.000 VND (Polymer)
- `8`: 500.000 VND (Polymer)

Xác suất mỗi lớp được tính qua hàm Softmax:
$$P(y = c \mid x) = \frac{e^{z_c}}{\sum_{j=0}^{8} e^{z_j}}$$

---

## 3. FACTORY MÔ HÌNH VÀ SO SÁNH 3 KIẾN TRÚC (`Stack205`)

Module `src/models/factory.py` cung cấp interface duy nhất `build_model(model_name, num_classes=9)` hỗ trợ khởi tạo linh hoạt 3 mô hình của đề tài:

| Tiêu chí | (1) Baseline CNN (Tự thiết kế) | (2) MobileNetV2 (Transfer Learning) | (3) ResNet-18 (Transfer Learning) |
| :--- | :---: | :---: | :---: |
| **Triết lý kiến trúc** | Thuần CNN truyền thống, 4 blocks | Inverted Residuals & Linear Bottlenecks | Residual Connections (Skip connection) |
| **Số tham số (Params)** | **~1.44 triệu** | ~2.23 triệu | ~11.18 triệu |
| **Kích thước file trọng số** | **~5.5 MB** | ~8.9 MB | ~44.7 MB |
| **Độ phức tạp suy luận** | Rất thấp, tối ưu cho mobile | Thấp, chuyên dụng di động | Trung bình |
| **Mục đích trong đề tài** | Mốc so sánh cơ sở (Baseline) | Ứng dụng triển khai app di động | Đánh giá độ chính xác tối đa |

---

## 4. CHIẾN LƯỢC HUẤN LUYỆN VÀ HÀM MẤT MÁT (`Stack207`, `Stack208`)

### 4.1. Hàm mất mát có trọng số (Weighted Cross-Entropy Loss)
Để khắc phục hiện tượng mất cân bằng mẫu ở 3 lớp `5.000`, `20.000`, `50.000` (chỉ có 114 ảnh Train so với 170 ảnh ở các lớp khác), hàm mất mát tính trọng số nghịch đảo tần suất lớp:
$$w_c = \frac{N_{\text{train}}}{K \cdot n_c}$$
Hàm mục tiêu tối ưu:
$$\mathcal{L} = -\sum_{c=0}^{8} w_c \cdot y_c \log(\hat{y}_c)$$
Đồng thời áp dụng **Label Smoothing** ($0.05$) để ngăn chặn mô hình đưa ra dự đoán quá tự tin (overconfident predictions).

### 4.2. Bộ tối ưu hóa (Optimizer) & Điều chỉnh tốc độ học (LR Scheduler)
- **Optimizer**: `Adam` với Learning Rate khởi tạo $\eta = 0.001$, Weight Decay $= 10^{-4}$ (L2 regularization).
- **LR Scheduler**: `ReduceLROnPlateau` với `mode='min'`, `factor=0.5`, `patience=3`. Khi validation loss không giảm sau 3 epochs, tốc độ học tự động giảm 50%.
- **Gradient Clipping**: Giới hạn chuẩn gradient tối đa $\le 5.0$ nhằm ổn định quá trình lan truyền ngược.

### 4.3. Callbacks & Cơ chế kiểm soát huấn luyện
1. **EarlyStopping**: Dừng huấn luyện nếu validation loss không cải thiện sau $6$ epochs liên tiếp (`patience=6`).
2. **ModelCheckpoint**: Tự động lưu checkpoint tốt nhất vào `artifacts/checkpoints/baseline_cnn_best.pt` và xuất kèm `baseline_cnn_manifest.json`.
3. **MetricLogger**: Ghi lại lịch sử từng epoch ra cả định dạng `.csv` và `.json` tại `artifacts/logs/`.
4. **Learning Curves Plotter**: Tự động vẽ đồ thị hàm mất mát và độ chính xác (Train vs Validation) tại `artifacts/figures/`.

---

## 5. HƯỚNG DẪN THỰC THI (CLI COMMANDS)

### 5.1. Chạy huấn luyện từ dòng lệnh (`Stack209`)
```bash
# Huấn luyện mô hình Baseline CNN mặc định theo configs/baseline.yaml
python scripts/train.py

# Hoặc truyền tham số tùy chỉnh:
python scripts/train.py --epochs 20 --batch-size 32 --lr 0.001
```

### 5.2. Chạy bộ kiểm thử mô hình
```bash
python -m pytest tests/test_models.py -v
```

### 5.3. Xem trực quan hóa kết quả trong Notebook (`Stack210`)
Mở notebook `notebooks/03_model_experiments.ipynb` trong VS Code để theo dõi các đường cong học tập (Learning Curves), ma trận nhầm lẫn sơ bộ và trực quan hóa các mẫu dự đoán.
