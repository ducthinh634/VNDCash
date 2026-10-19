# HƯỚNG DẪN SỬ DỤNG DATA PIPELINE & GHI CHÚ BÀN GIAO CHO CÁC THÀNH VIÊN
**Dự án**: XÂY DỰNG ỨNG DỤNG NHẬN DIỆN TIỀN GIẤY VIỆT NAM BẰNG CNN (VNDCash)  
**Người lập**: Thành viên A (Nguyễn Đức Thịnh - Stack01: Data Engineering & Data Pipeline)  
**Đối tượng bàn giao**: Thành viên B (Training/Baseline), C (Models), D (Evaluation), E (Inference/Testing)  
**Ngày cập nhật**: 19/10/2026  

---

## I. TỔNG QUAN HIỆN TRẠNG DỮ LIỆU ĐÃ HOÀN THIỆN

1. **Phạm vi 09 mệnh giá chuẩn**:
   - `1.000`, `2.000`, `5.000` (Tiền giấy Cotton)
   - `10.000`, `20.000`, `50.000`, `100.000`, `200.000`, `500.000` (Tiền Polymer)
   - Tổng cộng: **1.962 ảnh hợp lệ** (0 ảnh hỏng, 0 ảnh rỗng, 0 trùng lặp).
2. **Quản lý tờ tiền vật lý (`note_id`)**:
   - Toàn bộ ảnh được gom thành **198 tờ tiền vật lý độc lập** (`NOTE_...`).
3. **Phân chia tập (Grouped Split) & Cam kết Zero Data Leakage**:
   - **Tập Train**: **1.362 ảnh** (69.42%) — 138 tờ tiền
   - **Tập Val**: **300 ảnh** (15.29%) — 30 tờ tiền
   - **Tập Test**: **300 ảnh** (15.29%) — 30 tờ tiền
   - **Giao thoa**: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$ (Không có bất kỳ tờ tiền hay ảnh nào bị rò rỉ giữa các tập).
4. **Các lớp ngoài phạm vi (Excluded)**:
   - 750 ảnh thuộc các thư mục `000000` (nền/ảnh khác), `000200` (200đ), `000500` (500đ) được giữ nguyên trong `data/raw/` nhưng đã được loại trừ khỏi pipeline huấn luyện 9 lớp.

---

## II. HƯỚNG DẪN DÀNH CHO THÀNH VIÊN B (HUẤN LUYỆN & BASELINE CNN)

### 1. Cách nạp DataLoaders vào code huấn luyện (`train.py`)
Thành viên B không cần viết lại logic đọc ảnh hay biến đổi dữ liệu, chỉ cần import hàm có sẵn:
```python
from src.data.dataset import get_dataloaders

# Tự động nạp cấu hình từ configs/data.yaml và splits.csv
train_loader, val_loader, test_loader = get_dataloaders(
    config_path="configs/data.yaml",
    batch_size=32,      # Có thể tùy chỉnh batch size theo GPU/CPU
    num_workers=0       # 0 là an toàn nhất trên Windows
)

# Sử dụng trong training loop:
for images, labels in train_loader:
    # images shape: [batch_size, 3, 224, 224]
    # labels shape: [batch_size] với giá trị từ 0 đến 8
    outputs = model(images)
    loss = criterion(outputs, labels)
    ...
```

### 2. CẢNH BÁO QUAN TRỌNG: Xử lý 3 lớp thiếu mẫu
- **Hiện tượng**: Mệnh giá **5.000**, **20.000**, và **50.000** chỉ có **154 ảnh/lớp** (thiếu 38.4% so với các lớp khác có 250 ảnh).
- **Giải pháp cho bạn B**: Để tránh việc mô hình bị thiên lệch (bias) về các lớp nhiều ảnh, hãy áp dụng **Class Weights** vào hàm mất mát `CrossEntropyLoss`:
```python
import torch
import torch.nn as nn

# Tính trọng số nghịch đảo tần suất xuất hiện trong tập Train
# Số ảnh Train: 1k (170), 2k (170), 5k (114), 10k (170), 20k (114), 50k (114), 100k (170), 200k (170), 500k (170)
train_counts = [170, 170, 114, 170, 114, 114, 170, 170, 170]
total_train = sum(train_counts)
weights = [total_train / (len(train_counts) * c) for c in train_counts]
class_weights = torch.tensor(weights, dtype=torch.float)

# Khởi tạo loss có trọng số:
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
```

### 3. Quy tắc Data Augmentation
- Pipeline đã được thiết kế sẵn theo đúng nguyên tắc: **Chỉ áp dụng Augmentation trên tập Train** (xoay góc $\pm 15^\circ$, lật ngang, chỉnh sáng, co giãn, làm mờ).
- Tập **Val** và **Test** đã tự động khóa, chỉ thực hiện Resize 224x224 và chuẩn hóa ImageNet để số liệu đánh giá hoàn toàn khách quan.

### 4. Quy tắc lưu Checkpoints
- Lưu các checkpoint huấn luyện vào thư mục `artifacts/checkpoints/`.
- **Tuyệt đối không commit các file nặng (`.pt`, `.pth`, `.h5`) lên GitHub** (file `.gitignore` của nhóm đã chặn sẵn).

---

## III. HƯỚNG DẪN DÀNH CHO THÀNH VIÊN C (MÔ HÌNH MOBILENETV2 & RESNET18)

1. **Kích thước đầu vào chuẩn hóa (Input Shape)**:
   - Cả 2 mô hình phải nhận tensor đầu vào có kích thước: **`[batch_size, 3, 224, 224]`**.
   - Giá trị pixel đã được chuẩn hóa theo ImageNet:
     - Mean: `[0.485, 0.456, 0.406]`
     - Std: `[0.229, 0.224, 0.225]`
2. **Đầu ra phân loại (Classification Head)**:
   - Lớp cuối cùng (FC/Linear) của cả MobileNetV2 và ResNet18 bắt buộc phải có số node ra là **9** (`num_classes = 9`).
   - Thứ tự mapping nhãn từ 0 đến 8 tương ứng:
     - `0`: 1.000 VND
     - `1`: 2.000 VND
     - `2`: 5.000 VND
     - `3`: 10.000 VND
     - `4`: 20.000 VND
     - `5`: 50.000 VND
     - `6`: 100.000 VND
     - `7`: 200.000 VND
     - `8`: 500.000 VND

---

## IV. HƯỚNG DẪN DÀNH CHO THÀNH VIÊN D (ĐÁNH GIÁ, METRICS & PHÂN TÍCH LỖI)

1. **Tập dữ liệu kiểm thử bắt buộc (Frozen Test Protocol)**:
   - Khi đánh giá nghiệm thu mô hình, bạn D **bắt buộc phải lấy đúng 300 ảnh thuộc tập `test`** trong file [data/metadata/splits.csv](file:///c:/Users/ndt06/Desktop/VNDCash/data/metadata/splits.csv).
   - Tuyệt đối không tự ý random split lại dữ liệu.
2. **Phân tích theo chất liệu (Cotton vs Polymer)**:
   - Trong bảng `metadata.csv` và `splits.csv` đã có sẵn cột `material` (`cotton` cho 1k, 2k, 5k và `polymer` cho 10k đến 500k).
   - Hãy trích xuất bảng so sánh xem mô hình nhận diện tiền Cotton hay Polymer tốt hơn để đưa vào báo cáo đồ án.
3. **Phân tích chi tiết per-class Recall**:
   - Đặc biệt chú ý kiểm tra độ chính xác trên 3 lớp thiếu mẫu (`5.000`, `20.000`, `50.000`) xem có bị nhầm lẫn sang các mệnh giá có màu tương đồng không (ví dụ: 20k xanh lam với 500k xanh lam, hoặc 10k với 200k).
4. **Kiểm thử Robustness / Out-of-Distribution (OOD)**:
   - Bạn D có thể sử dụng 750 ảnh từ 3 thư mục `data/raw/000000`, `data/raw/000200`, `data/raw/000500` để thử nghiệm đưa ảnh không hợp lệ vào mô hình, kiểm tra xem mô hình có bị phán đoán bừa hay không.

---

## V. HƯỚNG DẪN DÀNH CHO THÀNH VIÊN E (INFERENCE, APP CAMERA & TFLITE)

1. **Cơ chế lọc độ tin cậy (Confidence Gate)**:
   - Khi đưa mô hình vào app điện thoại hoặc demo webcam, lấy xác suất qua hàm `softmax`.
   - Thiết lập ngưỡng tin cậy `threshold >= 0.75 - 0.80`:
     - Nếu $\max(\text{Softmax}) \ge 0.80$: Đọc to mệnh giá (Text-to-Speech) và hiển thị kết quả.
     - Nếu $\max(\text{Softmax}) < 0.80$: Thông báo "Không nhận diện được tờ tiền / Vui lòng đưa gần hoặc lấy nét lại", tránh việc nhận diện sai mệnh giá tiền của người dùng.
2. **Lưu mô hình Inference**:
   - Khi chọn được mô hình tối ưu nhất (giữa Baseline CNN, MobileNetV2, ResNet18), lưu file model vào `artifacts/models/`.
   - Tạo file `manifest.json` ghi chú rõ: nguồn checkpoint từ epoch nào, ngày xuất, độ chính xác Accuracy/F1 trên tập Test.
3. **Quy tắc nghiệm thu**:
   - Các kết quả demo nhanh tại `artifacts/predictions/` chỉ dùng làm minh họa, không dùng thay thế cho số liệu chính thức của Test Protocol do bạn D xuất ra tại `artifacts/reports/`.

---

## VI. BẢNG TRA CỨU CÂU LỆNH NHANH (CHEATSHEET)

| Thao tác | Câu lệnh Terminal |
| :--- | :--- |
| **Kiểm tra / Audit lại dữ liệu** | `python scripts/inspect_dataset.py` |
| **Tạo lại file metadata** | `python scripts/build_metadata.py` |
| **Phân chia lại tập Train/Val/Test** | `python scripts/create_split.py` |
| **Chạy kiểm thử tự động toàn bộ pipeline** | `python -m pytest tests/ -v` |
| **Cập nhật code từ GitHub** | `git pull origin main` |

---
*Mọi thắc mắc về luồng dữ liệu, đường dẫn ảnh hoặc nhãn tờ tiền, các thành viên liên hệ trực tiếp Thành viên A (Thịnh).*
