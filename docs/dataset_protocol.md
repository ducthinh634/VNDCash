# GIAO THỨC DỮ LIỆU & PIPELINE YOLOv11 (DATASET PROTOCOL)
**Dự án**: XÂY DỰNG HỆ THỐNG NHẬN DIỆN ĐA MỆNH GIÁ TIỀN GIẤY VIỆT NAM BẰNG YOLOv11 (VNDCash)  
**Mã công việc**: Stack 01 - Data Engineering, Hard-Case Curation & Anti-Leakage Split (`Stack101` - `Stack114`)  
**Thành viên phụ trách (Nhóm A)**: Nguyễn Đức Thịnh (MSSV: 22696331)  
**Ngày cập nhật**: 17/11/2026  
**Trạng thái**: Hoàn thiện 100% (Đã nghiệm thu qua kiểm thử tự động `pytest` và kiểm định độc lập)

---

## 1. Bối cảnh chuyển đổi & Nguồn dữ liệu

### 1.1. Nâng cấp từ Phân loại đơn mẫu (CNN) sang Nhận diện đa vật thể (YOLOv11)
- Ban đầu bài toán phân loại đơn mẫu (Classification) chỉ nhận diện 1 tờ tiền trên nền tĩnh, không đáp ứng được thực tế khi người dùng thanh toán hay quét một xấp tiền gồm nhiều mệnh giá cùng lúc.
- Do đó, dự án nâng cấp toàn diện sang mô hình **YOLOv11 Object Detection**, cho phép định vị (Bounding Box) và phân loại đồng thời nhiều tờ tiền thuộc các mệnh giá khác nhau trong một khung hình duy nhất.

### 1.2. Nguồn gốc dữ liệu
Bộ dữ liệu được tích hợp và chuẩn hóa từ Roboflow Universe (tác giả Anng - `Vietnamese Currency.v10-v4.yolov11`):
- **Tổng số ảnh multi-note được chọn lọc**: **1.898 ảnh** lưu trữ tại `data/raw/multi_note/images/` và `data/raw/multi_note/labels/`.
- **Tổng số Bounding Box chuẩn hóa**: **6.731 nhãn tờ tiền**.
- **Số lớp nhận diện (nc = 9)**: Đầy đủ 9 mệnh giá tiền giấy Việt Nam lưu hành:
  `['1000', '10000', '100000', '2000', '20000', '200000', '5000', '50000', '500000']`.

---

## 2. Quy chuẩn Bounding Box & Định dạng Nhãn YOLOv11

### 2.1. Cấu trúc nhãn YOLOv11
Mỗi file nhãn `.txt` tương ứng với 1 file ảnh có cùng tên, mỗi dòng đại diện cho một tờ tiền:
```
<class_id> <x_center> <y_center> <width> <height>
```
Trong đó:
- `class_id`: Số nguyên từ `0` đến `8`, tương ứng với danh sách lớp theo thứ tự chuẩn:
  - `0`: 1.000 VND (Cotton)
  - `1`: 10.000 VND (Polymer)
  - `2`: 100.000 VND (Polymer)
  - `3`: 2.000 VND (Cotton)
  - `4`: 20.000 VND (Polymer)
  - `5`: 200.000 VND (Polymer)
  - `6`: 5.000 VND (Cotton)
  - `7`: 50.000 VND (Polymer)
  - `8`: 500.000 VND (Polymer)
- `x_center`, `y_center`: Tọa độ tâm hộp bao chuẩn hóa $\in [0.0, 1.0]$.
- `width`, `height`: Chiều rộng và chiều cao hộp bao chuẩn hóa $\in (0.0, 1.0]$.

### 2.2. Kiểm định và Làm sạch nhãn (`data/annotations/review_needed.csv`)
Tất cả 6.731 Bounding Box được rà soát tự động thông qua `src/vndcash/data/validation.py`:
- Phát hiện và xuất báo cáo toàn bộ các file nghi ngờ vào `data/annotations/review_needed.csv`.
- Sửa triệt để các nhãn bất thường (ví dụ: artifact kích thước $w = 0.0$ do lỗi sinh polygon).
- **Kết quả nghiệm thu**: 100% tọa độ Bounding Box nằm chính xác trong khoảng $[0.0, 1.0]$, tỷ lệ vi phạm biên hoặc suy biến là **0%**.

---

## 3. Tiêu chí Hard-Case Curation

Hệ thống thiết lập cờ `is_hard_case` trong metadata để phân loại và đảm bảo mô hình được huấn luyện trên các tình huống phức tạp thực tế:
1. **Mật độ đa tờ tiền (Multi-Note Density)**: Ảnh chứa từ 2 đến 9 tờ tiền xếp chồng, đan xen. Có tới **1.897 / 1.898 ảnh (99.9%)** là ca nhận diện đa tờ tiền.
2. **Đa mệnh giá đồng thời (Multi-Denomination)**: **1.886 / 1.898 ảnh (99.4%)** chứa từ 2 mệnh giá tiền khác nhau trở lên trong cùng một bối cảnh.
3. **Che khuất & Góc chụp nghiêng**: Bounding Box bao phủ các tờ tiền bị che khuất một phần (occlusion), bị gập nếp hoặc xoay góc trong phạm vi $0^\circ - 360^\circ$.
4. **Hộp bao kích thước nhỏ**: Bounding Box có chiều rộng hoặc chiều cao $< 0.15$ (tờ tiền chụp ở khoảng cách xa hoặc bị lấp dưới tờ tiền khác).

---

## 4. Giao thức Chia tập Chống rò rỉ dữ liệu (Anti-Leakage Group Split)

### 4.1. Nguy cơ Rò rỉ Dữ liệu trên Ảnh Tăng cường (Augmented Scenes)
Trong bộ dữ liệu Roboflow, các ảnh thường được sinh ra từ các cảnh gốc thông qua biến đổi (flip, rotate, color jitter). Nếu chia ngẫu nhiên (random split), các biến thể của cùng một cảnh vật lý sẽ lọt vào cả Train và Test, gây ra hiện tượng **Data Leakage** nghiêm trọng (mô hình ghi nhớ hình nền và hoa văn cụ thể thay vì khái quát hóa mệnh giá).

### 4.2. Giải pháp: Gom cụm theo `note_id` (Physical Base Scene)
- Trích xuất định danh cảnh gốc `note_id` từ tiền tố tên file (trước đoạn mã hash `.rf.`).
- Tổng cộng toàn bộ 1.898 ảnh được quy về **1.158 nhóm `note_id` vật lý độc lập**.
- Thuật toán `stratified_group_split_yolo` thực hiện chia phân tầng theo tỷ lệ chuẩn **70% Train / 15% Val / 15% Test** sao cho tất cả ảnh cùng `note_id` phải nằm trọn vẹn trong **DUY NHẤT một tập**.

### 4.3. Bảng phân chia thực tế (`data/metadata/splits.csv`)
| Phân vùng | Số lượng ảnh | Tỷ lệ ảnh | Số nhóm `note_id` | Tỷ lệ `note_id` | Tổng Bounding Box | Hard-Case |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 1.307 | 68.9% | 810 | 69.9% | 4.611 | 1.306 |
| **Validation** | 295 | 15.5% | 176 | 15.2% | 1.044 | 295 |
| **Test** | 296 | 15.6% | 172 | 14.9% | 1.076 | 296 |
| **Tổng cộng** | **1.898** | **100.0%** | **1.158** | **100.0%** | **6.731** | **1.897** |

> **Cam kết Zero Data Leakage**:
> - Số lượng `note_id` giao thoa giữa Train - Val: **0**
> - Số lượng `note_id` giao thoa giữa Train - Test: **0**
> - Số lượng `note_id` giao thoa giữa Val - Test: **0**
> - Số lượng mã băm MD5 giao thoa giữa các tập: **0**

---

## 5. Thống kê Phân bố Bounding Box theo Mệnh giá

Kết quả kiểm toán chi tiết từ `scripts/inspect_dataset.py` (lưu tại `artifacts/reports/yolo_audit_report.json`):

| Mệnh giá | Số BBox (Train) | Số BBox (Val) | Số BBox (Test) | Tổng BBox | Tỷ lệ toàn tập | Đánh giá |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1.000 VND** | 489 | 107 | 118 | 714 | 10.6% | Cân bằng lý tưởng |
| **2.000 VND** | 485 | 110 | 120 | 715 | 10.6% | Cân bằng lý tưởng |
| **5.000 VND** | 480 | 110 | 118 | 708 | 10.5% | Đã bù đắp đủ dữ liệu |
| **10.000 VND** | 469 | 109 | 117 | 695 | 10.3% | Cân bằng lý tưởng |
| **20.000 VND** | 500 | 115 | 111 | 726 | 10.8% | Đã bù đắp đủ dữ liệu |
| **50.000 VND** | 606 | 134 | 141 | 881 | 13.1% | Đã bù đắp đủ dữ liệu |
| **100.000 VND** | 609 | 139 | 150 | 898 | 13.3% | Đủ mẫu chuẩn |
| **200.000 VND** | 493 | 115 | 108 | 716 | 10.6% | Cân bằng lý tưởng |
| **500.000 VND** | 480 | 105 | 93 | 678 | 10.1% | Cân bằng lý tưởng |
| **Tổng cộng** | **4.611** | **1.044** | **1.076** | **6.731** | **100.0%** | **Không còn mất cân bằng lớp** |

---

## 6. Cấu trúc Bộ dữ liệu Chuẩn Ultralytics YOLOv11

Thư mục `data/processed/yolo_dataset/` được tổ chức đúng cấu trúc chuẩn của Ultralytics, sẵn sàng đưa vào lệnh `yolo train`:
```
data/processed/yolo_dataset/
├── data.yaml
├── images/
│   ├── train/  (1.307 ảnh)
│   ├── val/    (295 ảnh)
│   └── test/   (296 ảnh)
└── labels/
    ├── train/  (1.307 file .txt)
    ├── val/    (295 file .txt)
    └── test/   (296 file .txt)
```

Nội dung file `data.yaml`:
```yaml
path: C:/Users/ndt06/Desktop/VNDCash/data/processed/yolo_dataset
train: images/train
val: images/val
test: images/test
nc: 9
names:
  - '1000'
  - '10000'
  - '100000'
  - '2000'
  - '20000'
  - '200000'
  - '5000'
  - '50000'
  - '500000'
```

---

## 7. Hướng dẫn Lệnh kiểm thử & Tái lập (Reproducibility)

Để kiểm tra và tái lập toàn bộ pipeline từ đầu:

1. **Sinh lại metadata**:
   ```bash
   python scripts/build_metadata.py
   ```
2. **Chia tập dữ liệu và xuất cây thư mục Ultralytics**:
   ```bash
   python scripts/create_split.py
   ```
3. **Kiểm toán và xuất báo cáo JSON**:
   ```bash
   python scripts/inspect_dataset.py
   ```
4. **Chạy toàn bộ unit tests kiểm định Zero Leakage**:
   ```bash
   pytest tests/test_data_integrity.py -v
   ```
5. **Thực thi và xem trực quan hóa Bounding Box**:
   ```bash
   jupyter nbconvert --to notebook --execute --inplace notebooks/01_bbox_exploration.ipynb
   ```
