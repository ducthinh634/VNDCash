# GIAO THỨC DỮ LIỆU & PIPELINE (DATASET PROTOCOL)
**Dự án**: XÂY DỰNG ỨNG DỤNG NHẬN DIỆN TIỀN GIẤY VIỆT NAM BẰNG CNN (VNDCash)  
**Mã công việc**: Stack01 - Data Engineering & Data Pipeline  
**Thành viên phụ trách (Nhóm A)**: Nguyễn Đức Thịnh, Lê Quang Minh Nhật, Trương Tấn  
**Ngày hoàn thiện**: 15/10/2026  

---

## 1. Nguồn gốc dữ liệu & Phạm vi bài toán

### 1.1. Nguồn dữ liệu
Bộ dữ liệu được thu thập và đồng bộ trực tiếp từ KaggleHub:
```python
import kagglehub
path = kagglehub.dataset_download("nguyentrongdai/vietnamese-currency")
```
- Dữ liệu gốc bao gồm 12 thư mục với tổng cộng **2.712 ảnh**.
- Định dạng ảnh gốc: `.png`, kích thước chuẩn `192x144` pixels, hệ màu `RGB`.

### 1.2. Phạm vi 09 mệnh giá mục tiêu
Dự án chốt phạm vi nhận diện tự động **09 mệnh giá tiền giấy Việt Nam** đang lưu hành hợp pháp:
- **Tiền giấy Cotton**: `1.000 VND`, `2.000 VND`, `5.000 VND`
- **Tiền Polymer**: `10.000 VND`, `20.000 VND`, `50.000 VND`, `100.000 VND`, `200.000 VND`, `500.000 VND`
- **Quy tắc gán nhãn**: Mặt trước và mặt sau của cùng một tờ tiền đều được quy về cùng một nhãn mệnh giá.

### 1.3. Xử lý các lớp ngoài phạm vi (Excluded Folders)
Trong dữ liệu gốc từ Kaggle có 3 thư mục không thuộc phạm vi bài toán 9 mệnh giá:
- `000000` (250 ảnh): Nền / ảnh không phải tiền (Background / Other)
- `000200` (250 ảnh): Mệnh giá 200 VND (tiền cotton cũ, ít phát hành và lưu thông)
- `000500` (250 ảnh): Mệnh giá 500 VND (ngoài phạm vi 9 mệnh giá 1k-500k)

> **Quy tắc**: Toàn bộ 750 ảnh từ 3 thư mục này được giữ nguyên trong `data/raw/` nhưng loại khỏi tập huấn luyện 9 lớp chính, có thể dùng làm dữ liệu kiểm thử robustness/out-of-distribution (OOD) sau này.

---

## 2. Kết quả kiểm toán dữ liệu (Data Audit)

Chạy script kiểm toán:
```bash
python scripts/inspect_dataset.py
```

### Kết quả kiểm tra chất lượng ảnh:
- **Số ảnh lỗi (Corrupted)**: `0` ảnh.
- **Số ảnh rỗng/trắng phẳng (Blank / Zero variance)**: `0` ảnh.
- **Số ảnh trùng lặp nhị phân (Exact MD5 Duplicates)**: `0` ảnh.
- **Tổng số ảnh thuộc 9 mệnh giá mục tiêu**: **1.962 ảnh**.

### Thống kê và ghi nhận hiện tượng mất cân bằng mẫu:
| Mệnh giá | Thư mục gốc | Chất liệu | Số lượng ảnh | Tỷ lệ so với lớp chuẩn (250 ảnh) | Đánh giá & Xử lý |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **1.000 VND** | `001000` | Cotton | 250 | 100% | Đủ mẫu chuẩn |
| **2.000 VND** | `002000` | Cotton | 250 | 100% | Đủ mẫu chuẩn |
| **5.000 VND** | `005000` | Cotton | **154** | **61.6%** | **Thiếu 96 ảnh (-38.4%)** |
| **10.000 VND** | `010000` | Polymer | 250 | 100% | Đủ mẫu chuẩn |
| **20.000 VND** | `020000` | Polymer | **154** | **61.6%** | **Thiếu 96 ảnh (-38.4%)** |
| **50.000 VND** | `050000` | Polymer | **154** | **61.6%** | **Thiếu 96 ảnh (-38.4%)** |
| **100.000 VND** | `100000` | Polymer | 250 | 100% | Đủ mẫu chuẩn |
| **200.000 VND** | `200000` | Polymer | 250 | 100% | Đủ mẫu chuẩn |
| **500.000 VND** | `500000` | Polymer | 250 | 100% | Đủ mẫu chuẩn |

> **Ghi chú nghiệm thu Thành viên A**: Ba lớp `5.000`, `20.000`, và `50.000` chỉ có 154 ảnh/lớp. Pipeline đã ghi nhận rõ rệt sự mất cân bằng này. Trong quá trình training (Thành viên B), khuyến nghị áp dụng **Class Weights** hoặc **Focal Loss** để cân bằng trọng số giữa các lớp.

---

## 3. Cấu trúc Metadata (`data/metadata/metadata.csv`)

File metadata được sinh tự động bằng script `scripts/build_metadata.py` với schema nhất quán:

| Tên cột | Kiểu dữ liệu | Ý nghĩa | Ví dụ |
| :--- | :--- | :--- | :--- |
| `image_id` | String | Mã định danh duy nhất của ảnh | `1000_0000` |
| `filename` | String | Tên file vật lý | `001000_0.png` |
| `rel_path` | String | Đường dẫn tương đối từ gốc project | `data/raw/001000/001000_0.png` |
| `denomination`| String/Int | Mệnh giá tiền | `1000` |
| `label_idx` | Integer | Chỉ số lớp dùng trong classification head (0 - 8) | `0` |
| `material` | String | Chất liệu tờ tiền (`cotton` hoặc `polymer`) | `cotton` |
| `note_id` | String | **Mã tờ tiền vật lý** (phục vụ Grouped Split) | `NOTE_1000_000` |
| `width` | Integer | Chiều rộng ảnh (pixels) | `192` |
| `height` | Integer | Chiều cao ảnh (pixels) | `144` |
| `channels` | Integer | Số kênh màu | `3` |
| `color_mode`| String | Hệ màu | `RGB` |
| `aspect_ratio`| Float | Tỷ lệ khung hình (width / height) | `1.333` |
| `file_size_bytes` | Integer | Kích thước file (bytes) | `52550` |
| `md5_hash` | String | Mã băm MD5 kiểm tra trùng lặp | `9b61b958bf3295f3...` |
| `is_valid` | Boolean | Cờ trạng thái hợp lệ | `True` |

---

## 4. Quản lý `note_id` & Giao thức chia tập chống rò rỉ dữ liệu (Anti-Leakage Split)

### 4.1. Nguy cơ rò rỉ dữ liệu (Data Leakage)
Nếu thực hiện random split ngẫu nhiên theo từng ảnh đơn lẻ, các ảnh chụp cùng một tờ tiền (cùng số serial, cùng độ nhàu nát hoặc nền chụp) sẽ xuất hiện đồng thời ở cả tập Train và Test. Điều này khiến mô hình "học vẹt" đặc trưng riêng của tờ tiền thay vì học đặc trưng tổng quát của mệnh giá, dẫn đến độ chính xác ảo (over-optimistic test metrics).

### 4.2. Giải pháp: Stratified Grouped Split theo `note_id`
- Gom cụm các ảnh chụp cùng một tờ tiền vào chung một `note_id`. Toàn bộ bộ dữ liệu gồm **198 tờ tiền độc lập**.
- Thuật toán `stratified_group_split`:
  1. Phân nhóm theo mệnh giá (`denomination`).
  2. Với mỗi mệnh giá, xáo trộn danh sách `note_id` với seed cố định (`42`).
  3. Phân bổ nguyên vẹn từng `note_id` vào tập Train, Val, hoặc Test.
  4. Ánh xạ split ngược lại toàn bộ ảnh thuộc tờ tiền đó.

### 4.3. Kết quả phân chia tập dữ liệu (`data/metadata/splits.csv`)
- **Tập Train**: **1.362 ảnh** (69.42%) | 138 tờ tiền
- **Tập Val**: **300 ảnh** (15.29%) | 30 tờ tiền
- **Tập Test**: **300 ảnh** (15.29%) | 30 tờ tiền
- **Tổng cộng**: **1.962 ảnh** (100.00%) | 198 tờ tiền

#### Chi tiết phân bổ theo mệnh giá:
| Mệnh giá | Train (ảnh / notes) | Val (ảnh / notes) | Test (ảnh / notes) | Tổng ảnh | Tỷ lệ thực tế (Train/Val/Test) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1.000** | 170 / 17 | 40 / 4 | 40 / 4 | 250 | 68.0% / 16.0% / 16.0% |
| **2.000** | 170 / 17 | 40 / 4 | 40 / 4 | 250 | 68.0% / 16.0% / 16.0% |
| **5.000** | 114 / 12 | 20 / 2 | 20 / 2 | 154 | 74.0% / 13.0% / 13.0% |
| **10.000** | 170 / 17 | 40 / 4 | 40 / 4 | 250 | 68.0% / 16.0% / 16.0% |
| **20.000** | 114 / 12 | 20 / 2 | 20 / 2 | 154 | 74.0% / 13.0% / 13.0% |
| **50.000** | 114 / 12 | 20 / 2 | 20 / 2 | 154 | 74.0% / 13.0% / 13.0% |
| **100.000** | 170 / 17 | 40 / 4 | 40 / 4 | 250 | 68.0% / 16.0% / 16.0% |
| **200.000** | 170 / 17 | 40 / 4 | 40 / 4 | 250 | 68.0% / 16.0% / 16.0% |
| **500.000** | 170 / 17 | 40 / 4 | 40 / 4 | 250 | 68.0% / 16.0% / 16.0% |

> **Kết quả nghiệm thu chống rò rỉ**:
> - Giao giữa Train & Val: $\emptyset$ (0 notes, 0 hash trùng)
> - Giao giữa Train & Test: $\emptyset$ (0 notes, 0 hash trùng)
> - Giao giữa Val & Test: $\emptyset$ (0 notes, 0 hash trùng)  
> **-> Đạt 100% tiêu chí Zero Data Leakage.**

---

## 5. Quy tắc Tiền xử lý & Tăng cường dữ liệu

### 5.1. Tiền xử lý chuẩn (Preprocessing)
- **Kích thước đầu vào chuẩn**: `224 x 224 x 3` (phù hợp với các backbone CNN: Baseline CNN, MobileNetV2, ResNet18).
- **Chuẩn hóa điểm ảnh (ImageNet Normalization)**:
  - Mean: `[0.485, 0.456, 0.406]`
  - Std: `[0.229, 0.224, 0.225]`

### 5.2. Tăng cường dữ liệu (Data Augmentation)
> [!IMPORTANT]
> **Quy tắc bất biến**: Data Augmentation **CHỈ** được áp dụng cho tập `Train`. Tuyệt đối không áp dụng cho `Val` và `Test`.

Các phép biến đổi tập Train (trong `src/data/augmentation.py`):
1. **Lật ngang ngẫu nhiên (`RandomHorizontalFlip`, $p=0.5$)**: Mô phỏng việc người dùng lật chiều tờ tiền.
2. **Xoay ngẫu nhiên (`RandomRotation`, $\pm 15^\circ$)**: Mô phỏng góc nghiêng tự nhiên khi cầm tiền trước camera.
3. **Biến đổi ánh sáng/màu sắc (`ColorJitter`, brightness=0.2, contrast=0.2, saturation=0.2)**: Mô phỏng điều kiện ánh sáng bóng đèn, ánh sáng ban ngày và độ cũ/mới của tờ tiền.
4. **Co giãn & dịch chuyển ngẫu nhiên (`RandomAffine`, scale 0.9 - 1.1, translate 5%)**: Mô phỏng khoảng cách xa gần từ camera tới tờ tiền.
5. **Làm mờ Gauss (`GaussianBlur`, $p=0.2$)**: Mô phỏng rung tay khi chụp hoặc camera lấy nét chưa chuẩn.

---

## 6. Hướng dẫn sử dụng & Tái lập (CLI Commands)

1. **Kiểm tra và audit dataset**:
   ```bash
   python scripts/inspect_dataset.py
   ```
2. **Xây dựng bảng metadata chuẩn**:
   ```bash
   python scripts/build_metadata.py
   ```
3. **Phân chia tập Train/Val/Test có kiểm soát note_id**:
   ```bash
   python scripts/create_split.py
   ```
4. **Chạy bộ kiểm thử tự động (Unit Tests)**:
   ```bash
   python -m pytest tests/test_dataset.py tests/test_preprocessing.py tests/test_split.py -v
   ```
5. **Nạp DataLoader trong code huấn luyện (Dành cho Thành viên B)**:
   ```python
   from src.data.dataset import get_dataloaders

   train_loader, val_loader, test_loader = get_dataloaders(
       config_path="configs/data.yaml",
       batch_size=32,
       num_workers=0
   )
   ```
