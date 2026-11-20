# VNDCash - Nhận Diện Tiền Giấy Việt Nam (YOLOv11 & CNN)

Hệ thống phát hiện, định vị và nhận diện đồng thời nhiều tờ tiền giấy Việt Nam (9 mệnh giá từ 1.000 VND đến 500.000 VND) bằng mô hình YOLOv11.

## Nguồn Dữ Liệu & Bản Quyền (Dataset & Attribution)
- **Nguồn dữ liệu chính**: Bộ dữ liệu của tác giả **Anng** trên nền tảng **[Roboflow Universe]**: [Vietnamese Currency.v10-v4.yolov11](https://universe.roboflow.com/anng-ismb4/vietnamese-currency-lgi9i-sl5x1).
- **Đặc trưng tập dữ liệu**: Cung cấp gần **2.400 hình ảnh gốc độc lập** (hơn 3.800 ảnh sau khi tăng cường) với nhiều góc chụp đa dạng khác nhau của 9 mệnh giá tiền giấy Việt Nam (chụp nghiêng, xoay góc, xếp chồng, che khuất một phần, gập nếp, đa mệnh giá trong một khung hình).
- **Giấy phép bản quyền (License)**: Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
- **Tập dữ liệu chuẩn hóa**: Được chọn lọc **1.898 ảnh đa tờ tiền thực tế** ($\ge 2$ tờ/ảnh) với **6.731 Bounding Box** chuẩn hóa theo chuẩn YOLOv11.

## Kiểm thử dữ liệu (Stack 01)
```bash
pytest tests/test_data_integrity.py -v
```
*(7/7 tests passed - Cam kết 100% Zero Data Leakage qua phân nhóm `note_id`).*
