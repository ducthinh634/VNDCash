# VNDCash - Nhận Diện Tiền Giấy Việt Nam (YOLOv11 & CNN)

Hệ thống phát hiện, định vị và nhận diện đồng thời nhiều tờ tiền giấy Việt Nam (9 mệnh giá từ 1.000 VND đến 500.000 VND) bằng mô hình YOLOv11.

## Nguồn Dữ Liệu & Bản Quyền (Dataset & Attribution)
- **Bộ dữ liệu v2 (Đa tờ tiền YOLOv11)**: Tác giả **Anng** trên nền tảng **[Roboflow Universe]**: [Vietnamese Currency.v10-v4.yolov11](https://universe.roboflow.com/anng-ismb4/vietnamese-currency-lgi9i-sl5x1). Cung cấp gần 2.400 hình ảnh gốc độc lập (hơn 3.800 ảnh tăng cường) với đa dạng góc chụp của 9 mệnh giá tiền giấy Việt Nam (gập nếp, chụp nghiêng, xếp chồng, đa mệnh giá đồng thời). Giấy phép: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Bộ dữ liệu v1 (Tiền đơn lẻ tiền đề)**: Tác giả **Nguyễn Trọng Đại** trên **Kaggle**: [Vietnamese Currency](https://www.kaggle.com/datasets/nguyentrongdai/vietnamese-currency) (2.712 ảnh tĩnh 9 mệnh giá). Dùng làm baseline đối chứng và nạp vào pipeline gán nhãn bán tự động OpenCV.
- **Tập dữ liệu chuẩn hóa (Stack 01)**: Trích chọn **1.898 ảnh đa tờ tiền thực tế** ($\ge 2$ tờ/ảnh) với **6.731 Bounding Box** chuẩn hóa theo chuẩn YOLOv11.

## Kiểm thử dữ liệu (Stack 01)
```bash
pytest tests/test_data_integrity.py -v
```
*(7/7 tests passed - Cam kết 100% Zero Data Leakage qua phân nhóm `note_id`).*
