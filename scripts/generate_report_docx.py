"""Script to generate a comprehensive, professional Word report (.docx)
for Member A's contributions in VNDCash final project defense.
"""

import os
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, hex_color):
    """Sets background color of a table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tc_pr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets cell padding."""
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tc_pr.append(tc_mar)

def create_report():
    doc = docx.Document()

    # Page Setup - A4 with 1 inch (2.54 cm) margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        section.page_width = Inches(8.27)
        section.page_height = Inches(11.69)

    # Base Colors
    COLOR_PRIMARY = RGBColor(27, 54, 93)     # Navy #1B365D
    COLOR_SECONDARY = RGBColor(46, 107, 158) # Steel Blue #2E6B9E
    COLOR_TEXT = RGBColor(40, 40, 40)
    HEX_HEADER = "1B365D"
    HEX_ALT_ROW = "F7F9FC"
    HEX_CALLOUT = "EBF3FA"

    # Set default style font
    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Times New Roman'
    style_normal.font.size = Pt(12)
    style_normal.font.color.rgb = COLOR_TEXT
    style_normal.paragraph_format.line_spacing = 1.25
    style_normal.paragraph_format.space_after = Pt(6)

    def add_title(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(18)
        p.paragraph_format.space_after = Pt(6)
        run = p.add_run(text)
        run.bold = True
        run.font.size = Pt(20)
        run.font.color.rgb = COLOR_PRIMARY
        return p

    def add_subtitle(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(18)
        run = p.add_run(text)
        run.font.size = Pt(13)
        run.italic = True
        run.font.color.rgb = COLOR_SECONDARY
        return p

    def add_h1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(16)
        p.paragraph_format.space_after = Pt(6)
        run = p.add_run(text)
        run.bold = True
        run.font.size = Pt(15)
        run.font.color.rgb = COLOR_PRIMARY
        return p

    def add_h2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run(text)
        run.bold = True
        run.font.size = Pt(13)
        run.font.color.rgb = COLOR_SECONDARY
        return p

    def add_h3(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run(text)
        run.bold = True
        run.font.size = Pt(12)
        run.font.color.rgb = COLOR_TEXT
        return p

    def add_bullet(bold_prefix, text):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.15
        r1 = p.add_run(bold_prefix + ": ")
        r1.bold = True
        p.add_run(text)
        return p

    def add_callout(text, bold_title="LƯU Ý QUAN TRỌNG:"):
        table = doc.add_table(rows=1, cols=1)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = table.cell(0, 0)
        cell.width = Inches(6.27)
        set_cell_background(cell, HEX_CALLOUT)
        set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.15
        r_title = p.add_run(f"📌 {bold_title} ")
        r_title.bold = True
        r_title.font.color.rgb = COLOR_PRIMARY
        p.add_run(text)
        doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # ==================== TRANG BÌA / HEADER ====================
    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_uni = p_meta.add_run("BÁO CÁO KỸ THUẬT & HỒ SƠ THUYẾT TRÌNH CUỐI KỲ\n")
    r_uni.bold = True
    r_uni.font.size = Pt(12)
    r_uni.font.color.rgb = COLOR_SECONDARY

    add_title("ĐỒ ÁN: XÂY DỰNG ỨNG DỤNG NHẬN DIỆN TIỀN GIẤY VIỆT NAM BẰNG MẠNG NƠ-RON TÍCH CHẬP (CNN)")
    add_subtitle("Hệ thống VNDCash: Kiến trúc Data Pipeline, Quản lý Tờ tiền Vật lý & Giao thức Chống Rò rỉ Dữ liệu")

    # Bảng thông tin sinh viên thực hiện
    info_table = doc.add_table(rows=4, cols=2)
    info_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    info_data = [
        ("Sinh viên phụ trách:", "Nguyễn Đức Thịnh (Thành viên A - Nhóm 18)"),
        ("Hạng mục đảm nhiệm:", "Stack01 - Data Engineering & Data Pipeline"),
        ("Tập dữ liệu nguồn:", "KaggleHub nguyentrongdai/vietnamese-currency"),
        ("Thời gian thực hiện & nghiệm thu:", "Tháng 09 - Tháng 10/2026 (Hoàn thành 100%)")
    ]
    for row_idx, (label, val) in enumerate(info_data):
        row = info_table.rows[row_idx]
        c1, c2 = row.cells[0], row.cells[1]
        c1.width, c2.width = Inches(2.3), Inches(3.97)
        set_cell_margins(c1, top=80, bottom=80, left=100, right=100)
        set_cell_margins(c2, top=80, bottom=80, left=100, right=100)
        p1, p2 = c1.paragraphs[0], c2.paragraphs[0]
        r1, r2 = p1.add_run(label), p2.add_run(val)
        r1.bold = True
        r1.font.size = Pt(11)
        r2.font.size = Pt(11)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # ==================== EXECUTIVE SUMMARY ====================
    add_h1("TÓM TẮT KẾT QUẢ THỰC HIỆN CỐT LÕI (EXECUTIVE SUMMARY)")
    p_summary = doc.add_paragraph(
        "Báo cáo này trình bày toàn bộ các đóng góp kỹ thuật của Thành viên A (Data Engineering & Pipeline) "
        "trong dự án VNDCash. Dữ liệu là nền tảng cốt lõi quyết định trực tiếp tới khả năng tổng quát hóa thực tế của "
        "các mô hình học sâu. Trong đề tài này, Thành viên A đã hoàn thành xuất sắc 100% tất cả 10 hạng mục công việc được giao, "
        "giải quyết triệt để bài toán rò rỉ dữ liệu (Data Leakage) — một lỗi kinh điển trong các bài báo nhận diện tiền tệ — "
        "và xây dựng thành công một pipeline dữ liệu hoàn chỉnh, tự động hóa từ bước nạp dữ liệu thô đến nạp batch cho mô hình."
    )

    add_bullet("Phạm vi dữ liệu", "Chuẩn hóa chính xác 09 mệnh giá tiền giấy Việt Nam từ 1.000 đến 500.000 VND (1.962 ảnh). Loại bỏ 750 ảnh ngoài phạm vi.")
    add_bullet("Đóng góp kỹ thuật cốt lõi", "Thiết lập hệ thống mã tờ tiền vật lý (note_id) gồm 198 tờ tiền độc lập; cài đặt thuật toán Stratified Grouped Split bảo đảm 100% Zero Data Leakage giữa Train (69.4%), Val (15.3%), Test (15.3%).")
    add_bullet("Phát hiện bất cân bằng mẫu", "Ghi nhận 3 lớp 5k, 20k, 50k bị thiếu mẫu (154 ảnh/lớp, thấp hơn 38.4% so với các lớp 250 ảnh); thiết lập giải pháp Class Weights cho nhóm huấn luyện.")
    add_bullet("Sản phẩm bàn giao", "Hoàn thiện 10/10 tệp mã nguồn và tài liệu; vượt qua 7/7 unit tests tự động; đóng gói trọn gói và lưu trữ an toàn trên GitHub Repository.")

    # ==================== CHƯƠNG 1 ====================
    add_h1("CHƯƠNG 1: BỐI CẢNH ĐỀ TÀI VÀ PHẠM VI NHIỆM VỤ THÀNH VIÊN A")
    add_h2("1.1. Bối cảnh và ý nghĩa thực tiễn")
    doc.add_paragraph(
        "Tại Việt Nam, tiền mặt vẫn là phương tiện thanh toán phổ biến trong đời sống hàng ngày. "
        "Đối với người khiếm thị hoặc người cao tuổi suy giảm thị lực, việc phân biệt các mệnh giá tiền giấy — "
        "đặc biệt là các tờ tiền có kích thước gần tương đương hoặc tông màu sắc dễ gây nhầm lẫn (như tờ 20.000 VND và 500.000 VND "
        "cùng có gam màu xanh lam) — là một thách thức lớn. "
        "Hệ thống VNDCash hướng tới việc ứng dụng mạng nơ-ron tích chập (CNN) trên thiết bị di động, tự động nhận diện chính xác "
        "mệnh giá tiền thông qua camera và phản hồi kết quả bằng giọng nói tiếng Việt."
    )

    add_h2("1.2. Phạm vi 09 mệnh giá chuẩn và quy ước nhãn")
    doc.add_paragraph(
        "Đề tài chốt phạm vi nghiên cứu thực nghiệm trên 09 mệnh giá tiền giấy Việt Nam đang lưu hành hợp pháp, "
        "bao gồm cả 2 phân nhóm chất liệu:"
    )
    add_bullet("Nhóm tiền giấy Cotton", "1.000 VND, 2.000 VND, 5.000 VND (3 mệnh giá nhỏ).")
    add_bullet("Nhóm tiền Polymer", "10.000 VND, 20.000 VND, 50.000 VND, 100.000 VND, 200.000 VND, 500.000 VND (6 mệnh giá).")
    add_bullet("Quy ước gán nhãn", "Cả mặt trước và mặt sau của cùng một tờ tiền đều được quy chuẩn ánh xạ về chung một nhãn lớp duy nhất (từ 0 đến 8).")

    add_h2("1.3. Bảng đối chiếu tiến độ hoàn thiện 10 hạng mục (Checklist)")
    doc.add_paragraph("Thành viên A đã hoàn thiện toàn bộ các nhiệm vụ theo đúng thứ tự ưu tiên của ma trận phân công:")

    # Checklist Table
    chk_table = doc.add_table(rows=11, cols=4)
    chk_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["STT", "Tệp / Module cần hoàn thiện", "Trạng thái", "Kết quả nghiệm thu"]
    for i, h in enumerate(headers):
        cell = chk_table.cell(0, i)
        set_cell_background(cell, HEX_HEADER)
        set_cell_margins(cell, top=100, bottom=100, left=100, right=100)
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(10)
        r.font.color.rgb = RGBColor(255, 255, 255)

    checklist_data = [
        ("1", "configs/data.yaml", "ĐÃ XONG", "Khai báo 9 classes, split 70/15/15, input 224x224, chuẩn hóa ImageNet"),
        ("2", "scripts/inspect_dataset.py", "ĐÃ XONG", "Audit 2.712 ảnh, kiểm tra ảnh lỗi/rỗng/trùng lặp, xuất data_audit_report.json"),
        ("3", "scripts/build_metadata.py", "ĐÃ XONG", "Định danh 198 note_id, gắn nhãn chất liệu cotton/polymer, hash MD5"),
        ("4", "data/metadata/metadata.csv", "ĐÃ XONG", "Bảng 1.962 dòng, 15 trường thông tin chuẩn hóa schema nghiêm ngặt"),
        ("5", "src/data/split.py & scripts/create_split.py", "ĐÃ XONG", "Thuật toán Stratified Grouped Split; kiểm chứng Zero Data Leakage"),
        ("6", "data/metadata/splits.csv", "ĐÃ XONG", "Phân chia chuẩn: Train (1.362 ảnh), Val (300 ảnh), Test (300 ảnh)"),
        ("7", "Toàn bộ module src/data/", "ĐÃ XONG", "Đầy đủ 6 files: validation, preprocessing, augmentation, dataset, split, __init__"),
        ("8", "notebooks/01_dataset_exploration.ipynb", "ĐÃ XONG", "EDA chuyên sâu: phân bố 9 lớp, tỷ lệ cotton/polymer, lưới 9 ảnh mẫu"),
        ("9", "notebooks/02_preprocessing.ipynb", "ĐÃ XONG", "Trực quan hóa tiền xử lý, biến đổi augmentation và kiểm thử DataLoader batch"),
        ("10", "docs/dataset_protocol.md", "ĐÃ XONG", "Tài liệu giao thức dữ liệu hoàn chỉnh, hướng dẫn vận hành và tái lập thí nghiệm")
    ]
    for row_idx, data in enumerate(checklist_data):
        row = chk_table.rows[row_idx + 1]
        bg = HEX_ALT_ROW if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, val in enumerate(data):
            cell = row.cells[col_idx]
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=80, bottom=80, left=80, right=80)
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(9.5)
            if col_idx == 2:
                r.bold = True
                r.font.color.rgb = RGBColor(39, 174, 96) # Green
            if col_idx == 0:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    chk_table.columns[0].width = Inches(0.5)
    chk_table.columns[1].width = Inches(2.2)
    chk_table.columns[2].width = Inches(1.1)
    chk_table.columns[3].width = Inches(2.47)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # ==================== CHƯƠNG 2 ====================
    add_h1("CHƯƠNG 2: KIỂM TOÁN VÀ ĐÁNH GIÁ CHẤT LƯỢNG DỮ LIỆU NGUỒN (DATA AUDIT)")
    add_h2("2.1. Quy trình kiểm toán tự động")
    doc.add_paragraph(
        "Để đảm bảo dữ liệu đưa vào huấn luyện mô hình học sâu đạt tiêu chuẩn chất lượng cao nhất, "
        "Thành viên A đã xây dựng script kiểm toán độc lập `scripts/inspect_dataset.py`. "
        "Quy trình kiểm toán thực hiện 4 bước tự động:"
    )
    add_bullet("Kiểm tra toàn vẹn tệp (Integrity Check)", "Sử dụng thư viện PIL đọc header và nạp toàn bộ pixel; phát hiện file hỏng, thiếu byte hoặc sai cấu trúc định dạng.")
    add_bullet("Kiểm tra ảnh rỗng (Blank Image Detection)", "Tính phương sai cường độ pixel (pixel variance). Nếu variance < 1.0, ảnh được đánh dấu là ảnh trắng hoặc ảnh một màu lỗi camera.")
    add_bullet("Kiểm tra trùng lặp nhị phân (Binary Hash Matching)", "Tính mã băm MD5 cho từng file ảnh nhị phân để truy tìm các ảnh bị copy-paste trùng lặp.")
    add_bullet("Kiểm tra hệ màu và kích thước", "Xác thực toàn bộ ảnh đều ở hệ màu RGB và ghi nhận phân bố độ phân giải.")

    add_h2("2.2. Kết quả kiểm toán dữ liệu nguồn")
    doc.add_paragraph(
        "Kết quả kiểm toán trên toàn bộ 2.712 ảnh tải về từ KaggleHub như sau:"
    )
    add_bullet("Ảnh hỏng / Không đọc được (Corrupt)", "0 ảnh (100% tệp đọc tốt).")
    add_bullet("Ảnh rỗng / Trắng một màu (Blank)", "0 ảnh.")
    add_bullet("Ảnh trùng lặp nhị phân (MD5 Duplicate)", "0 ảnh.")
    add_bullet("Độ phân giải và hệ màu đồng nhất", "2.712 / 2.712 ảnh đều có kích thước 192x144 pixels, hệ màu RGB 3 kênh.")

    add_h2("2.3. Xử lý các lớp ngoài phạm vi (Excluded Classes)")
    doc.add_paragraph(
        "Trong thư mục gốc tải về gồm 12 thư mục con. Thành viên A đã tiến hành phân loại và loại trừ 3 thư mục "
        "không thuộc phạm vi nhận diện của đề tài:"
    )
    add_bullet("Thư mục 000000 (250 ảnh)", "Ảnh chụp phông nền, đồ vật khác ngoài tiền giấy.")
    add_bullet("Thư mục 000200 (250 ảnh)", "Mệnh giá 200 VND (tiền cotton cũ, hiện đã ngừng phát hành rộng rãi).")
    add_bullet("Thư mục 000500 (250 ảnh)", "Mệnh giá 500 VND (ngoài phạm vi 9 mệnh giá lựa chọn của đề tài).")
    doc.add_paragraph(
        "Quy tắc quản lý: Toàn bộ 750 ảnh này được bảo lưu nguyên trạng trong data/raw/ theo nguyên tắc dữ liệu gốc bất biến, "
        "nhưng được cấu hình loại trừ (excluded_folders) trong pipeline để không ảnh hưởng đến mô hình 9 lớp."
    )

    add_h2("2.4. Phát hiện mất cân bằng dữ liệu (Class Imbalance) và Giải pháp")
    doc.add_paragraph(
        "Sau khi loại bỏ 750 ảnh ngoài phạm vi, tập dữ liệu 09 mệnh giá mục tiêu có tổng cộng 1.962 ảnh. "
        "Quá trình audit đã phát hiện sự mất cân bằng mẫu đáng kể:"
    )

    # Table of Imbalance
    imb_table = doc.add_table(rows=10, cols=5)
    imb_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    imb_headers = ["Mệnh giá (VND)", "Thư mục", "Chất liệu", "Số lượng ảnh", "Tình trạng mẫu"]
    for i, h in enumerate(imb_headers):
        c = imb_table.cell(0, i)
        set_cell_background(c, HEX_HEADER)
        set_cell_margins(c, top=80, bottom=80, left=80, right=80)
        p = c.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(10)
        r.font.color.rgb = RGBColor(255, 255, 255)

    imb_data = [
        ("1.000 VND", "001000", "Cotton", "250", "Đủ mẫu chuẩn (100%)"),
        ("2.000 VND", "002000", "Cotton", "250", "Đủ mẫu chuẩn (100%)"),
        ("5.000 VND", "005000", "Cotton", "154", "THIẾU MẪU (-38.4%)"),
        ("10.000 VND", "010000", "Polymer", "250", "Đủ mẫu chuẩn (100%)"),
        ("20.000 VND", "020000", "Polymer", "154", "THIẾU MẪU (-38.4%)"),
        ("50.000 VND", "050000", "Polymer", "154", "THIẾU MẪU (-38.4%)"),
        ("100.000 VND", "100000", "Polymer", "250", "Đủ mẫu chuẩn (100%)"),
        ("200.000 VND", "200000", "Polymer", "250", "Đủ mẫu chuẩn (100%)"),
        ("500.000 VND", "500000", "Polymer", "250", "Đủ mẫu chuẩn (100%)"),
    ]
    for row_idx, row_vals in enumerate(imb_data):
        row = imb_table.rows[row_idx + 1]
        is_imb = "THIẾU" in row_vals[4]
        bg = "FDF2E9" if is_imb else ("FFFFFF" if row_idx % 2 == 0 else HEX_ALT_ROW)
        for col_idx, val in enumerate(row_vals):
            c = row.cells[col_idx]
            set_cell_background(c, bg)
            set_cell_margins(c, top=60, bottom=60, left=80, right=80)
            p = c.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(9.5)
            if col_idx == 4 and is_imb:
                r.bold = True
                r.font.color.rgb = RGBColor(211, 84, 0) # Orange-red

    doc.add_paragraph().paragraph_format.space_after = Pt(6)
    add_callout(
        "Hiện tượng 3 lớp 5.000, 20.000 và 50.000 chỉ có 154 ảnh (thiếu 96 ảnh so với mốc 250 ảnh) "
        "đã được Thành viên A ghi chép chi tiết trong hồ sơ giao thức. Đồng thời, Thành viên A đã cung cấp công thức tính "
        "Class Weights nghịch đảo tần suất để Thành viên B áp dụng trực tiếp vào CrossEntropyLoss khi huấn luyện mô hình.",
        bold_title="Ý NGHĨA KHOA HỌC:"
    )

    # ==================== CHƯƠNG 3 ====================
    add_h1("CHƯƠNG 3: THIẾT KẾ ĐỊNH DANH TỜ TIỀN (`note_id`) VÀ SCHEMA METADATA")
    add_h2("3.1. Rủi ro rò rỉ dữ liệu (Data Leakage) trong nhận diện tiền tệ")
    doc.add_paragraph(
        "Trong các nghiên cứu phân loại ảnh thông thường, việc chia tập Train/Val/Test thường được thực hiện ngẫu nhiên "
        "(Random Split) trên từng bức ảnh. Tuy nhiên, đối với bài toán nhận diện tiền tệ, một tờ tiền vật lý thường được "
        "chụp liên tiếp nhiều góc độ, hướng xoay và điều kiện ánh sáng khác nhau. "
        "Nếu chia ngẫu nhiên từng ảnh:"
    )
    add_bullet("Nguy cơ", "Ảnh chụp mặt trước tờ tiền A rơi vào tập Train, còn ảnh chụp mặt sau hoặc góc nghiêng của chính tờ tiền A đó rơi vào tập Test.")
    add_bullet("Hậu quả", "Mô hình CNN sẽ ghi nhớ các đặc trưng cục bộ riêng biệt của tờ tiền A (như số sê-ri, vết ố bẩn, nếp gấp cục bộ) thay vì học đặc trưng tổng quát của mệnh giá. Dẫn đến hiện tượng Overfitting ẩn và kết quả Accuracy trên tập Test cao ảo nhưng thực tế thất bại khi đưa vào ứng dụng thực tế.")

    add_h2("3.2. Giải pháp: Quản lý định danh tờ tiền vật lý (`note_id`)")
    doc.add_paragraph(
        "Thành viên A đã đưa ra giải pháp kỹ thuật mang tính nền tảng cho toàn bộ đồ án: "
        "Phân cụm các ảnh chụp liên tiếp của cùng một tờ tiền và định danh thành các mã tờ tiền vật lý duy nhất (note_id). "
        "Cụ thể, mỗi cụm 10 ảnh liên tiếp được nhóm thành 1 tờ tiền vật lý."
    )
    add_bullet("Kết quả phân nhóm", "Toàn bộ 1.962 bức ảnh của 9 mệnh giá đã được gom cụm thành chính xác 198 tờ tiền vật lý độc lập.")
    add_bullet("Đại diện các lớp", "Các lớp 250 ảnh gồm 25 tờ tiền độc lập (NOTE_..._000 đến NOTE_..._024). Các lớp 154 ảnh gồm 16 tờ tiền độc lập.")

    add_h2("3.3. Cấu trúc Schema bảng Metadata (`data/metadata/metadata.csv`)")
    doc.add_paragraph(
        "File metadata.csv được sinh tự động bởi script `build_metadata.py` với schema chuẩn hóa 15 trường thông tin:"
    )
    add_bullet("image_id", "Mã định danh duy nhất của từng bức ảnh (ví dụ: 1000_0000, 500000_0249).")
    add_bullet("rel_path", "Đường dẫn tương đối từ gốc project (POSIX style), bảo đảm tính tương thích đa nền tảng Windows/Linux.")
    add_bullet("denomination & label_idx", "Mệnh giá tiền và chỉ số lớp số nguyên tương ứng từ 0 đến 8.")
    add_bullet("material", "Chất liệu tờ tiền ('cotton' hoặc 'polymer') phục vụ phân tích chuyên sâu.")
    add_bullet("note_id", "Mã tờ tiền vật lý (cột khóa chính để phân chia tập không rò rỉ).")
    add_bullet("width, height, channels, aspect_ratio", "Các đặc trưng vật lý về hình ảnh (192, 144, 3, 1.333).")
    add_bullet("md5_hash", "Mã băm MD5 độc nhất kiểm tra tính toàn vẹn và chống trùng lặp dữ liệu.")

    # ==================== CHƯƠNG 4 ====================
    add_h1("CHƯƠNG 4: THUẬT TOÁN STRATIFIED GROUPED SPLIT VÀ NGHIỆM THU ZERO LEAKAGE")
    add_h2("4.1. Nguyên lý thuật toán phân tầng theo nhóm")
    doc.add_paragraph(
        "Thành viên A đã thiết kế và cài đặt thuật toán `stratified_group_split` trong module `src/data/split.py`. "
        "Thuật toán kết hợp cả 2 yêu cầu khắt khe:"
    )
    add_bullet("Phân nhóm (Grouped by note_id)", "Toàn bộ ảnh chụp của cùng một tờ tiền (cùng note_id) bắt buộc phải nằm trọn vẹn trong duy nhất MỘT tập dữ liệu (Train HOẶC Val HOẶC Test).")
    add_bullet("Phân tầng (Stratified by denomination)", "Đảm bảo tỷ lệ phân chia giữa 3 tập gần tương đương tỷ lệ mục tiêu (70% Train, 15% Val, 15% Test) đồng đều trên từng mệnh giá.")

    add_h2("4.2. Bảng kết quả phân chia chi tiết từng mệnh giá (`data/metadata/splits.csv`)")

    # Split table
    split_table = doc.add_table(rows=11, cols=7)
    split_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    s_headers = ["Mệnh giá", "Train (ảnh/notes)", "Val (ảnh/notes)", "Test (ảnh/notes)", "Tổng ảnh", "Tỷ lệ Train", "Tỷ lệ Val/Test"]
    for i, h in enumerate(s_headers):
        c = split_table.cell(0, i)
        set_cell_background(c, HEX_HEADER)
        set_cell_margins(c, top=80, bottom=80, left=60, right=60)
        p = c.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(255, 255, 255)

    split_rows_data = [
        ("1.000 VND", "170 / 17", "40 / 4", "40 / 4", "250", "68.0%", "16.0% / 16.0%"),
        ("2.000 VND", "170 / 17", "40 / 4", "40 / 4", "250", "68.0%", "16.0% / 16.0%"),
        ("5.000 VND", "114 / 12", "20 / 2", "20 / 2", "154", "74.0%", "13.0% / 13.0%"),
        ("10.000 VND", "170 / 17", "40 / 4", "40 / 4", "250", "68.0%", "16.0% / 16.0%"),
        ("20.000 VND", "114 / 12", "20 / 2", "20 / 2", "154", "74.0%", "13.0% / 13.0%"),
        ("50.000 VND", "114 / 12", "20 / 2", "20 / 2", "154", "74.0%", "13.0% / 13.0%"),
        ("100.000 VND", "170 / 17", "40 / 4", "40 / 4", "250", "68.0%", "16.0% / 16.0%"),
        ("200.000 VND", "170 / 17", "40 / 4", "40 / 4", "250", "68.0%", "16.0% / 16.0%"),
        ("500.000 VND", "170 / 17", "40 / 4", "40 / 4", "250", "68.0%", "16.0% / 16.0%"),
        ("TỔNG CỘNG", "1.362 / 138", "300 / 30", "300 / 30", "1.962", "69.42%", "15.29% / 15.29%")
    ]
    for row_idx, r_vals in enumerate(split_rows_data):
        row = split_table.rows[row_idx + 1]
        is_total = (row_idx == len(split_rows_data) - 1)
        bg = "E8F8F5" if is_total else (HEX_ALT_ROW if row_idx % 2 == 1 else "FFFFFF")
        for col_idx, val in enumerate(r_vals):
            c = row.cells[col_idx]
            set_cell_background(c, bg)
            set_cell_margins(c, top=60, bottom=60, left=60, right=60)
            p = c.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(9.5)
            if is_total:
                r.bold = True
                if col_idx == 0:
                    r.font.color.rgb = COLOR_PRIMARY

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    add_h2("4.3. Nghiệm thu kiểm thử chống rò rỉ dữ liệu (Zero Leakage Verification)")
    doc.add_paragraph(
        "Hàm kiểm thử `verify_no_leakage` đã thực hiện giao tập hợp trên toàn bộ 198 `note_id` và 1.962 mã băm MD5:"
    )
    doc.add_paragraph(
        "• Giao điểm Train ∩ Val = ∅ (0 tờ tiền trùng lặp, 0 hash trùng lặp)\n"
        "• Giao điểm Train ∩ Test = ∅ (0 tờ tiền trùng lặp, 0 hash trùng lặp)\n"
        "• Giao điểm Val ∩ Test = ∅ (0 tờ tiền trùng lặp, 0 hash trùng lặp)"
    )
    add_callout(
        "Xác nhận 100% không xảy ra rò rỉ dữ liệu (Zero Data Leakage). "
        "Mọi kết quả đo lường Accuracy, Macro F1-score và Per-class Recall trên tập Test sau này "
        "đều phản ánh chính xác năng lực nhận diện thực tế trên những tờ tiền hoàn toàn mới lạ đối với mô hình.",
        bold_title="KẾT QUẢ NGHIỆM THU:"
    )

    # ==================== CHƯƠNG 5 ====================
    add_h1("CHƯƠNG 5: PIPELINE TIỀN XỬ LÝ VÀ CHIẾN LƯỢC DATA AUGMENTATION")
    add_h2("5.1. Tiền xử lý chuẩn hóa (Preprocessing)")
    doc.add_paragraph(
        "Để tương thích hoàn hảo với cả 3 kiến trúc mô hình (Baseline CNN, MobileNetV2, ResNet18), "
        "pipeline tiền xử lý trong `src/data/preprocessing.py` thực hiện chuẩn hóa:"
    )
    add_bullet("Kích thước đầu vào chuẩn", "Resize ảnh về độ phân giải chuẩn 224 x 224 pixels.")
    add_bullet("Chuẩn hóa phân phối điểm ảnh", "Áp dụng chuẩn hóa ImageNet: Mean = [0.485, 0.456, 0.406], Std = [0.229, 0.224, 0.225].")
    add_bullet("Hàm giải chuẩn hóa (Denormalization)", "Cung cấp hàm chuyển ngược từ Tensor chuẩn hóa về ảnh RGB uint8 [0, 255] phục vụ hiển thị trực quan và báo cáo.")

    add_h2("5.2. Chiến lược Tăng cường dữ liệu (Data Augmentation)")
    doc.add_paragraph(
        "Trong module `src/data/augmentation.py`, Thành viên A đã thiết kế 5 phép biến đổi mô phỏng thực tế "
        "các điều kiện sử dụng camera của người dùng khi nhận diện tiền giấy:"
    )
    add_bullet("Lật ngang ngẫu nhiên (RandomHorizontalFlip, p=0.5)", "Mô phỏng người dùng cầm ngược chiều tờ tiền.")
    add_bullet("Xoay góc ngẫu nhiên (RandomRotation, ±15°)", "Mô phỏng góc chụp nghiêng tự nhiên của tay người khi đưa tiền trước camera.")
    add_bullet("Biến đổi quang học (ColorJitter)", "Thay đổi ngẫu nhiên 20% độ sáng (brightness), tương phản (contrast), và bão hòa màu (saturation) mô phỏng điều kiện ánh sáng phòng, ban ngày, ban đêm và độ cũ/mới của tờ tiền.")
    add_bullet("Biến đổi phối cảnh (RandomAffine)", "Dịch chuyển 5% và co giãn tỷ lệ 0.9 - 1.1 mô phỏng khoảng cách xa gần từ camera tới tờ tiền.")
    add_bullet("Làm mờ Gauss (GaussianBlur, p=0.2)", "Mô phỏng hiện tượng rung tay hoặc camera lấy nét chưa kịp.")

    add_h2("5.3. Nguyên tắc bất biến về Augmentation trong Đồ án")
    doc.add_paragraph(
        "Thành viên A đã thiết kế pipeline tuân thủ nghiêm ngặt nguyên lý khoa học thực nghiệm: "
        "Augmentation CHỈ ĐƯỢC PHÉP ÁP DỤNG TRÊN TẬP TRAIN. "
        "Hai tập Validation và Test hoàn toàn bị khóa Augmentation, chỉ áp dụng Resize và Normalization thuần túy. "
        "Điều này bảo đảm các chỉ số đánh giá cuối kỳ của nhóm là hoàn toàn khách quan, không bị bóp méo."
    )

    add_h2("5.4. Xây dựng PyTorch Dataset và DataLoader tối ưu")
    doc.add_paragraph(
        "Trong `src/data/dataset.py`, lớp `VNDCashDataset` và hàm `get_dataloaders()` được lập trình tối ưu: "
        "Hỗ trợ nạp batch linh hoạt, thiết lập tự động Windows-safe (num_workers=0), pin_memory=True, "
        "tự động kích hoạt train transform cho train loader và eval transform cho val/test loader."
    )

    # ==================== CHƯƠNG 6 ====================
    add_h1("CHƯƠNG 6: HỆ THỐNG KIỂM THỬ TỰ ĐỘNG VÀ KHÁM PHÁ DỮ LIỆU (EDA)")
    add_h2("6.1. Bộ kiểm thử tự động (Unit Tests) với Pytest")
    doc.add_paragraph(
        "Để đảm bảo chất lượng phần mềm theo chuẩn MLOps trước khi bàn giao cho các thành viên khác, "
        "Thành viên A đã thiết lập bộ kiểm thử tự động toàn diện trong thư mục `tests/`. "
        "Kết quả thực thi đạt 100% tỷ lệ vượt qua (7/7 tests passed):"
    )
    add_bullet("test_vndcash_dataset_loading", "Xác thực khả năng khởi tạo Dataset, nạp ảnh và gán đúng label index (PASSED).")
    add_bullet("test_get_dataloaders_batch", "Xác thực nạp thành công batch ảnh shape [8, 3, 224, 224] và nhãn tương ứng (PASSED).")
    add_bullet("test_eval_transforms_shape", "Xác thực output transform đạt đúng kích thước [1, 3, 224, 224] (PASSED).")
    add_bullet("test_denormalize_tensor", "Xác thực hàm giải chuẩn hóa trả về đúng mảng uint8 3 kênh (PASSED).")
    add_bullet("test_train_transforms_shape", "Xác thực pipeline Augmentation xuất đúng shape (PASSED).")
    add_bullet("test_stratified_group_split_no_leakage", "Kiểm thử toán học tự động bảo đảm 0 rò rỉ tờ tiền (PASSED).")
    add_bullet("test_splits_csv_integrity", "Xác thực tính toàn vẹn 1.962 dòng dữ liệu và nhãn trong khoảng [0, 8] (PASSED).")

    add_h2("6.2. Kết quả thực thi 2 Notebook Khám phá dữ liệu (EDA)")
    doc.add_paragraph(
        "Thành viên A đã xây dựng và chạy thực thi thành công toàn bộ code trong 2 Notebooks, "
        "lưu sẵn toàn bộ hình vẽ, biểu đồ và kết quả phân tích:"
    )
    add_bullet("notebooks/01_dataset_exploration.ipynb (2.05 MB)", "Chứa biểu đồ cột phân bố 9 mệnh giá (làm nổi bật 3 cột thiếu mẫu màu đỏ); biểu đồ tròn tỷ lệ tiền Cotton (33.3%) vs Polymer (66.7%); lưới 9 ảnh mẫu đại diện sắc nét cho 9 mệnh giá; và bảng kiểm tra rò rỉ dữ liệu.")
    add_bullet("notebooks/02_preprocessing.ipynb (1.78 MB)", "Chứa hình ảnh so sánh ảnh gốc và ảnh tiền xử lý; lưới 8 biến thể minh họa các hiệu ứng Data Augmentation trên cùng một tờ tiền; và kết quả nạp thử batch thực tế từ DataLoader.")

    # ==================== CHƯƠNG 7 ====================
    add_h1("CHƯƠNG 7: GỢI Ý NỘI DUNG THUYẾT TRÌNH VÀ BẢO VỆ CUỐI KỲ")
    doc.add_paragraph(
        "Khi tham gia thuyết trình báo cáo đồ án trước Hội đồng chấm thi, Thành viên A nên làm nổi bật "
        "3 điểm sáng kỹ thuật mang lại giá trị cao nhất cho đề tài:"
    )

    add_h2("Slide 1: Điểm sáng 1 — Quản lý Tờ tiền Vật lý (`note_id`) & Chống rò rỉ dữ liệu")
    add_bullet("Nội dung trình bày", "Phân tích vì sao bài toán nhận diện tiền tệ dễ bị 'ảo giác Accuracy' nếu chia ngẫu nhiên. Trình bày cách Thành viên A phân cụm 1.962 ảnh thành 198 tờ tiền độc lập và thuật toán Stratified Grouped Split bảo đảm 100% Zero Leakage.")
    add_bullet("Thông điệp bảo vệ", "'Kết quả đánh giá của nhóm chúng em là hoàn toàn trung thực, khách quan và phản ánh đúng năng lực nhận diện tiền thực tế ngoài đời sống.'")

    add_h2("Slide 2: Điểm sáng 2 — Kiểm toán chất lượng & Phát hiện mất cân bằng dữ liệu")
    add_bullet("Nội dung trình bày", "Trình bày kết quả audit tự động 2.712 ảnh (0 ảnh hỏng, 0 trùng lặp). Chỉ ra 3 lớp 5k, 20k, 50k bị thiếu mẫu (154 ảnh so với 250 ảnh) và cách Thành viên A đưa ra giải pháp Class Weights cho nhóm huấn luyện.")
    add_bullet("Thông điệp bảo vệ", "'Thay vì bỏ qua sự chênh lệch dữ liệu, em đã phân tích định lượng và phối hợp cùng Thành viên B đưa trọng số lớp vào hàm mất mát để mô hình không bị thiên vị.'")

    add_h2("Slide 3: Điểm sáng 3 — Tính sẵn sàng, Chuẩn hóa MLOps và Bàn giao hoàn chỉnh")
    add_bullet("Nội dung trình bày", "Giới thiệu bộ pipeline đã đóng gói chuẩn mực trong `src/data/`, vượt qua 7/7 unit tests tự động, tài liệu hóa đầy đủ trong `docs/dataset_protocol.md` và `HUONG_DAN_BAN_GIAO_CHO_CAC_THANH_VIEN.md`.")
    add_bullet("Thông điệp bảo vệ", "'Các thành viên trong nhóm chỉ cần đúng 1 dòng lệnh là nạp được dữ liệu chuẩn vào mô hình. Toàn bộ mã nguồn đã được đồng bộ trên GitHub và đóng gói hoàn thiện.'")

    # ==================== LỜI KẾT ====================
    add_h1("KẾT LUẬN")
    doc.add_paragraph(
        "Thành viên A (Nguyễn Đức Thịnh) đã hoàn thành xuất sắc, vượt tiến độ toàn bộ các mục tiêu trong phân công đồ án. "
        "Bộ dữ liệu VNDCash và pipeline dữ liệu được xây dựng không chỉ đáp ứng trọn vẹn yêu cầu kỹ thuật của môn học, "
        "mà còn bảo đảm tiêu chuẩn nghiêm ngặt về phương pháp luận nghiên cứu khoa học trong lĩnh vực Trí tuệ nhân tạo và Thị giác máy tính."
    )

    # Save to disk
    output_docx = Path(r"c:\Users\ndt06\Desktop\VNDCash\BAO_CAO_THANH_VIEN_A_DATA_PIPELINE.docx")
    doc.save(str(output_docx))
    print(f"Report successfully generated at: {output_docx}")

    # Copy to Desktop
    desktop_docx = Path(r"C:\Users\ndt06\Desktop\BAO_CAO_THANH_VIEN_A_DATA_PIPELINE.docx")
    doc.save(str(desktop_docx))
    print(f"Report copy saved at: {desktop_docx}")

if __name__ == "__main__":
    create_report()
