import os
import sys
import fitz 

# Nhận đường dẫn folder chứa file PDF từ tham số dòng lệnh hoặc nhập thủ công
if len(sys.argv) > 1:
    input_folder = sys.argv[1]
else:
    input_folder = input("Nhập đường dẫn tới folder chứa file PDF: ").strip()

# Kiểm tra folder có tồn tại không
if not os.path.isdir(input_folder):
    print("Folder không tồn tại.")
    sys.exit(1)

# Tạo folder lưu kết quả nếu chưa tồn tại
output_folder = "results"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Duyệt qua các file trong folder input
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(input_folder, filename)
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"Lỗi khi mở file {filename}: {e}")
            continue
        
        # Tạo folder con trong 'results' với tên file PDF (không có đuôi)
        base_name = os.path.splitext(filename)[0]
        pdf_output_folder = os.path.join(output_folder, base_name)
        if not os.path.exists(pdf_output_folder):
            os.makedirs(pdf_output_folder)
        
        # Sử dụng zoom = 1.0 để giữ nguyên chất lượng gốc (72 DPI)
        zoom = 1.0  
        mat = fitz.Matrix(zoom, zoom)
        
        # Chuyển từng trang của PDF thành ảnh PNG
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=mat)
            output_path = os.path.join(pdf_output_folder, f"page_{i+1}.png")
            pix.save(output_path)
            print(f"Trang {i+1} của {filename} đã được lưu tại: {output_path}")
        
        doc.close()
