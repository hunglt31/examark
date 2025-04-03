# Hướng dẫn cắt ảnh

B1: Clone folder **examark** về máy. Sau khi cài xong, không tắt terminal.

```bash
$ git clone https://github.com/hunglt31/examark.git
```

B2: Cài thư viện **opencv-python**. Có thể cài vào môi trường đã có hoặc tạo môi trường ảo đều được.

```bash
$ pip install opencv-python
```

B3: Mở VSCode, vào folder **data**:
- Folder **scan_images** chứa 2 ảnh scan demo.
- Các folder **align_images**, **unlabel_metadata** và **unlabel_assignment** sẽ được tạo trong quá trình chạy code. Kết quả hiện tại là demo, hãy xóa các folder này đi.

B4: Vào folder **data/scan_images**, xóa 2 ảnh demo. Copy tất cả ảnh scan vào folder này.

B5: Mở terminal ban đầu, chạy các lệnh sau để vào folder **src** và thực hiện cắt ảnh:

```bash
$ cd examark/src
$ python align_and_split.py
```

