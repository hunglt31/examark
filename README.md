# Hướng dẫn cắt ảnh

B1: Clone folder **examark** về máy. Sau khi cài xong, không tắt terminal.

```bash
$ git clone https://github.com/hunglt31/examark.git
```

B2: Cài thư viện **opencv-python**. Có thể cài vào môi trường đã có hoặc tạo môi trường ảo đều được.

```bash
$ pip install opencv-python
```

B3: Mở VSCode, tại folder **examark**, tạo folder **data/scan_images**. Copy tất cả ảnh scan vào folder này.

B4: Mở terminal ban đầu, chạy các lệnh sau để vào folder **src** và thực hiện cắt ảnh:

```bash
$ cd examark/src
$ python align_and_split.py
```

B5: Zip 2 folder **unlabel_metadata** và **unlabel_assignment** đẩy lên roboflow đánh nhãn riêng, vì mình train 2 model cho 2 phần này.
