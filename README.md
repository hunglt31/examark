# Examark - Hệ thống chấm trắc nghiệm tự động

Hệ thống chấm trắc nghiệm tự động sử dụng AI để phân tích và chấm điểm các bài thi trắc nghiệm với backend C++/CUDA và frontend React.js.

## Yêu cầu hệ thống

### Phần cứng
- GPU NVIDIA với CUDA Compute Capability ≥ 7.5 (RTX 20xx, RTX 30xx, RTX 40xx, Tesla T4, A100)
- RAM: Tối thiểu 16GB, khuyến nghị 32GB+
- Ổ cứng: Tối thiểu 50GB dung lượng trống

### Phần mềm
- Ubuntu 22.04.5 LTS
- NVIDIA Driver (≥ 470.x)
- CUDA Toolkit 12.4
- Docker & Docker Compose
- Node.js 16+ và npm
- CMake 3.17+
- GCC/G++ 9+

## Cài đặt

### 1. Cài đặt NVIDIA Driver và CUDA

```bash
# Cài đặt NVIDIA driver
# Check danh sách driver bằng 2 lệnh sau
sudo apt update
sudo ubuntu-drivers devices

# Cài phiên bản được đề xuất cho máy (có chữ recommend)
sudo apt install nvidia-driver-5xx

# Tải và cài đặt CUDA Toolkit 12.4: Vào trang chủ của CUDA và tải về phiên bản phù hợp với OS máy, sau đó cài CUDA driver như hướng dẫn trên trang
# https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local

# Thêm CUDA 12.4 vào ~/.bashrc
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# Kiểm tra kết quả
source ~/.bashrc
```

### 2. Cài đặt OpenCV sử dụng GPU

```bash
# Build from source vì opencv mặc định chỉ hỗ trợ CPU
cd Downloads
git clone -b 4.10.0 https://github.com/opencv/opencv.git

cd opencv/opencv_contrib-4.10.0

cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=~/Downloads/opencv/opencv_contrib-4.10.0/modules \
      -D WITH_CUDA=ON \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D ENABLE_FAST_MATH=ON \
      -D CUDA_FAST_MATH=ON \
      -D WITH_CUBLAS=ON \
      -D BUILD_opencv_python3=OFF \
      -D BUILD_opencv_python2=OFF \
      ../opencv-4.10.0
```

### 3. Cài đặt TensorRT (nếu build engine, tạm thời có thể bỏ qua vì engine đã có trên git)
 ```bash
# Vào đây và tải về phiên bản TensorRT phù hợp với may (tạo tài khoản NVIDIA)
# TensorRT có thể không hỗ trợ một số tính năng trên Windows
# Ở đây sử dụng: TensorRT 10.3 GA for Ubuntu 22.04 and CUDA 12.0 to 12.5 DEB local repo Package
# https://developer.nvidia.com/tensorrt/download/10x

# Cài repo và key
sudo dpkg -i nv-tensorrt-local-repo-*.deb
sudo cp /var/nv-tensorrt-local-repo-*/.*.gpg /usr/share/keyrings/
sudo apt-get update

# Cài TensorRT
sudo apt-get install tensorrt libnvinfer-dev=10.3.0.26-1+cuda12.4 \
  libnvonnxparsers-dev=10.3.0.26-1+cuda12.4
```

### 4. Cài đặt Docker và NVIDIA Container Toolkit

```bash
# Cài đặt Docker
# Triton Inference Server cài đặt native rất phức tạp và được khuyên khích chạy trên Docker
# https://docs.docker.com/engine/install/ubuntu/

# Cài đặt NVIDIA Container Toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
```

### 5. Cài Triton Inference Server và MinIO bằng Docker

```bash
# Cài Triton
docker pull nvcr.io/nvidia/tritonserver:24.08-py3

# Cài MinIO
docker pull quay.io/minio/minio
```


### 6. Tải Triton Client SDK

```bash
cd ~/Downloads
mkdir TritonClientSDK && cd TritonClientSDK
wget https://github.com/triton-inference-server/server/releases/download/v2.49.0_ubuntu2204.clients.tar.gz
tar -xzvf v2.49.0_ubuntu2204.clients.tar.gz .
```

### 7. Cài đặt dependencies cho Backend

```bash
# Cài đặt build tools
sudo apt install -y cmake build-essential pkg-config

# Cài đặt Poppler (PDF processing)
sudo apt install -y libpoppler-cpp-dev

# Cài đặt các thư viện khác
sudo apt install -y libcurl4-openssl-dev libssl-dev // Cho backend API và MinIO Client
sudo apt install nlohmann-json3-dev // Đê đọc viết JSON
```

### 8. Clone repository

```bash
# Clone repository
git clone https://github.com/hunglt31/examark
```

Cập nhật các đường dẫn trong CMakeLists.txt nếu cần:

```cmake
# Cập nhật đường dẫn Triton Client SDK
set(TRITON_CLIENT_DIR "/home/username/Downloads/TritonClientSDK")

# Cập nhật CUDA architecture cho GPU của bạn
set(TARGET_ARCH 89) # RTX 4090
# Hoặc: 86 (RTX 30xx), 75 (RTX 20xx), 80 (A100)
```

### 9. Cài đặt Backend

```bash
# Clone repository
git clone https://github.com/hunglt31/examark
cd examark/backend

# Tạo thư mục build
mkdir build && cd build

# Configure với CMake (điều chỉnh CUDA architecture, ở đây 89 - RTX 4060)
cmake -DCMAKE_CUDA_ARCHITECTURES=89 ..

# Build
make -j$(nproc)
```

### 10. Cài đặt Frontend

```bash
# Chuyển đến thư mục frontend
cd ../../frontend

# Cài đặt dependencies
npm install

# Build production (tùy chọn)
npm run build
```


## Chạy hệ thống

### 1. Khởi động Triton Server

```bash
# Tạo và chạy Triton container
# Đổi tên đường dẫn đến model repository Triton, ở đây là '/home/hunglt31/examark/models'
docker run --gpus=all -d --name triton -p 8001:8001 \
    -v /home/hunglt31/examark/models:/models \
    nvcr.io/nvidia/tritonserver:24.08-py3 \
    tritonserver --model-repository=/models \
                 --allow-http=false \
                 --allow-metrics=false

# Dừng Triton
docker stop triton

# Chạy lại Triton
docker start triton

# Xóa Triton
docker rm -f triton
```

### 2. Chạy MinIO
```bash
# Tạo và chạy MinIO container
docker run -d --name minio -p 9000:9000 -p 9001:9001 \
  -v /home/hunglt31/examark-data:/data \
  -e "MINIO_ROOT_USER=minioadmin" \
  -e "MINIO_ROOT_PASSWORD=minioadmin123" \
  minio/minio server /data --console-address ":9001"

# Dừng MinIO
docker stop minio

# Chạy lại MinIO
docker start minio

# Xóa MinIO
docker rm -f minio
```

### 3. Chạy Backend

```bash
cd examark/backend/build
# Đã có engine
./server

# Chưa có engine
./engine-builder
./server
```

### 4. Chạy Frontend

```bash
cd examark/frontend

# Development mode
npm start

# Production mode (sau khi build)
serve -s build
```

## Kiến trúc hệ thống

```
Examark/
├── backend/            # C++ Backend with CUDA
|   ├── assets/         # Reference image
│   ├── src/            # Source code
│   ├── includes/       # Header files
│   └── CMakeLists.txt
├── frontend/           # React.js Frontend
|   ├── node_modules/   # Thư viện
│   ├── src/            # Source code
│   ├── public/         # Header files
│   └── CMakeLists.txt
└── nodels/             # Triton model repository
```

## Liên hệ

- Email: hungthanh3123@gmail.com
- Phone: (+84) 869 030 103

---

