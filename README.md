# Examark - Hệ thống chấm trắc nghiệm tự động

Hệ thống chấm trắc nghiệm tự động sử dụng AI để phân tích và chấm điểm các bài thi trắc nghiệm với backend C++/CUDA và frontend React.js.

## Yêu cầu hệ thống

### Phần cứng
- GPU NVIDIA với CUDA Compute Capability ≥ 7.5 (RTX 20xx, RTX 30xx, RTX 40xx, Tesla T4, A100)
- RAM: Tối thiểu 8GB, khuyến nghị 16GB+
- Ổ cứng: Tối thiểu 20GB dung lượng trống

### Phần mềm
- Ubuntu 20.04/22.04 LTS
- NVIDIA Driver (≥ 470.x)
- CUDA Toolkit 12.4
- Docker & Docker Compose
- Node.js 16+ và npm
- CMake 3.17+
- GCC/G++ 9+

## Cài đặt

### 1. Cài đặt NVIDIA Driver và CUDA

```bash
# Cài đặt NVIDIA Driver
sudo apt update
sudo apt install nvidia-driver-470

# Tải và cài đặt CUDA Toolkit 12.4
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
sudo sh cuda_12.4.0_550.54.14_linux.run

# Thêm vào ~/.bashrc
echo 'export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc
```

### 2. Cài đặt Docker và NVIDIA Container Toolkit

```bash
# Cài đặt Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Cài đặt NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 3. Cài đặt dependencies cho Backend

```bash
# Cài đặt build tools
sudo apt install -y cmake build-essential pkg-config

# Cài đặt OpenCV
sudo apt install -y libopencv-dev libopencv-contrib-dev

# Cài đặt Poppler (PDF processing)
sudo apt install -y libpoppler-cpp-dev

# Cài đặt TensorRT
sudo apt install -y libnvinfer8 libnvinfer-plugin8 libnvinfer-dev libnvinfer-plugin-dev libnvonnxparsers8 libnvonnxparsers-dev

# Cài đặt các thư viện khác
sudo apt install -y libcurl4-openssl-dev libssl-dev
```

### 4. Tải Triton Client SDK

```bash
# Tạo thư mục Downloads nếu chưa có
mkdir -p ~/Downloads
cd ~/Downloads

# Tải Triton Client SDK
wget https://github.com/triton-inference-server/client/releases/download/v2.41.0/tritonserver2.41.0-clientsdk.tar.gz
tar -xzf tritonserver2.41.0-clientsdk.tar.gz
mv tritonserver2.41.0-clientsdk TritonClientSDK
```

### 5. Clone và build Backend

```bash
# Clone repository
git clone <repository-url> examark
cd examark/backend

# Tạo thư mục build
mkdir build && cd build

# Configure với CMake (điều chỉnh CUDA architecture nếu cần)
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_ARCHITECTURES=89 ..

# Build
make -j$(nproc)
```

### 6. Cài đặt Frontend

```bash
# Chuyển đến thư mục frontend
cd ../../frontend

# Cài đặt dependencies
npm install

# Build production (tùy chọn)
npm run build
```

## Cấu hình

### Backend Configuration

Cập nhật các đường dẫn trong CMakeLists.txt nếu cần:

```cmake
# Cập nhật đường dẫn Triton Client SDK
set(TRITON_CLIENT_DIR "/home/username/Downloads/TritonClientSDK")

# Cập nhật CUDA architecture cho GPU của bạn
set(TARGET_ARCH 89) # RTX 4090
# Hoặc: 86 (RTX 30xx), 75 (RTX 20xx), 80 (A100)
```

### Setup Triton Server

```bash
# Tạo thư mục cho models
mkdir -p ~/triton-models

# Chạy Triton Server với Docker
docker run --gpus=all -it --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v ~/triton-models:/models \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models --strict-model-config=false
```

## Chạy hệ thống

### 1. Khởi động Triton Server

```bash
docker run --gpus=all -it --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v ~/triton-models:/models \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models
```

### 2. Chạy Backend

```bash
cd examark/backend/build
./server
```

### 3. Chạy Frontend

```bash
# Development mode
cd examark/frontend
npm start

# Production mode (sau khi build)
npx serve -s build -p 3000
```

## API Endpoints

Backend server chạy trên port 8080:

- `POST /upload` - Upload file PDF
- `POST /process` - Xử lý và chấm điểm
- `GET /results` - Lấy kết quả
- `GET /export` - Export kết quả ra CSV

## Troubleshooting

### Lỗi CUDA

```bash
# Kiểm tra CUDA
nvidia-smi
nvcc --version
```

### Lỗi TensorRT

```bash
# Kiểm tra TensorRT libraries
ldconfig -p | grep nvinfer
```

### Lỗi Triton Client

```bash
# Kiểm tra Triton server status
curl -X GET http://localhost:8000/v2/health/ready

# Kiểm tra models
curl -X GET http://localhost:8000/v2/models
```

### Lỗi Build

```bash
# Xóa build cache và rebuild
rm -rf build/
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_ARCHITECTURES=89 ..
make VERBOSE=1
```

## Kiến trúc hệ thống

```
ExamArk/
├── backend/           # C++ Backend with CUDA
│   ├── src/          # Source code
│   ├── includes/     # Header files
│   └── CMakeLists.txt
├── frontend/         # React.js Frontend
├── models/           # AI Models
└── docs/            # Documentation
```

## License

[Thông tin license]

## Liên hệ

- Email: hungthanh3123@gmail.com
- Phone: (+84) 869 030 103

---

**Lưu ý**: Đảm bảo GPU có đủ VRAM (tối thiểu 6GB) để chạy các models AI.