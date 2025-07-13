cmake_minimum_required(VERSION 3.17)
project(examark LANGUAGES CXX CUDA)

add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-deprecated-declarations>)

# Host debug
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG}   -g -O0")

# CUDA debug (-G turns on device‑side debugging info)
set(CMAKE_CUDA_FLAGS_DEBUG  "${CMAKE_CUDA_FLAGS_DEBUG}  -G -g -O0")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# --- Specify Target CUDA Architectures ---
# Common Architectures:
#   60 (Pascal P100)
#   61 (Pascal GTX 10xx)
#   70 (Volta V100)
#   75 (Turing RTX 20xx, T4)
#   80 (Ampere A100)
#   86 (Ampere RTX 30xx, A10, A40)
#   89 (Ada Lovelace RTX 40xx, L4)
#   90 (Hopper H100)             

if(NOT CMAKE_CUDA_ARCHITECTURES)
    message(WARNING "CMAKE_CUDA_ARCHITECTURES is not set. Defaulting to '75'. \
Use -DCMAKE_CUDA_ARCHITECTURES='<arch1>[;<arch2>...]' to specify target(s).")
    set(TARGET_ARCH 89) 
else()
    set(TARGET_ARCH ${CMAKE_CUDA_ARCHITECTURES})
endif()

# Set the variable used by CMake's CUDA integration
set(CMAKE_CUDA_ARCHITECTURES ${TARGET_ARCH} CACHE STRING "Target CUDA Architectures (e.g., '75', '86', '75;86')")

SET(CMAKE_BUILD_TYPE Release)
SET(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)

# Define TensorRT and CUDA paths if necessary
set(TensorRT_DIR "/usr/lib/x86_64-linux-gnu") 
set(CUDA_DIR "/usr/local/cuda-12.4/targets/x86_64-linux")   

# Set triton client SDK dir
set(TRITON_CLIENT_DIR "/home/hunglt31/Downloads/TritonClientSDK")
include_directories(${TRITON_CLIENT_DIR}/include)
link_directories(${TRITON_CLIENT_DIR}/lib)

find_library(GRPC_LIBRARY grpc++)
find_library(GPR_LIBRARY gpr)

# Find OpenCV
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED) 
find_package(CUDAToolkit REQUIRED)
find_package(CURL REQUIRED)
find_package(OpenSSL REQUIRED) 
pkg_check_modules(POPPLERCPP REQUIRED poppler-cpp)

# Include CUDA and TensorRT
find_package(CUDA REQUIRED)
include_directories(${TensorRT_DIR}/include)
include_directories(${CUDA_DIR}/include)
include_directories(${POPPLERCPP_INCLUDE_DIRS}) 
include_directories(${PROJECT_SOURCE_DIR}/includes)

# Link TensorRT libraries
set(TensorRT_LIBS
    ${TensorRT_DIR}/libnvinfer.so
    ${TensorRT_DIR}/libnvonnxparser.so
    ${TensorRT_DIR}/libnvinfer_plugin.so
    ${CUDA_DIR}/lib/libcudart.so
)

# Source files
set(SOURCES 
    src/MainServer.cpp
    src/utils/ImageProcessor.cpp
    src/utils/ExamGrader.cpp
    src/utils/Logger.cpp
    src/utils/MinIOHTTPClient.cpp
    src/models/ModelBuilder.cpp
    src/models/TritonClient.cpp
    src/services/Service.cpp
    src/controllers/Controller.cpp
)

# Server executable
add_executable(server ${SOURCES})
set_target_properties(server PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(server PRIVATE
    ${OpenCV_LIBS} 
    ${CUDA_LIBRARIES} 
    ${CUDA_LIBS} 
    ${TensorRT_LIBS} 
    ${POPPLERCPP_LIBRARIES}
    ${TRITON_CLIENT_DIR}/lib/libgrpcclient.so
    ${PROTOBUF_LIBRARIES}
    ${GRPC_LIBRARY}
    ${GPR_LIBRARY}
    ${CURL_LIBRARIES}
    OpenSSL::SSL OpenSSL::Crypto 
    
)

target_include_directories(server PRIVATE 
    ${CURL_INCLUDE_DIRS}
)

# Builder executable
add_executable(engine-builder 
    src/MainBuilder.cpp 
    src/models/ModelBuilder.cpp
    src/utils/Logger.cpp
)
set_target_properties(engine-builder PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(engine-builder PRIVATE
    ${TensorRT_LIBS} 
    ${OpenCV_LIBS} 
    ${CUDA_LIBRARIES} 
    ${CUDA_LIBS} 
)