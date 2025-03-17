cmake_minimum_required(VERSION 3.22)
project(yolo_demo)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)
find_package(FFMPEG REQUIRED)

# 将yolov8n.onnx模型所在路径作为宏传入
add_compile_definitions(-DYOLO_FILE_PATH=\"${CMAKE_CURRENT_LIST_DIR}/yolov8n.onnx\")
add_compile_definitions(-DLENA_FILE_PATH=\"${CMAKE_CURRENT_LIST_DIR}/LenaRGB.tif\")

add_library(ffmpeg_library_interface INTERFACE)
add_library(ort INTERFACE)
add_executable(yolo_demo detect_img.cpp)

target_include_directories(ffmpeg_library_interface INTERFACE ${FFMPEG_INCLUDE_DIRS})
target_link_directories(ffmpeg_library_interface INTERFACE ${FFMPEG_LIBRARY_DIRS})
target_link_libraries(ffmpeg_library_interface INTERFACE ${FFMPEG_LIBRARIES})

# 1.将onnxruntime的路径换成vcpkg安装路径下对应的onnxruntime-gpu*\include和lib
target_include_directories(ort INTERFACE "D:\\APP\\vcpkg-2024.11.16\\packages\\onnxruntime-gpu_x64-windows\\include")
target_link_directories(ort INTERFACE "D:\\APP\\vcpkg-2024.11.16\\packages\\onnxruntime-gpu_x64-windows\\lib")
target_link_libraries(ort INTERFACE onnxruntime)

target_link_libraries(yolo_demo PRIVATE
        ffmpeg_library_interface
        ort
        ${OpenCV_LIBS}
)