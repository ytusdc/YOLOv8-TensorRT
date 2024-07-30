cmake_minimum_required(VERSION 3.1)   # 声明要求的 cmake 最低版本

# 设置项目名称 和编程语言
# 只需要在project的LANGUAGES 加入cuda，就可以添加对cuda的支持了
project(yolov8 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86 89 90)  # 通过 CMake 设置架构版本号
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)         # 选择指定的cuda编译器

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")
set(CMAKE_CXX_STANDARD 14)

# 设置工程编译类型
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

#定义CUDA_USE_STATIC_CUDA_RUNTIME，并将其值设为OFF
#为了解决cannot find -lopencv_dep_cudart问题
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# CUDA， REQUIRED：标志查找不到就报错
find_package(CUDA REQUIRED)
message(STATUS "CUDA Libs: \n${CUDA_LIBRARIES}\n")
get_filename_component(CUDA_LIB_DIR ${CUDA_LIBRARIES} DIRECTORY)
message(STATUS "CUDA Headers: \n${CUDA_INCLUDE_DIRS}\n")

# OpenCV
find_package(OpenCV REQUIRED)
# message 向终端输出信息
message(STATUS "OpenCV Libs: \n${OpenCV_LIBS}\n")
message(STATUS "OpenCV Libraries: \n${OpenCV_LIBRARIES}\n")
message(STATUS "OpenCV Headers: \n${OpenCV_INCLUDE_DIRS}\n")

# TensorRT
set(TensorRT_INCLUDE_DIRS /usr/include/x86_64-linux-gnu)
set(TensorRT_LIBRARIES /usr/lib/x86_64-linux-gnu)
message(STATUS "TensorRT Libs: \n${TensorRT_LIBRARIES}\n")
message(STATUS "TensorRT Headers: \n${TensorRT_INCLUDE_DIRS}\n")

list(APPEND INCLUDE_DIRS
        ${CUDA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
        include
        )

list(APPEND ALL_LIBS
        ${CUDA_LIBRARIES}
        ${CUDA_LIB_DIR}
        ${OpenCV_LIBRARIES}
        ${TensorRT_LIBRARIES}
        )

include_directories(${INCLUDE_DIRS})

#生成程序
add_executable(${PROJECT_NAME}
        main.cpp
        include/yolov8.cpp
        include/common.hpp
        include/utils.hpp
        include/affine.cu
        )

# 将链接目录添加到target
target_link_directories(${PROJECT_NAME} PUBLIC ${ALL_LIBS})
# 将库文件链接到可执行程序上
target_link_libraries(${PROJECT_NAME} PRIVATE nvinfer nvinfer_plugin cudart ${OpenCV_LIBS})

if (${OpenCV_VERSION} VERSION_GREATER_EQUAL 4.7.0)
    message(STATUS "Build with -DBATCHED_NMS")
    add_definitions(-DBATCHED_NMS)
endif ()
