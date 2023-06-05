cmake_minimum_required(VERSION 3.25)
project(FCnet)

set(CMAKE_CXX_STANDARD 17)

if(CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE STREQUAL "Debug"))
elseif(CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE STREQUAL "Release"))
    if(MSVC)
        set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
        set(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} /O2 /Ob2 /GL /Gy /EHsc /fp:fast /arch:AVX2 /MP /openmp /source-charset:utf-8")
    else()
        set(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} -O3 -mavx -mfma -ffast-math")
    endif()
else()
    message("Unknown build type:${CMAKE_BUILD_TYPE}")
endif()


add_executable(FCnet src/main.cpp
        include/read_csv.h
        include/Softmax_dense.h
        include/Frame.h
        include/Dense.h
        include/Relu_dense.h
        src/read_csv.cpp
        src/Dense.cpp
        src/Relu_dense.cpp
        src/Frame.cpp
        src/Softmax_dense.cpp src/Linearalgebra.cpp src/Convert_to_grayscale.cpp)
# OpenCV-MinGW-Build
set(OpenCV_DIR D:/Opencv/OpenCV-MinGW-Build-OpenCV-4.5.5-x64)
FIND_PACKAGE(OpenCV REQUIRED)
# 项目名称
TARGET_LINK_LIBRARIES(FCnet ${OpenCV_LIBS})

include_directories(D:/Ei_gen/Eigen)

set(SOURCE_FILES main.cpp)






