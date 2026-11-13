cmake_minimum_required (VERSION 3.18)
project (PRPA)

set (CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Enable CUDA
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)


pkg_check_modules(GStreamer REQUIRED gstreamer-1.0 IMPORTED_TARGET)
pkg_check_modules(GStreamerVideo REQUIRED gstreamer-video-1.0 IMPORTED_TARGET)


set(CMAKE_CXX_STANDARD 20)

add_library(stb_image src/stb_image.h src/stb_image.c)
add_library(Compute
  src/Compute.cpp
  src/Compute.cu
  src/logo.c
  src/stb_image.c
  # CPU
  src/process/cpu/BGEProcess.cpp
  src/process/cpu/MCProcess.cpp
  # GPU
  src/process/gpu/BGEProcess.cu
  src/process/gpu/MCProcess.cu
)

target_compile_options(Compute PRIVATE
    -Wall -W -g
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
target_include_directories(Compute PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/src/process
)
target_link_libraries(Compute PRIVATE CUDA::cudart CUDA::curand)

add_executable(stream src/stream.cpp src/gstfilter.c src/gstfilter.h)

target_link_libraries(stream PUBLIC
    PkgConfig::GStreamerVideo
    Compute
    CUDA::cudart
)
