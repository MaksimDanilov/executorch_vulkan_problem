cmake_minimum_required(VERSION 3.15)
project(test_executorch LANGUAGES CXX)

# Use C++17 (ExecuTorch supports C++17) :contentReference[oaicite:1]{index=1}
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Specify the executable
add_executable(app main.cpp)

# Link libraries: adapt names/path to your build.
# For example, ExecuTorch may provide static libs like libexecutorch.a etc. :contentReference[oaicite:2]{index=2}
find_package(ExecuTorch REQUIRED)
target_link_libraries(app PRIVATE executorch executorch_core executorch_extensions executorch_backends executorch_kernels)

target_link_options(
    app INTERFACE
    "SHELL:LINKER:-force_load,$<TARGET_FILE:executorch_backends>"
)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs)
target_link_libraries(app PRIVATE opencv_core opencv_imgcodecs)
