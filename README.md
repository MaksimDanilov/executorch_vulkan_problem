# Installation

1. create .venv and install requirements.  
1.1. download weights using setup_model_weights.py
2. build executorch from source from branch v1.0.0 with
```sh
# Build options common to all ABIs
CMAKE_COMMON_OPTIONS="-DCMAKE_BUILD_TYPE=Release \
-DEXECUTORCH_BUILD_COREML=OFF \
-DBUILD_SHARED_LIBS=OFF \
-DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF \
-DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
-DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
-DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
-DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
-DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
-DEXECUTORCH_BUILD_XNNPACK=ON \
-DEXECUTORCH_BUILD_VULKAN=ON"
cmake -S . -B build ${CMAKE_COMMON_OPTIONS}
cmake --build build --config Release -j
```
3. build cmake project to current directory (or build in separate folder and but start app in root project directory)

## Hint

1. setup_model_weights.pt - download weights
2. test.py - test origin model by test_data
3. convert.py - modify model to do pre- and post-processing and converts to pte.
4. app (build) - run pte and save/load images (requires opencv and executorch).
5. https://disk.yandex.ru/d/1cW638087PnZgg have two exported programs (vulkan and xnnpack).

## Tested on

1. MacBook Pro M1 Max
2. glslc --version
```sh
shaderc v2023.8 v2025.3-10-gc7e73e8
spirv-tools v2025.4 v2022.4-970-g19042c89
glslang 11.1.0-1302-gd213562e

Target: SPIR-V 1.0
```
