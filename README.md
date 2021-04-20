# ProtoGen-MLIR-V2

## Build and Run

To build this project you first need to build `llvm` and `murphi-lib`, which are included as submodules.

Requirements
- CMake
- Ninja
- clang


### Build LLVM
- ```zsh 
  cd llvm-project && mkdir build && cd build
  ```

- ```zsh 
  cmake -G Ninja ../llvm\
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_EH=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_INSTALL_UTILS=ON\
  -DLLVM_BUILD_EXAMPLES=ON
  ```
- ```zsh
    cmake --build .
  ```


## Build murphi-lib
- ```zsh 
    cd murphi-lib && mkdir build && cd build
    ```
- ```zsh 
    cmake -G Ninja ..
    ```
- ```zsh 
    cmake --build .
    ```


## build ProtoGen-MLIR-V2

- ```zsh 
    mkdir build && cd build
    ```

- ```zsh 
    cmake -G Ninja .. -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_BUILD_TYPE=DEBUG
    ```
