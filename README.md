# ProtoGen-MLIR-V2

##Requirements
- `Git`
- `CMake`
- `Ninja`
- `clang`
- `lld`
- `uuid-dev`

On Ubuntu you can install all dependencies with the following:
```zsh
sudo apt-get install -y git cmake ninja-build clang lld uuid-dev
```


## Build and Run

To build this project you first need to build `llvm`, which is included as a submodule.
Follow the instructions to clone and set up the repository.
```zsh
git clone https://github.com/veselypeta/protogen-mlir-v2.git
```
```zsh
cd protogen-mlir-v2
```
```zsh
git submodule init
```
```zsh
git submodule update
```

*Begin each step from within the project root directory*

### Step 1 - Build LLVM
```zsh 
cd llvm-project && mkdir build && cd build
```

```zsh 
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_EH=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_LLD=ON \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_INCLUDE_TESTS=OFF
```

```zsh
cmake --build .
```

Note: Building LLVM will take a long time (~30 min) and is quite heavy on resources. I recommend that you use a minimum 4-core modern
CPU and at least 16 GB of RAM.

### Step 2 - Build ProtoGen-MLIR-V2

```zsh 
mkdir build && cd build
```

```zsh 
cmake -G Ninja .. -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_BUILD_TYPE=DEBUG
```

```zsh 
cmake --build .
```
Note: a clean build will take a while to complete (~5 min), it needs to download and compile the ANTLR4 cpp runtime.
However, incremental builds are much faster.

## Tests
To run all tests
```zsh
cd build/test
```
```asm
ctest
```