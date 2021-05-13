# ProtoGen-MLIR-V2

## Build and Run

To build this project you first need to build `llvm` and `murphi-lib`, which are included as submodules.
```
$ git clone https://github.com/veselypeta/protogen-mlir-v2.git
$ cd protogen-mlir-v2
$ git submodule init
$ git submodule update
```

Requirements
- `Git`
- `CMake`
- `Ninja`
- `clang`
- `uuid-dev`

Begin each step from within the project root directory

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
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_BUILD_EXAMPLES=OFF
```

```zsh
cmake --build .
```


### Step 2 - Build murphi-lib
```zsh 
cd murphi-lib && mkdir build && cd build
```
```zsh 
cmake -G Ninja ..
```
```zsh 
cmake --build .
```


### Step 3 - Build ProtoGen-MLIR-V2

```zsh 
mkdir build && cd build
```

```zsh 
cmake -G Ninja .. -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_BUILD_TYPE=DEBUG
```

```zsh 
cmake --build .
```
Note: This step may fail, but running again will usually succeed. This is caused by the codegen for ANTLR4 not completing before the compilation of generated files.
