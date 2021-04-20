cmake_minimum_required(VERSION 3.13.4)

project(protogen-mlir LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED YES)



set(MLIR_DIR ${PROJECT_SOURCE_DIR}/llvm-project/build/lib/cmake/mlir)
set(LLVM_DIR ${PROJECT_SOURCE_DIR}/llvm-project/build/lib/cmake/llvm)
set(LLVM_EXTERNAL_LIT ${PROJECT_SOURCE_DIR}/llvm-project/llvm/build/bin/llvm-lit)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

find_package(MLIR REQUIRED CONFIG)

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

set(LLVM_LIT_ARGS "-sv" CACHE STRING "lit default options")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")
include(sanitizers)

# ANTLR4 Stuff ----- 

# required if linking to static library
add_definitions(-DANTLR4CPP_STATIC)

# add external build for antlrcpp
include(ExternalAntlr4Cpp)

# add antrl4cpp artifacts to project environment
include_directories(${ANTLR4_INCLUDE_DIRS})

# set variable pointing to the antlr tool that supports C++
# this is not required if the jar file can be found under PATH environment
set(ANTLR_EXECUTABLE ${PROJECT_SOURCE_DIR}/antlr-4.9.2-complete.jar )

# add macros to generate ANTLR Cpp code from grammar
find_package(ANTLR REQUIRED)

# Call macro to add lexer and grammar to your build dependencies.
ANTLR_TARGET(ProtoCC ProtoCC.g4 VISITOR)

# include generated files in project environment
include_directories(${ANTLR_ProtoCC_OUTPUT_DIR})
message(STATUS "Including ANTLR4 headers from: ${ANTLR_ProtoCC_OUTPUT_DIR}")
message(STATUS "ANTLR is found = ${ANTLR_FOUND} and path to jar file is ${ANTLR_EXECUTABLE}")

add_library(protocc-antlr4 STATIC ${ANTLR_ProtoCC_CXX_OUTPUTS})
target_link_libraries(protocc-antlr4 antlr4_static)
# --- END ANTLR

# murphi-lib
add_library(murphi-lib SHARED IMPORTED)
set_property(TARGET murphi-lib PROPERTY IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/murphi-lib/build/libmurphi.dylib)
include_directories(murphi-lib/include)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)