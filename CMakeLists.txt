cmake_minimum_required(VERSION 3.16.2 FATAL_ERROR)

project(protogen-mlir LANGUAGES CXX C)
 
if (NOT CMAKE_BUILD_TYPE EQUAL RELEASE)
    include(FetchContent)
    FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif ()

# compile cmurphi before enabling warnings as errors
add_subdirectory(externals/CMurphi)

# enable errors
if (MSVC)
    # warning level 4 and all warnings as errors
    add_compile_options(/W4 /WX)
else()
    # lots of warnings and all warnings as errors
    add_compile_options(-Wall -Wextra -pedantic -Werror)
    if(NOT CMAKE_BUILD_TYPE EQUAL DEBUG)
        message("OPTIMIZATIONS HAVE BEEN ENABLED")
        add_compile_options(-O3)
    endif()
endif()

set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(MLIR_DIR ${PROJECT_SOURCE_DIR}/llvm-project/build/lib/cmake/mlir)
set(LLVM_DIR ${PROJECT_SOURCE_DIR}/llvm-project/build/lib/cmake/llvm)
if(NOT DEFINED LLVM_EXTERNAL_LIT)
    set(LLVM_EXTERNAL_LIT ${PROJECT_SOURCE_DIR}/llvm-project/llvm/build/bin/llvm-lit)
endif()
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

# make these system headers to disable compiler warnings
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(SYSTEM ${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

set(LLVM_LIT_ARGS "-sv" CACHE STRING "lit default options")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")
include(AddPROTOGEN)
include(sanitizers)

# ANTLR4 Stuff ----- 
set(ANTLR4_TAG 4.9.3)
# required if linking to static library
add_definitions(-DANTLR4CPP_STATIC)

# add external build for antlrcpp
include(ExternalAntlr4Cpp)

# add antrl4cpp artifacts to project environment
include_directories(${ANTLR4_INCLUDE_DIRS})

# set variable pointing to the antlr tool that supports C++
# this is not required if the jar file can be found under PATH environment
set(ANTLR_EXECUTABLE ${PROJECT_SOURCE_DIR}/antlr-4.9.2-complete.jar)

# add macros to generate ANTLR Cpp code from grammar
find_package(ANTLR REQUIRED)

# Call macro to add lexer and grammar to your build dependencies.
antlr_target(ProtoCC ProtoCC.g4 VISITOR)

# include generated files in project environment
include_directories(${ANTLR_ProtoCC_OUTPUT_DIR})
message(STATUS "Including ANTLR4 headers from: ${ANTLR_ProtoCC_OUTPUT_DIR}")
message(STATUS "ANTLR is found = ${ANTLR_FOUND} and path to jar file is ${ANTLR_EXECUTABLE}")

add_library(protocc-antlr4 ${ANTLR_ProtoCC_CXX_OUTPUTS})
add_dependencies(protocc-antlr4 antlr4_static)

# --- END ANTLR

# Sanitisers
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/externals/sanitizers-cmake/cmake" ${CMAKE_MODULE_PATH})
find_package(Sanitizers)

# -- valijson
set(valijson_BUILD_TESTS OFF CACHE BOOL "disable valijson tests")
add_subdirectory(third-party/valijson)
include_directories(SYSTEM third-party/valijson/include)

# --- include 3rd party code
include_directories(SYSTEM third-party)

# code is dependant on tablegen'd targets
add_custom_target(protogen-headers)
set_target_properties(protogen-headers PROPERTIES FOLDER "Misc")

add_subdirectory(include)
add_subdirectory(lib)
# don't include tests in release build
if (NOT CMAKE_BUILD_TYPE EQUAL RELEASE)
    add_subdirectory(test)
endif ()
add_subdirectory(tools)


