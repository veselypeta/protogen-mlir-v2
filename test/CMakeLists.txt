enable_testing()

add_executable(protogen-mlir-test
        mlirGenTest.cpp
        MurphiTemplatesTest.cpp
        InjaTests.cpp
        # codegen tests
        codegen/InjaEnvSingletonTests.cpp
        codegen/ConstDeclTemplateTests.cpp
        codegen/TypeDeclTemplateTests.cpp
        codegen/EnumTypeTests.cpp
        codegen/ArrayTypeTests.cpp
        codegen/SubRangeTests.cpp
        murphi-translate/MurphiTranslateTests.cpp)

target_link_libraries(protogen-mlir-test gtest_main MLIRIR antlr4_static mlir-gen trans-murphi)

include(GoogleTest)
gtest_discover_tests(protogen-mlir-test)