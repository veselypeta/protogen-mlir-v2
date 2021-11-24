enable_testing()

add_executable(protogen-mlir-test
        mlirGen/mlirGenTest.cpp
        misc/NestedStructureTests.cpp
        MurphiTemplatesTest.cpp
        InjaTests.cpp
        ValiJSONTests.cpp
        # codegen tests
        codegen/InjaEnvSingletonTests.cpp
        codegen/ConstDeclTemplateTests.cpp
        codegen/TypeDeclTemplateTests.cpp
        codegen/EnumTypeTests.cpp
        codegen/ArrayTypeTests.cpp
        codegen/SubRangeTests.cpp
        murphi-translate/ModuleInterpreterTests.cpp
        murphi-translate/MurphiCodeGenTests.cpp
        codegen/FunctionDeclTests.cpp
        codegen/FormalParamsTests.cpp codegen/ExpressionTests.cpp codegen/StatementTests.cpp
        codegen/QuantifierTests.cpp codegen/MachineHandlerTests.cpp)

target_link_libraries(protogen-mlir-test gtest_main MLIRIR antlr4_static mlir-gen trans-murphi)

include(GoogleTest)
gtest_discover_tests(protogen-mlir-test)