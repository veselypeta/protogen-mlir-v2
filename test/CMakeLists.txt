enable_testing()

add_executable(protogen-mlir-test
        mlirGen/mlirGenTest.cpp
        misc/NestedStructureTests.cpp
        misc/TranslateUtilsTest.cpp
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
        murphi-translate/FSMDialectInterpreterTests.cpp
        murphi-translate/MurphiAssemblerTests.cpp
        codegen/FunctionDeclTests.cpp
        codegen/FormalParamsTests.cpp codegen/ExpressionTests.cpp codegen/StatementTests.cpp
        codegen/QuantifierTests.cpp codegen/MachineHandlerTests.cpp codegen/RuleTests.cpp

        # fsm dialect tests
        FSM_Dialect/OpConstructionTests.cpp FSM_Dialect/OpTypeTests.cpp FSM_Dialect/OpHelper.h FSM_Dialect/UtilitiesTests.cpp)

target_link_libraries(protogen-mlir-test gtest_main MLIRIR antlr4_static mlir-gen trans-murphi ProtogenFSM)

include(GoogleTest)
gtest_discover_tests(protogen-mlir-test)