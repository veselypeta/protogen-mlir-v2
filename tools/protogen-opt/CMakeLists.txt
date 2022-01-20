set(LLVM_LINK_COMPONENTS
        Support
        )

add_llvm_executable(protogen-opt
        protogen-opt.cpp
        )
llvm_update_compile_flags(protogen-opt)

target_link_libraries(protogen-opt
        PRIVATE
        ${dialect_libs}
        ${translation_libs}
        ${LIBS}

        MLIRPCC

        MLIRParser
        MLIRSupport
        MLIRIR
        MLIROptLib
        MLIRStandard
        MLIRTransforms
        MLIRLLVMIR
        PCCPasses
        FSMPasses
        )