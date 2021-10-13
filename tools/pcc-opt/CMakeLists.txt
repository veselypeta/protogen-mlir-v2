set(LLVM_LINK_COMPONENTS
        Support
        )

add_llvm_executable(pcc-opt
        pcc-opt.cpp
        )
llvm_update_compile_flags(pcc-opt)

target_link_libraries(pcc-opt
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
        )