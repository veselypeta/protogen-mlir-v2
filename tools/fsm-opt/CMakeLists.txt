set(LLVM_LINK_COMPONENTS Support)

add_llvm_executable(fsm-opt fsm-opt.cpp)
llvm_update_compile_flags(fsm-opt)

target_link_libraries(fsm-opt
        PRIVATE
        ${dialect_libs}
        ${translation_libs}
        ${LIBS}

        MLIRParser
        MLIRSupport
        MLIRIR
        MLIROptLib
        MLIRStandard
        MLIRTransforms
        MLIRLLVMIR
        FSMPasses

        ProtogenFSM
        )
