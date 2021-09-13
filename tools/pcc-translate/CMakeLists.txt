set(LLVM_LINK_COMPONENTS
        Support
        )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_executable(pcc-translate
        pcc-translate.cpp
        TranslatePCCToMLIR.cpp
        )

llvm_update_compile_flags(pcc-translate)
target_link_libraries(pcc-translate
        PRIVATE
        ${dialect_libs}
        ${translation_libs}
        ${LIBS}
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRSPIRV
        MLIRTranslation
        MLIRSupport
        antlr4_static
        protocc-antlr4
        )
mlir_check_link_libraries(pcc-translate)
