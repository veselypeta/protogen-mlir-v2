
set(LLVM_LINK_COMPONENTS
        Support
        )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_executable(murphi-translate
        murphi-translate.cpp
        DEPENDS
        protogen-headers
        )
llvm_update_compile_flags(murphi-translate)


target_link_libraries(murphi-translate
        PRIVATE
        ${dialect_libs}
        ${translation_libs}
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRSPIRV
        MLIRTranslation
        MLIRSupport
        )

mlir_check_link_libraries(murphi-translate)
add_sanitizers(murphi-translate)