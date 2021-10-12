set(LLVM_LINK_COMPONENTS
        Support
        )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_mlir_dialect_library(
        mlir-gen
        mlirGen.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/PCC

        DEPENDS
        protogen-headers
        protocc-antlr4

        LINK_LIBS PRIVATE
        ${dialect_libs}
        ${translation_libs}
        ${LIBS}
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRTranslation
        MLIRSupport
        antlr4_static
        protocc-antlr4
)
