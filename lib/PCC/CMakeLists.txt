add_mlir_dialect_library(
        MLIRPCC
        PCCDialect.cpp
        PCCOps.cpp
        PCCTypes.cpp
        PCCAttributes.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/PCC

        DEPENDS
        MLIRPCCIncGen
        protocc-antlr4


        LINK_LIBS PUBLIC MLIRIR
        PRIVATE antlr4_static
        protocc-antlr4
)
add_subdirectory(Passes)