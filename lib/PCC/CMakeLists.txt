add_mlir_dialect_library(
    MLIRPCC
    PCCDialect.cpp
    PCCOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/PCC

    DEPENDS
    MLIRPCCOpsIncGen

    LINK_LIBS PUBLIC MLIRIR
)