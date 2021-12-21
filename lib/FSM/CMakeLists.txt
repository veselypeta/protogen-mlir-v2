add_mlir_dialect_library(ProtogenFSM
        FSMDialect.cpp
        FSMOps.cpp
        FSMTypes.cpp

        DEPENDS
        MLIRFSMIncGen

        LINK_LIBS PUBLIC
        MLIRStandard
        MLIRIR

        )
add_sanitizers(ProtogenFSM)