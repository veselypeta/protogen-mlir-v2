add_mlir_dialect_library(ProtogenFSM
        FSMDialect.cpp

        DEPENDS
        MLIRFSMIncGen

        LINK_LIBS PUBLIC
        MLIRStandard
        MLIRIR

        )