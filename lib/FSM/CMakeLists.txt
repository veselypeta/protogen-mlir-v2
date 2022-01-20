add_mlir_dialect_library(ProtogenFSM
        FSMDialect.cpp
        FSMOps.cpp
        FSMTypes.cpp
        FSMUtils.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/FSM

        DEPENDS
        MLIRFSMIncGen
        protocc-antlr4

        LINK_LIBS PUBLIC
        MLIRStandard
        MLIRIR
        PRIVATE antlr4_static
        protocc-antlr4

        )
add_sanitizers(ProtogenFSM)
add_subdirectory(Passes)