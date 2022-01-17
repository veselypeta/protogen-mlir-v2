add_mlir_dialect_library(FSMPasses
        StallingOptimizationPass.cpp
        PassRegistration.cpp
        DEPENDS
        mlir-headers
        LINK_LIBS PUBLIC
        MLIRIR
        )