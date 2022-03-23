add_mlir_dialect_library(FSMPasses
        StallingOptimizationPass.cpp
        PassRegistration.cpp
        NonStallingOptimizationPass.cpp
        FullOptPipeline.cpp
        RemoveAwaitPass.cpp
        DEPENDS
        mlir-headers
        LINK_LIBS PUBLIC
        MLIRIR
        ProtogenFSM
        )