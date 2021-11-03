add_mlir_dialect_library(PCCPasses
        PassRegistration.cpp
        TransientStatesPass.cpp
        EndStatesPass.cpp
        StableCompilationPipeline.cpp
        MutexOpsPass.cpp
        StallingProtogenPass.cpp
        StallingCompilationPipeline.cpp

        DEPENDS
        mlir-headers
        LINK_LIBS PUBLIC
        MLIRIR
        )