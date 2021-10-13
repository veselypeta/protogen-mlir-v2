add_mlir_dialect_library(PCCPasses
        PassRegistration.cpp
        TransientStatesPass.cpp

        DEPENDS
        mlir-headers
        LINK_LIBS PUBLIC
        MLIRIR
        )