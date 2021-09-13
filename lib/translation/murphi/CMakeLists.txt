add_mlir_dialect_library(trans-murphi
        TranslatePCCToMurphi.cpp
        codegen/InjaEnvSingleton.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/translation/murphi

        LINK_LIBS PUBLIC
        MLIRIR
        translation-utils
        )

target_link_libraries(trans-murphi PUBLIC MLIRIR)
