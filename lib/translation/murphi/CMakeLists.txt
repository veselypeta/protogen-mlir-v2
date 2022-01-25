add_mlir_dialect_library(trans-murphi STATIC
        PCCTranslate.cpp
        FSMTranslate.cpp

        codegen/InjaEnvSingleton.cpp
        codegen/MurphiCodeGen.cpp
        codegen/FSMDialectInterpreter.cpp
        codegen/Boilerplate.cpp
        codegen/FSMOperationConverter.cpp
        codegen/MurphiStructs.cpp
        codegen/MurphiConstants.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/translation/murphi

        LINK_LIBS PUBLIC
        MLIRIR
        translation-utils
        )

target_link_libraries(trans-murphi PUBLIC translation-utils)
