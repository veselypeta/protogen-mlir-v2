add_library(translation-utils
        utils.cpp
        ModuleInterpreter.cpp)

target_include_directories(translation-utils PUBLIC ${PROJECT_SOURCE_DIR}/include/translation/utils)



