add_library(translation-utils
        Utils.cpp
        ModuleInterpreter.cpp
        JSONValidation.cpp)

target_include_directories(translation-utils PUBLIC ${PROJECT_SOURCE_DIR}/include/translation/utils)
target_link_libraries(translation-utils ValiJSON::valijson)



