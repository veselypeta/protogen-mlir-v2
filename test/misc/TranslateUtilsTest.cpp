#include <gtest/gtest.h>
#include "translation/utils/utils.h"

TEST(UtilsTests, DemangeTest){
  std::string mangledStateName = "cache_I_load_fwd_get_m";
  auto demnagledStateName = translation::utils::demangleState(mangledStateName);
  EXPECT_STREQ(demnagledStateName.c_str(), "I_load_fwd_get_m");
}

TEST(UtilsTests, MangleState){
  std::string unmangled = "I_load";
  auto mangled = translation::utils::mangleState(unmangled, "mangle_");
  EXPECT_STREQ(mangled.c_str(), "mangle_I_load");
}

TEST(UtilsTests, MangleCache){
  std::string unmangled = "I_load";
  auto mangled = translation::utils::mangleCacheState(unmangled);
  EXPECT_STREQ(mangled.c_str(), "cache_I_load");
}

TEST(UtilsTests, MangleDirectory){
  std::string unmangled = "I_load";
  auto mangled = translation::utils::mangleDirectoryState(unmangled);
  EXPECT_STREQ(mangled.c_str(), "directory_I_load");
}
