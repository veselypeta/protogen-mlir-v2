#include "translation/murphi/codegen/InjaEnvSingleton.h"
#include <gtest/gtest.h>

TEST(InjaEnvSingleton, SingletonWorkCorrectly) {
  ASSERT_EQ(
      &InjaEnvSingleton::getInstance(),
      &InjaEnvSingleton::getInstance()
      );
}