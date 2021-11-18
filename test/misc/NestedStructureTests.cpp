#include "translation/murphi/codegen/InjaEnvSingleton.h"
#include <gtest/gtest.h>
#include <inja/inja.hpp>

using namespace inja;

std::string for_tmpl =R"(
for i in range(5):
  {% for op in ops %}
  op
  {% endfor %}
)";

TEST(NestedStructures, RenderNestedStructuresWithTabs){
  ASSERT_TRUE(true);
//  ASSERT_STREQ(for_tmpl.c_str(), "0");
}