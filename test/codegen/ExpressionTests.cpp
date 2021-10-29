#include "translation/murphi/codegen/InjaEnvSingleton.h"
#include "translation/murphi/codegen/MurphiCodeGen.h"
#include <gtest/gtest.h>
#include <inja/inja.hpp>

using namespace inja;
using namespace murphi;
TEST(ExpressionTests, DesignatorObjectType) {
  json data =
      detail::Designator<detail::ExprID>{"myObj", "object", {"theIndex"}};

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("expression.tmpl");
  auto result = env.render(tmpl, data);

  ASSERT_STREQ(result.c_str(), "myObj.theIndex");
}

TEST(ExpressionTests, DesignatorArrayType) {
  json data = detail::Designator<detail::ExprID>{
      "newObj", "array", {"arr_index"}
  };

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("expression.tmpl");
  auto result = env.render(tmpl, data);

  ASSERT_STREQ(result.c_str(), "newObj[arr_index]");
}