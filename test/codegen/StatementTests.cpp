#include "translation/murphi/codegen/InjaEnvSingleton.h"
#include "translation/murphi/codegen/MurphiCodeGen.h"
#include <gtest/gtest.h>
#include <inja/inja.hpp>

using namespace inja;
using namespace murphi;
TEST(StatementTests, AssignmentStatement) {
  json data = detail::Assignment<detail::Designator<detail::ExprID>, detail::ExprID>{
      {"msg", "object", {"address"}}, {"val"}};

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("statement.tmpl");
  auto result = env.render(tmpl, data);

  ASSERT_STREQ(result.c_str(), "msg.address := val;");
}

TEST(StatementTests, AssertStmt) {
  json data = detail::Assert<detail::ExprID>{{"value_to_be_tested"},
                                             "assertion failed!"};

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("statement.tmpl");
  auto result = env.render(tmpl, data);

  ASSERT_STREQ(result.c_str(),
               "assert( value_to_be_tested ) \"assertion failed!\";");
}