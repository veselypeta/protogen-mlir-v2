#include "translation/murphi/codegen/InjaEnvSingleton.h"
#include <gtest/gtest.h>
#include <inja/inja.hpp>
#include <regex>

using namespace inja;

TEST(SubRangeTests, SimpleSubrange){
  json data;
  data["typeId"] = "sub_range";
  data["type"] = {
      {"start", 0},
      {"stop", 100},
  };
  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("type_expr.tmpl");
  const auto result = env.render(tmpl, data);
  std::regex re(R"(\d+\.\.\d+)");
  ASSERT_TRUE(std::regex_match(result.c_str(), re));
}