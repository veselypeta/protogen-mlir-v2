#include "translation/murphi/codegen/InjaEnvSingleton.h"
#include <gtest/gtest.h>
#include <inja/inja.hpp>
#include <regex>

using namespace inja;

TEST(ArrayTypeTests, simple_array ){
  json data;
  data["typeId"] = "array";
  data["type"] = {
    {"index", {{"typeId", "ID"}, {"type", "CL"}}},
    {"type", {{"typeId", "ID"}, {"type", "Machines"}}}
  };

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("type_expr.tmpl");
  const auto result = env.render(tmpl, data);


  ASSERT_NE(result.find("CL"), std::string::npos);
  ASSERT_NE(result.find("Machines"), std::string::npos);

  std::regex re(R"(array \[ \w+ \] of \w+)");
  ASSERT_TRUE(std::regex_match(result.c_str(), re));
}