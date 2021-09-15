#include "translation/murphi/codegen/InjaEnvSingleton.h"
#include "translation/utils/utils.h"
#include <gtest/gtest.h>
#include <inja/inja.hpp>
#include <regex>

using namespace inja;
TEST(EnumTypeTest, RenderTemplate) {
  json data;
  data["decls"] = {"load", "store", "evict"};

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("enum_type.tmpl");
  auto result = env.render(tmpl, data);

  const std::regex enum_regex("enum\\{[a-z, A-Z]+(,[a-z, A-Z]+)*\\}");

  result.erase(std::remove_if(result.begin(), result.end(),
                              &translation::utils::isWhitespace),
               result.end());
  ASSERT_TRUE(std::regex_match(result.c_str(), enum_regex));
  ASSERT_NE(result.find("load"), std::string::npos);
  ASSERT_NE(result.find("store"), std::string::npos);
  ASSERT_NE(result.find("evict"), std::string::npos);
}