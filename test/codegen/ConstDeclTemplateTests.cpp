#include "translation/murphi/codegen/InjaEnvSingleton.h"
#include <gtest/gtest.h>
#include <inja/inja.hpp>

using namespace inja;

TEST(ConstDeclTemplate, RendersCorrectly) {
  json data;
  auto env = InjaEnvSingleton::getInstance();
  auto tmpl = env.parse_template("const_decl.tmpl");

  data["const_decl_tmpl_data"].push_back(
      {{"id", "NrCaches"}, {"value", 5}});

  data["const_decl_tmpl_data"].push_back(
      {{"id", "AddressSpace"}, {"value", 66}});

  auto result = env.render(tmpl, data);

  ASSERT_TRUE(result.find("NrCaches : 5;") != std::string::npos);
  ASSERT_TRUE(result.find("AddressSpace : 66;") != std::string::npos);
}
