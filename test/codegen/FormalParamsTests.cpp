#include "translation/murphi/codegen/InjaEnvSingleton.h"
#include <gtest/gtest.h>
#include <inja/inja.hpp>
#include "translation/murphi/codegen/MurphiCodeGen.h"


using namespace inja;
using namespace murphi;

TEST(FormalParamTests, RenderTemplate){
  json data;
  data["params"] = {
      detail::Formal<detail::ID>{"first_param", {{"first_param_type"}}},
      detail::Formal<detail::ID>{"second_param", {{"second_param_type"}}},
      detail::Formal<detail::ID>{"third_param", {{"third_param_type"}}},
  };

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("formal_params.tmpl");
  auto result = env.render(tmpl, data);

  ASSERT_FALSE(result.empty());
}

TEST(FormalParamTests, EmptyData){
  json data;
  data["params"] = json::array();

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("formal_params.tmpl");
  auto result = env.render(tmpl, data);

  ASSERT_TRUE(result.empty());
}