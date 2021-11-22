#include "translation/murphi/codegen/InjaEnvSingleton.h"
#include "translation/utils/JSONValidation.h"
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

  // verify json
  std::string schema_path =
      std::string(JSONValidation::schema_base_directory) + "gen_Formal.json";
  for(auto &d : data["params"]){
    bool is_valid = JSONValidation::validate_json(schema_path, d);
    ASSERT_TRUE(is_valid);
  }
}

TEST(FormalParamTests, EmptyData){
  json data;
  data["params"] = json::array();

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("formal_params.tmpl");
  auto result = env.render(tmpl, data);

  ASSERT_TRUE(result.empty());
}