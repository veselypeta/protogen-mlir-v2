#include <gtest/gtest.h>
#include <inja/inja.hpp>
#include "translation/murphi/codegen/MurphiCodeGen.h"
#include "translation/utils/JSONValidation.h"



using namespace inja;
using namespace murphi;


TEST(QuantifierTests, ForEachQuantifier){

  detail::ForEachQuantifier<detail::ID> quant{"i", {"iterator_type"}};
  json data = quant;

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("quantifier.tmpl");
  auto result = env.render(tmpl, data);

  ASSERT_STREQ("i : iterator_type", result.c_str());
  // verify json
  std::string schema_path =
      std::string(JSONValidation::schema_base_directory) + "gen_Quantifier.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  ASSERT_TRUE(is_valid);
  ASSERT_FALSE(result.empty());
}

TEST(QuantifierTests, ForRangeQuantifier){

  detail::ForRangeQuantifier<detail::ExprID, detail::ExprID> quant{"i", {"start_range"}, {"end_range"}};
  json data = quant;

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("quantifier.tmpl");
  auto result = env.render(tmpl, data);

  ASSERT_STREQ("i := start_range to end_range", result.c_str());

  // verify json
  std::string schema_path =
      std::string(JSONValidation::schema_base_directory) + "gen_Quantifier.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  ASSERT_TRUE(is_valid);
}