#include <gtest/gtest.h>
#include <inja/inja.hpp>
#include "translation/murphi/codegen/MurphiCodeGen.h"


using namespace inja;
using namespace murphi;


TEST(QuantifierTests, ForEachQuantifier){

  detail::ForEachQuantifier<detail::ID> quant{"i", {"iterator_type"}};
  json data = quant;

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("quantifier.tmpl");
  auto result = env.render(tmpl, data);

  ASSERT_STREQ("i : iterator_type", result.c_str());
}

TEST(QuantifierTests, ForRangeQuantifier){

  detail::ForRangeQuantifier<detail::ExprID, detail::ExprID> quant{"i", {"start_range"}, {"end_range"}};
  json data = quant;

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("quantifier.tmpl");
  auto result = env.render(tmpl, data);

  ASSERT_STREQ("i := start_range to end_range", result.c_str());
}