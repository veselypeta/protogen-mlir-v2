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

TEST(ExpressionTests, DesignatorExprObjType){
  detail::Designator<detail::ExprID> designatorIdx{
      "newObj", "array", {"arr_index"}
  };
  detail::DesignatorExpr<decltype(designatorIdx), detail::ExprID> desExpr{
      designatorIdx,
      "object",
      {"theIndex"}
  };
  json data = desExpr;
  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("expression.tmpl");
  auto result = env.render(tmpl, data);

  ASSERT_STREQ(result.c_str(), "newObj[arr_index].theIndex");
}

TEST(ExpressionTests, DesignatorExprArrType){
  detail::Designator<detail::ExprID> designatorIdx{
      "newObj", "array", {"arr_index"}
  };
  detail::DesignatorExpr<decltype(designatorIdx), detail::ExprID> desExpr{
      designatorIdx,
      "array",
      {"next_idx"}
  };
  json data = desExpr;
  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("expression.tmpl");
  auto result = env.render(tmpl, data);

  ASSERT_STREQ(result.c_str(), "newObj[arr_index][next_idx]");
}

TEST(ExpressionTests, BinaryExpressionStrRef){

  json data = detail::BinaryExpr<detail::ExprID, detail::ExprID>{
      {"my_lhs_value"},
      {"my_rhs_value"},
      detail::BinaryOps.n_eq
  };
  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("expression.tmpl");
  auto result = env.render(tmpl, data);

  ASSERT_STREQ(result.c_str(), "my_lhs_value != my_rhs_value");
}

TEST(ExpressionTests, MultisetCountTest){
  json data = detail::MultisetCount{
      "i",
      detail::ExprID{"i_type"},
      detail::ExprID{"true"}
  };

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("expression.tmpl");
  auto result = env.render(tmpl, data);

  ASSERT_STREQ(result.c_str(), "MultisetCount(i:i_type, true)");
}