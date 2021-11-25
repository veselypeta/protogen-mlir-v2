//
// Created by petr on 25/11/2021.
//
#include "translation/murphi/codegen/InjaEnvSingleton.h"
#include "translation/murphi/codegen/MurphiCodeGen.h"
#include "translation/utils/JSONValidation.h"
#include <gtest/gtest.h>
#include <inja/inja.hpp>

using namespace inja;
using namespace murphi;

TEST(RuleTests, SimpleRule) {
  auto start_state = "cache_I_load";
  auto simpleRule = detail::SimpleRule{start_state, {}, {}, {}};
  auto equal_expr =
      detail::BinaryExpr<detail::Designator<detail::ExprID>, detail::ExprID>{
          {detail::cle_a, "object", {detail::c_state}},
          {"cache_I"},
          detail::BinaryOps.eq};
  auto proc_call = detail::ProcCall{
      std::string(detail::send_pref_f) + start_state,
      {detail::ExprID{detail::c_adr}, detail::ExprID{detail::c_mach}}};

  simpleRule.expr = equal_expr;
  simpleRule.statements.emplace_back(proc_call);

  json data = simpleRule;

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("rule.tmpl");
  auto result = env.render(tmpl, data);

  // verify json
  std::string schema_path = std::string(JSONValidation::schema_base_directory) +
                            "gen_RuleDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  EXPECT_TRUE(is_valid);

//  ASSERT_STREQ("", result.c_str());
  ASSERT_FALSE(result.empty());
}