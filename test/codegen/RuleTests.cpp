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
      detail::BinaryExpr<detail::Designator, detail::ExprID>{
          {detail::cle_a, {detail::Indexer{"object", detail::ExprID{detail::c_state}}}},
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

  //  EXPECT_STREQ("", result.c_str());
  EXPECT_FALSE(result.empty());
}

TEST(RuleTests, RuleSet) {
  // m:OBJSET_cache -> Quantifier
  auto mach_quant = detail::ForEachQuantifier<detail::ID>{
      detail::c_mach, {detail::SetKey + detail::machines.cache.str()}};

  // basic simple rule
  auto start_state = "cache_I_load";
  auto simpleRule = detail::SimpleRule{start_state, {}, {}, {}};
  auto equal_expr =
      detail::BinaryExpr<detail::Designator, detail::ExprID>{
          {detail::cle_a, {detail::Indexer{"object", detail::ExprID{detail::c_state}}}},
          {"cache_I"},
          detail::BinaryOps.eq};
  auto proc_call = detail::ProcCall{
      std::string(detail::send_pref_f) + start_state,
      {detail::ExprID{detail::c_adr}, detail::ExprID{detail::c_mach}}};

  simpleRule.expr = equal_expr;
  simpleRule.statements.emplace_back(proc_call);

  auto ruleset = detail::RuleSet{{mach_quant, mach_quant}, {simpleRule}};

  // render template
  json data = ruleset;

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("rule.tmpl");
  auto result = env.render(tmpl, data);

  // verify json
  std::string schema_path = std::string(JSONValidation::schema_base_directory) +
                            "gen_RuleDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  EXPECT_TRUE(is_valid);

  //  EXPECT_STREQ("", result.c_str());
  EXPECT_FALSE(result.empty());
}

TEST(RuleTests, AliasRule) {
  // basic simple rule
  auto start_state = "cache_I_load";
  auto simpleRule = detail::SimpleRule{start_state, {}, {}, {}};
  auto equal_expr =
      detail::BinaryExpr<detail::Designator, detail::ExprID>{
          {detail::cle_a, {detail::Indexer{"object", detail::ExprID{detail::c_state}}}},
          {"cache_I"},
          detail::BinaryOps.eq};
  auto proc_call = detail::ProcCall{
      std::string(detail::send_pref_f) + start_state,
      {detail::ExprID{detail::c_adr}, detail::ExprID{detail::c_mach}}};

  simpleRule.expr = equal_expr;
  simpleRule.statements.emplace_back(proc_call);

  auto alias_rule = detail::AliasRule{
      "my_alias", detail::ExprID{"expr_to_be_aliased"}, {simpleRule}};

  json data = alias_rule;

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("rule.tmpl");
  auto result = env.render(tmpl, data);

  // verify json
  std::string schema_path = std::string(JSONValidation::schema_base_directory) +
                            "gen_RuleDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  EXPECT_TRUE(is_valid);

  //  EXPECT_STREQ("", result.c_str());
  EXPECT_FALSE(result.empty());
}

TEST(RuleTests, ChooseRule) {

  auto chooseRule = detail::ChooseRule{
      "i", detail::ExprID{"n"}, {detail::CPUEventRule{"start_state", "evict"}}};

  json data = chooseRule;

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("rule.tmpl");
  auto result = env.render(tmpl, data);

  // verify json
  std::string schema_path = std::string(JSONValidation::schema_base_directory) +
                            "gen_RuleDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  EXPECT_TRUE(is_valid);

  //  EXPECT_STREQ("", result.c_str());
  EXPECT_FALSE(result.empty());
}

TEST(RuleTests, StartState) {

  auto ss = detail::StartState{
      "my start state",
      {detail::ForwardDecl<detail::VarDecl<detail::ID>>{"var",
                                                        {"msg", {"message"}}}},
      {detail::Assert<detail::ExprID>{{"is_valid"}, "the message"}}};

  json data = ss;

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("rule.tmpl");
  auto result = env.render(tmpl, data);

  // verify json
  std::string schema_path = std::string(JSONValidation::schema_base_directory) +
                            "gen_RuleDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  EXPECT_TRUE(is_valid);

//  EXPECT_STREQ("", result.c_str());
  EXPECT_FALSE(result.empty());
}