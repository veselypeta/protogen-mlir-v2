#include "translation/murphi/codegen/InjaEnvSingleton.h"
#include "translation/murphi/codegen/MurphiCodeGen.h"
#include "translation/utils/JSONValidation.h"
#include <gtest/gtest.h>
#include <inja/inja.hpp>
using namespace inja;
using namespace murphi;

detail::SimpleRule getSimpleRule() {

  detail::SimpleRule sr{
      "simple_rule",
      detail::BinaryExpr<detail::ExprID, detail::ExprID>{
          {"lhs"}, {"rhs"}, detail::BinaryOps.eq},
      {detail::ForwardDecl<detail::VarDecl<detail::ID>>{
          "var", {"new_var", {"var_type"}}}},
      {detail::Assignment<detail::Designator, detail::ExprID>{
          {"cle", {detail::Indexer{"object", detail::ExprID{"State"}}}},
          {"cache_M"}}}};
  return sr;
}

TEST(MachineHandler, BasicTest) {
  auto mach_handler = detail::MachineHandler{"cache", {}};

  json data = mach_handler;
  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("proc_decl.tmpl");
  auto result = env.render(tmpl, data);

  // verify json
  std::string schema_path =
      std::string(JSONValidation::schema_base_directory) + "gen_ProcDecl.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  ASSERT_TRUE(is_valid);

  //  ASSERT_STREQ("", result.c_str());
  ASSERT_FALSE(result.empty());
}

TEST(CPUEventHandler, BasicTest) {
  auto cpu_handle =
      detail::CPUEventHandler{detail::machines.cache.str() + "_I_load", {}};

  json data = cpu_handle;
  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("proc_decl.tmpl");
  auto result = env.render(tmpl, data);

  // verify json
  std::string schema_path =
      std::string(JSONValidation::schema_base_directory) + "gen_ProcDecl.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  EXPECT_TRUE(is_valid);

  //  EXPECT_STREQ("", result.c_str());
  EXPECT_FALSE(result.empty());
}

TEST(CacheRuleHandler, BasicTest) {
  // Create a basic rule

  auto cpu_handle = detail::CacheRuleHandler{{getSimpleRule()}};

  json data = cpu_handle;
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

TEST(CPUEventRuleTest, BasicTest) {

  auto cpuEventRule = detail::CPUEventRule{"state_I", "evict"};
  json data = cpuEventRule;

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

TEST(OrderedRuleset, BasicTest) {
  auto ordRuleset = detail::OrderedRuleset{"fwd"};

  json data = ordRuleset;

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

TEST(UnorderedRuleset, BasicTest) {
  auto unord_ruleset = detail::UnorderedRuleset{"fwd"};
  json data = unord_ruleset;
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

TEST(RuleTest, SWMRInvariantTest) {
  auto inv = detail::SWMRInvariant{};
  json data = inv;
  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("rule.tmpl");
  auto result = env.render(tmpl, data);

  // verify json
  std::string schema_path = std::string(JSONValidation::schema_base_directory) +
                            "gen_RuleDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  EXPECT_TRUE(is_valid);
  auto expectedStr = "invariant \"SWMR Invariant\"\n"
                     "    forall c1 : OBJSET_cache do\n"
                     "        forall c2 : OBJSET_cache do\n"
                     "            forall a : Address do\n"
                     "                ( c1 != c2 & i_cache[c1][a].State = cache_M ) -> ( i_cache[c2][a].State != cache_M )\n"
                     "            endforall\n"
                     "        endforall\n"
                     "    endforall;\n";
  EXPECT_STREQ(expectedStr, result.c_str());
  EXPECT_FALSE(result.empty());
}