#include "translation/murphi/codegen/InjaEnvSingleton.h"
#include "translation/murphi/codegen/MurphiCodeGen.h"
#include "translation/utils/JSONValidation.h"
#include <gtest/gtest.h>
#include <inja/inja.hpp>

using namespace inja;
using namespace murphi;

TEST(ExpressionTests, ExprIdTest) {

  json data = detail::ExprID{"anID"};

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("expression.tmpl");
  auto result = env.render(tmpl, data);

  ASSERT_STREQ(result.c_str(), "anID");

  // verify json
  std::string schema_path = std::string(JSONValidation::schema_base_directory) +
                            "gen_ExpressionDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  ASSERT_TRUE(is_valid);
}

TEST(ExpressionTests, SimpleDesWithIndexer) {
  auto simpleDes = detail::Designator{
      "my_object", {detail::Indexer{"object", detail::ExprID{"indexing"}}}};

  json data = simpleDes;
  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("expression.tmpl");
  auto result = env.render(tmpl, data);

  EXPECT_STREQ(result.c_str(), "my_object.indexing");

  // verify json
  std::string schema_path = std::string(JSONValidation::schema_base_directory) +
                            "gen_ExpressionDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  EXPECT_TRUE(is_valid);
}

TEST(ExpressionTests, SimpleDesWithArrayIndexer) {
  auto simpleDes = detail::Designator{
      "my_object", {detail::Indexer{"array", detail::ExprID{"indexing"}}}};

  json data = simpleDes;
  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("expression.tmpl");
  auto result = env.render(tmpl, data);

  EXPECT_STREQ(result.c_str(), "my_object[indexing]");

  // verify json
  std::string schema_path = std::string(JSONValidation::schema_base_directory) +
                            "gen_ExpressionDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  EXPECT_TRUE(is_valid);
}

TEST(ExpressionTests, SimpleDesWithMultipleIndexes){
  auto simpleDes = detail::Designator{
    "cle",
    {
      detail::Indexer{"array", detail::ExprID{"0"}},
          detail::Indexer{"object", detail::ExprID{"obj_id"}},
          detail::Indexer{"array", detail::ExprID{"n"}}
    }
  };

  json data = simpleDes;
  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("expression.tmpl");
  auto result = env.render(tmpl, data);

  EXPECT_STREQ(result.c_str(), "cle[0].obj_id[n]");

  // verify json
  std::string schema_path = std::string(JSONValidation::schema_base_directory) +
                            "gen_ExpressionDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  EXPECT_TRUE(is_valid);
}

TEST(ExpressionTests, SimpleDesWOIndexing) {
  auto simpleDes = detail::Designator{
      "my_object", {}};

  json data = simpleDes;
  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("expression.tmpl");
  auto result = env.render(tmpl, data);

  EXPECT_STREQ(result.c_str(), "my_object");

  // verify json
  std::string schema_path = std::string(JSONValidation::schema_base_directory) +
                            "gen_ExpressionDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  EXPECT_TRUE(is_valid);
}

TEST(ExpressionTests, BinaryExpressionStrRef) {

  json data = detail::BinaryExpr<detail::ExprID, detail::ExprID>{
      {"my_lhs_value"}, {"my_rhs_value"}, detail::BinaryOps.n_eq};
  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("expression.tmpl");
  auto result = env.render(tmpl, data);

  ASSERT_STREQ(result.c_str(), "my_lhs_value != my_rhs_value");

  // verify json
  std::string schema_path = std::string(JSONValidation::schema_base_directory) +
                            "gen_ExpressionDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  ASSERT_TRUE(is_valid);
}

TEST(ExpressionTests, MultisetCountTest) {
  json data = detail::MultisetCount{"i", detail::ExprID{"i_type"},
                                    detail::ExprID{"true"}};

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("expression.tmpl");
  auto result = env.render(tmpl, data);

  ASSERT_STREQ(result.c_str(), "MultisetCount(i:i_type, true)");

  // verify json
  std::string schema_path = std::string(JSONValidation::schema_base_directory) +
                            "gen_ExpressionDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  ASSERT_TRUE(is_valid);
}

TEST(ExpressionTests, ProcCallExpr) {
  detail::ProcCallExpr procCallExpr{
      "my_func", {detail::ExprID{"param1"}, detail::ExprID{"param2"}}};

  json data = procCallExpr;
  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("expression.tmpl");
  auto result = env.render(tmpl, data);

  ASSERT_STREQ(result.c_str(), "my_func(param1, param2)");

  // verify json
  std::string schema_path = std::string(JSONValidation::schema_base_directory) +
                            "gen_ExpressionDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  ASSERT_TRUE(is_valid);
}

TEST(ExpressionTests, NegExpr) {
  auto negExpr = detail::NegExpr{detail::ExprID{"to_be_negated"}};
  json data = negExpr;
  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("expression.tmpl");
  auto result = env.render(tmpl, data);

  ASSERT_STREQ(result.c_str(), "!to_be_negated");

  // verify json
  std::string schema_path = std::string(JSONValidation::schema_base_directory) +
                            "gen_ExpressionDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  ASSERT_TRUE(is_valid);
}

TEST(ExpressionTests, ForAll){
  auto forall = detail::ForAll<detail::ForEachQuantifier<detail::ID>>{
    {"i", detail::ID{"range"}},
        detail::ExprID{"myExpr"}
  };
  json data = forall;
  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("expression.tmpl");
  auto result = env.render(tmpl, data);

  auto expected_str = "forall i : range do\n"
                      "    myExpr\n"
                      "endforall\n";
  ASSERT_STREQ(result.c_str(), expected_str);

  // verify json
  std::string schema_path = std::string(JSONValidation::schema_base_directory) +
                            "gen_ExpressionDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  ASSERT_TRUE(is_valid);
}

TEST(ExpressionTests, ParensExpr){
  auto pExtr = detail::ParensExpr{detail::ExprID{"in the parens"}};
  json data = pExtr;
  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("expression.tmpl");
  auto result = env.render(tmpl, data);

  auto expected_str = "( in the parens )";
  ASSERT_STREQ(result.c_str(), expected_str);

  // verify json
  std::string schema_path = std::string(JSONValidation::schema_base_directory) +
                            "gen_ExpressionDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  ASSERT_TRUE(is_valid);
}