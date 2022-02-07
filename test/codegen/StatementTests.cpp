#include "translation/murphi/codegen/InjaEnvSingleton.h"
#include "translation/murphi/codegen/MurphiCodeGen.h"
#include "translation/utils/JSONValidation.h"
#include <gtest/gtest.h>
#include <inja/inja.hpp>

using namespace inja;
using namespace murphi;
TEST(StatementTests, AssignmentStatement) {
  json data = detail::Assignment<detail::Designator, detail::ExprID>{
      {"msg", {detail::Indexer{"object", detail::ExprID{"address"}}}}, {"val"}};

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("statement.tmpl");
  auto result = env.render(tmpl, data);

  // verify json
  std::string schema_path = std::string(JSONValidation::schema_base_directory) +
                            "gen_StatementDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  EXPECT_TRUE(is_valid);

  EXPECT_STREQ(result.c_str(), "msg.address := val;");
}

TEST(StatementTests, AssertStmt) {
  json data = detail::Assert<detail::ExprID>{{"value_to_be_tested"},
                                             "assertion failed!"};

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("statement.tmpl");
  auto result = env.render(tmpl, data);

  EXPECT_STREQ(result.c_str(),
               "assert( value_to_be_tested ) \"assertion failed!\";");

  // verify json
  std::string schema_path = std::string(JSONValidation::schema_base_directory) +
                            "gen_StatementDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  EXPECT_TRUE(is_valid);
}

TEST(StatementTests, ForStmt) {
  detail::ForStmt<detail::ForEachQuantifier<detail::ID>> forStmt{
      {"i", {"type"}}, {}};
  json assert_stmt = detail::Assert<detail::ExprID>{{"value_to_be_tested"},
                                                    "assertion failed!"};
  forStmt.stmts.push_back(assert_stmt);
  json data = forStmt;
  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("statement.tmpl");
  auto result = env.render(tmpl, data);
  EXPECT_FALSE(result.empty());

  // verify json
  std::string schema_path = std::string(JSONValidation::schema_base_directory) +
                            "gen_StatementDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  EXPECT_TRUE(is_valid);
}

TEST(StatementTests, IfStmt) {
  detail::IfStmt<detail::ExprID> ifstmt{{"true"}, {}, {}};

  json assert_stmt = detail::Assert<detail::ExprID>{{"value_to_be_tested"},
                                                    "assertion failed!"};
  ifstmt.thenStmts.emplace_back(assert_stmt);

  json data = ifstmt;

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("statement.tmpl");
  auto result = env.render(tmpl, data);
  EXPECT_FALSE(result.empty());

  // verify json
  std::string schema_path = std::string(JSONValidation::schema_base_directory) +
                            "gen_StatementDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  EXPECT_TRUE(is_valid);
}

TEST(StatementTests, IfStmt_withelse) {
  detail::IfStmt<detail::ExprID> ifstmt{{"true"}, {}, {}};

  json assert_stmt = detail::Assert<detail::ExprID>{{"value_to_be_tested"},
                                                    "assertion failed!"};
  ifstmt.thenStmts.emplace_back(assert_stmt);
  ifstmt.elseStmts.emplace_back(assert_stmt);

  json data = ifstmt;

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("statement.tmpl");
  auto result = env.render(tmpl, data);
  EXPECT_FALSE(result.empty());

  // verify json
  std::string schema_path = std::string(JSONValidation::schema_base_directory) +
                            "gen_StatementDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  EXPECT_TRUE(is_valid);
}

TEST(StatementTests, UndefineStmt) {
  detail::Designator des{"obj",
                         {detail::Indexer{"array", detail::ExprID{"index"}}}};
  detail::UndefineStmt<decltype(des)> undefineStmt{des};
  json data = undefineStmt;

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("statement.tmpl");
  auto result = env.render(tmpl, data);

  EXPECT_STREQ("undefine obj[index];", result.c_str());

  // verify json
  std::string schema_path = std::string(JSONValidation::schema_base_directory) +
                            "gen_StatementDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  EXPECT_TRUE(is_valid);
}

TEST(StatementTests, UndefineStmtWithDesignator) {
  detail::UndefineStmt<detail::Designator> undefineStmt{
      {"obj", {detail::Indexer{"array", detail::ExprID{"index"}}}}};

  json data = undefineStmt;

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("statement.tmpl");
  auto result = env.render(tmpl, data);

  EXPECT_STREQ("undefine obj[index];", result.c_str());

  // verify json
  std::string schema_path = std::string(JSONValidation::schema_base_directory) +
                            "gen_StatementDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  EXPECT_TRUE(is_valid);
}

TEST(StatementTests, ProcCallStmt) {
  json data = detail::ProcCall{
      "myfn", {detail::ExprID{"param1"}, detail::ExprID{"param2"}}};
  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("statement.tmpl");
  auto result = env.render(tmpl, data);

  EXPECT_STREQ("myfn(param1, param2);", result.c_str());

  // verify json
  std::string schema_path = std::string(JSONValidation::schema_base_directory) +
                            "gen_StatementDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  EXPECT_TRUE(is_valid);
}

TEST(StatementTests, AliasStmt) {
  auto aliasStmt =
      detail::AliasStmt<detail::ExprID>{"my_alias", {"to_be_aliased"}, {}};
  auto assignmentStmt = detail::Assignment<detail::Designator, detail::ExprID>{
      {"msg", {detail::Indexer{"object", detail::ExprID{"address"}}}}, {"val"}};
  aliasStmt.statements.emplace_back(assignmentStmt);

  json data = aliasStmt;
  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("statement.tmpl");
  auto result = env.render(tmpl, data);

  //  EXPECT_STREQ("alias my_alias : to_be_aliased do end;", result.c_str());

  // verify json
  std::string schema_path = std::string(JSONValidation::schema_base_directory) +
                            "gen_StatementDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  EXPECT_TRUE(is_valid);
}

TEST(StatementTest, CaseStmt) {
  json stateAssignment = detail::Assignment<detail::Designator, detail::ExprID>{
      {"cache", {detail::Indexer{"object", detail::ExprID{"State"}}}},
      {"cache_M"}};
  auto case1 = detail::CaseStmt{detail::ExprID{"theState"}, {stateAssignment}};
  json data = case1;
  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("case.tmpl");
  auto result = env.render(tmpl, data);

  // verify json
  std::string schema_path =
      std::string(JSONValidation::schema_base_directory) + "gen_CaseStmt.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  EXPECT_TRUE(is_valid);
  EXPECT_FALSE(result.empty());
}

TEST(StatementTest, SwitchStmt) {

  auto switchStmt =
      detail::SwitchStmt{detail::ExprID{"to_be_switched_over"}, {}, {}};
  json stateAssignment = detail::Assignment<detail::Designator, detail::ExprID>{
      {"cache", {detail::Indexer{"object", detail::ExprID{"State"}}}},
      {"cache_M"}};
  auto case1 = detail::CaseStmt{detail::ExprID{"theState"}, {stateAssignment}};
  switchStmt.cases.emplace_back(case1);
  switchStmt.cases.emplace_back(case1);
  switchStmt.elseStatements.emplace_back(stateAssignment);

  json data = switchStmt;

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("statement.tmpl");
  auto result = env.render(tmpl, data);

  // verify json
  std::string schema_path = std::string(JSONValidation::schema_base_directory) +
                            "gen_StatementDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  EXPECT_TRUE(is_valid);
  EXPECT_FALSE(result.empty());
}

TEST(StatementTest, ReturnStmt) {
  auto returnStmt = detail::ReturnStmt{detail::ExprID{"true"}};

  json data = returnStmt;
  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("statement.tmpl");
  auto result = env.render(tmpl, data);

  // verify json
  std::string schema_path = std::string(JSONValidation::schema_base_directory) +
                            "gen_StatementDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  EXPECT_TRUE(is_valid);
  EXPECT_FALSE(result.empty());
  EXPECT_STREQ(result.c_str(), "return true;\n");
}

TEST(StatementTest, MultisetRemove) {
  auto msRem = detail::MultisetRemovePred{"i", detail::ExprID{"val"},
                                          detail::ExprID{"true"}};

  json data = msRem;
  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("statement.tmpl");
  auto result = env.render(tmpl, data);

  // verify json
  std::string schema_path = std::string(JSONValidation::schema_base_directory) +
                            "gen_StatementDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  EXPECT_TRUE(is_valid);
  EXPECT_FALSE(result.empty());
  EXPECT_STREQ(result.c_str(), "MultisetRemovePred(i:val, true);");
}