#include "translation/murphi/codegen/InjaEnvSingleton.h"
#include "translation/murphi/codegen/MurphiCodeGen.h"
#include "translation/utils/JSONValidation.h"
#include <gtest/gtest.h>
#include <inja/inja.hpp>

using namespace inja;
using namespace murphi;
TEST(StatementTests, AssignmentStatement) {
  json data = detail::Assignment<detail::Designator<detail::ExprID>, detail::ExprID>{
      {"msg", "object", {"address"}}, {"val"}};

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("statement.tmpl");
  auto result = env.render(tmpl, data);

  // verify json
  std::string schema_path =
      std::string(JSONValidation::schema_base_directory) + "gen_StatementDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  ASSERT_TRUE(is_valid);

  ASSERT_STREQ(result.c_str(), "msg.address := val;");
}

TEST(StatementTests, AssertStmt) {
  json data = detail::Assert<detail::ExprID>{{"value_to_be_tested"},
                                             "assertion failed!"};

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("statement.tmpl");
  auto result = env.render(tmpl, data);

  ASSERT_STREQ(result.c_str(),
               "assert( value_to_be_tested ) \"assertion failed!\";");

  // verify json
  std::string schema_path =
      std::string(JSONValidation::schema_base_directory) + "gen_StatementDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  ASSERT_TRUE(is_valid);
}

TEST(StatementTests, ForStmt){
  detail::ForStmt<detail::ForEachQuantifier<detail::ID>> forStmt{{"i", {"type"}}};
  json assert_stmt = detail::Assert<detail::ExprID>{{"value_to_be_tested"},
                                             "assertion failed!"};
  forStmt.stmts.push_back(assert_stmt);
  json data = forStmt;
  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("statement.tmpl");
  auto result = env.render(tmpl, data);
  ASSERT_FALSE(result.empty());

  // verify json
  std::string schema_path =
      std::string(JSONValidation::schema_base_directory) + "gen_StatementDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  ASSERT_TRUE(is_valid);
}


TEST(StatementTests, IfStmt){
  detail::IfStmt<detail::ExprID> ifstmt{{"true"}};

  json assert_stmt = detail::Assert<detail::ExprID>{{"value_to_be_tested"},
                                                    "assertion failed!"};
  ifstmt.thenStmts.emplace_back(assert_stmt);

  json data = ifstmt;

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("statement.tmpl");
  auto result = env.render(tmpl, data);
  ASSERT_FALSE(result.empty());

  // verify json
  std::string schema_path =
      std::string(JSONValidation::schema_base_directory) + "gen_StatementDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  ASSERT_TRUE(is_valid);
}

TEST(StatementTests, IfStmt_withelse){
  detail::IfStmt<detail::ExprID> ifstmt{{"true"}};

  json assert_stmt = detail::Assert<detail::ExprID>{{"value_to_be_tested"},
                                                    "assertion failed!"};
  ifstmt.thenStmts.emplace_back(assert_stmt);
  ifstmt.elseStmts.emplace_back(assert_stmt);

  json data = ifstmt;

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("statement.tmpl");
  auto result = env.render(tmpl, data);
  ASSERT_FALSE(result.empty());

  // verify json
  std::string schema_path =
      std::string(JSONValidation::schema_base_directory) + "gen_StatementDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  ASSERT_TRUE(is_valid);
}


TEST(StatementTests, UndefineStmt){
  detail::Designator<detail::ExprID> des{"obj", "array", {"index"}};
  detail::UndefineStmt<decltype(des)> undefineStmt{des};
  json data = undefineStmt;

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("statement.tmpl");
  auto result = env.render(tmpl, data);

  ASSERT_STREQ("undefine obj[index];", result.c_str());

  // verify json
  std::string schema_path =
      std::string(JSONValidation::schema_base_directory) + "gen_StatementDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  ASSERT_TRUE(is_valid);
}


TEST(StatementTests, ProcCallStmt){
  json data = detail::ProcCall{"myfn", {detail::ExprID{"param1"}, detail::ExprID{"param2"}}};
  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("statement.tmpl");
  auto result = env.render(tmpl, data);

  ASSERT_STREQ("myfn(param1, param2);", result.c_str());

  // verify json
  std::string schema_path =
      std::string(JSONValidation::schema_base_directory) + "gen_StatementDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  ASSERT_TRUE(is_valid);
}


TEST(StatementTests, AliasStmt){
  auto aliasStmt = detail::AliasStmt<detail::ExprID>{"my_alias", {"to_be_aliased"}};
  auto assignmentStmt = detail::Assignment<detail::Designator<detail::ExprID>, detail::ExprID>{
    {"msg", "object", {"address"}}, {"val"}};
  aliasStmt.statements.emplace_back(assignmentStmt);


  json data = aliasStmt;
  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("statement.tmpl");
  auto result = env.render(tmpl, data);

//  ASSERT_STREQ("alias my_alias : to_be_aliased do end;", result.c_str());

  // verify json
  std::string schema_path =
      std::string(JSONValidation::schema_base_directory) + "gen_StatementDescription.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  ASSERT_TRUE(is_valid);
}