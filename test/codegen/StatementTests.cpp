#include "translation/murphi/codegen/InjaEnvSingleton.h"
#include "translation/murphi/codegen/MurphiCodeGen.h"
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
}


TEST(StatementTest, UndefineStmt){
  detail::Designator<detail::ExprID> des{"obj", "array", {"index"}};
  detail::UndefineStmt<decltype(des)> undefineStmt{des};
  json data = undefineStmt;

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("statement.tmpl");
  auto result = env.render(tmpl, data);

  ASSERT_STREQ("undefine obj[index];", result.c_str());
}