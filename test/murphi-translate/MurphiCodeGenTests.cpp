#include <gtest/gtest.h>
#include "translation/murphi/codegen/MurphiCodeGen.h"

using namespace murphi;
TEST(MurphiCodeGen, ConstDecl_to_json){
  detail::ConstDecl decl{"NrCaches", 3};
  json j = decl; // implicitly calls to_json function
  ASSERT_STREQ(std::string(j["id"]).c_str(), "NrCaches");
  ASSERT_EQ(j["value"], 3);
  // TODO - json validate would be really usefull here
}

TEST(MurphiCodeGen, EnumDecl_to_json){
  detail::Enum testEnum{"test-enum", {"a", "b", "c"}};
  json j = testEnum;
  ASSERT_STREQ(std::string(j["id"]).c_str(), "test-enum");
  ASSERT_STREQ(std::string(j["typeId"]).c_str(), "enum");
  auto elems = j["type"]["decls"].get<std::vector<std::string>>();
  ASSERT_EQ(elems.size(), 3);
  ASSERT_STREQ(elems.at(0).c_str(), "a");
  ASSERT_STREQ(elems.at(1).c_str(), "b");
  ASSERT_STREQ(elems.at(2).c_str(), "c");
}