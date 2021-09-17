#include "translation/murphi/codegen/MurphiCodeGen.h"
#include "translation/utils/JSONValidation.h"
#include <gtest/gtest.h>

using namespace murphi;
using namespace JSONValidation;
TEST(MurphiCodeGen, ConstDecl_to_json) {
  detail::ConstDecl decl{"NrCaches", 3};
  json j = decl; // implicitly calls to_json function
  ASSERT_STREQ(std::string(j["id"]).c_str(), "NrCaches");
  ASSERT_EQ(j["value"], 3);
  std::string schema_path =
      std::string(schema_base_directory) + "const_decl.json";

  ASSERT_TRUE(validate_json(schema_path, j));
}

TEST(MurphiCodeGen, EnumDecl_to_json) {
  detail::Enum testEnum{"test-enum", {"a", "b", "c"}};
  json j = testEnum;
  ASSERT_STREQ(std::string(j["id"]).c_str(), "test-enum");
  ASSERT_STREQ(std::string(j["typeId"]).c_str(), "enum");
  auto elems = j["type"]["decls"].get<std::vector<std::string>>();
  ASSERT_EQ(elems.size(), 3);
  ASSERT_STREQ(elems.at(0).c_str(), "a");
  ASSERT_STREQ(elems.at(1).c_str(), "b");
  ASSERT_STREQ(elems.at(2).c_str(), "c");

  std::string schema_path =
      std::string(schema_base_directory) + "type_decl.json";

  ASSERT_TRUE(validate_json(schema_path, j));
}

TEST(MurphiCodeGen, MurphiType){
  json valid_json = {
    {"decls", {"a", "b", "c"}}
  };
  json invalid_json = {
      {"decls", {"a", 21, "c"}}
  };

  std::string schema_path = std::string(schema_base_directory) + "murphi_type.json";
  ASSERT_TRUE(validate_json(schema_path, valid_json));
  ASSERT_FALSE(validate_json(schema_path, invalid_json));
}