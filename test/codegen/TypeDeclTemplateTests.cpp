#include "translation/murphi/codegen/InjaEnvSingleton.h"
#include <gtest/gtest.h>
#include <inja/inja.hpp>
#include <regex>

using namespace inja;

json getMessageType() {
  json data;
  data["decls"] = {{{"id", "adr"}, {"typeId", "ID"}, {"type", "Address"}},
                   {{"id", "mtype"}, {"typeId", "ID"}, {"type", "MessageType"}},
                   {{"id", "src"}, {"typeId", "ID"}, {"type", "Machines"}},
                   {{"id", "dst"}, {"typeId", "ID"}, {"type", "Machines"}},
                   {{"id", "cl"}, {"typeId", "ID"}, {"type", "ClValue"}}};
  return data;
}

TEST(TypeDeclTemplate, RecordType) {
  json data;

  data["type_decl_tmpl_data"].push_back(
      {
          {"id", "Message"},
          {"typeId", "record"},
          {"type", getMessageType()}
      });

  auto &env = InjaEnvSingleton::getInstance();

  auto tmpl = env.parse_template("type_decl.tmpl");
  auto result = env.render(tmpl, data);

  ASSERT_TRUE(result.find("Message : record") != std::string::npos);
  ASSERT_TRUE(result.find("adr : Address;") != std::string::npos);
  ASSERT_TRUE(result.find("mtype : MessageType") != std::string::npos);
  ASSERT_TRUE(result.find("src : Machines;") != std::string::npos);
  ASSERT_TRUE(result.find("dst : Machines;") != std::string::npos);
  ASSERT_TRUE(result.find("cl : ClValue;") != std::string::npos);


  // regex tests
  std::regex re(R"(type\s*\w+\s*:\s*record\s+(\w+\s*:\s*\w+;\s*)+end;\s*)");
  ASSERT_TRUE(std::regex_match(result, re));
}