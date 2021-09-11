#include "translation/murphi/codegen/InjaEnvSingleton.h"
#include <gtest/gtest.h>
#include <inja/inja.hpp>
#include <nlohmann/json.hpp>
#include <regex>

TEST(InjaSuite, BasicExample) {
  using namespace inja;
  json data;
  data["name"] = "world";
  auto result = inja::render("Hello {{ name }}!", data);

  ASSERT_STREQ("Hello world!", result.c_str());
}

TEST(InjaSuite, ComplexJsonTest) {
  using namespace inja;
  json data;

  data["name"]["forename"] = "bob";
  data["name"]["surname"] = "smith";

  const std::string templ =
      "My name is {{ name.forename }} {{ name.surname }}!";

  const auto result = inja::render(templ, data);

  ASSERT_STREQ("My name is bob smith!", result.c_str());
}

TEST(InjaSuite, InjaLoopsTest) {
  using namespace inja;
  json data;
  data["constdecls"] = {{{"id", "NrCaches"}, {"value", 3}},
                        {{"id", "ValCount"}, {"value", 5}}};

  const std::string templ = "const\n"
                            "{% for cd in constdecls %}\n"
                            "\t{{ cd.id }} : {{ cd.value }}\n"
                            "{% endfor %}\n";

  const auto result = inja::render(templ, data);

  const std::string expected_str = "const\n"
                                   "\n"
                                   "\tNrCaches : 3\n"
                                   "\n"
                                   "\tValCount : 5\n"
                                   "\n";

  ASSERT_STREQ(expected_str.c_str(), result.c_str());
}

TEST(JSONSuite, UnorderedJsonTest) {
  using namespace inja;
  json unordJson;
  unordJson["elements"] = json::array();
  unordJson["elements"].push_back({{"C", 1}});
  unordJson["elements"].push_back({{"B", 2}});
  unordJson["elements"].push_back({{"A", 3}});

  auto result = to_string(unordJson);

  ASSERT_STREQ("{\"elements\":[{\"C\":1},{\"B\":2},{\"A\":3}]}",
               result.c_str());
}

namespace {
using namespace inja;

struct MurphiTestConstant {
  std::string id;
  size_t value;
};

void to_json(json &j, const MurphiTestConstant &m) {
  j = json{{"id", m.id}, {"value", m.value}};
}

void from_json(const json &j, MurphiTestConstant &m) {
  j.at("id").get_to(m.id);
  j.at("value").get_to(m.value);
}

json getMessageType() {
  json data;
  data["decls"] = {{{"id", "adr"}, {"typeId", "ID"}, {"type", "Address"}},
                   {{"id", "mtype"}, {"typeId", "ID"}, {"type", "MessageType"}},
                   {{"id", "src"}, {"typeId", "ID"}, {"type", "Machines"}},
                   {{"id", "dst"}, {"typeId", "ID"}, {"type", "Machines"}},
                   {{"id", "cl"}, {"typeId", "ID"}, {"type", "ClValue"}}};
  return data;
}
} // end namespace

TEST(JSONSuite, StructPushBack) {
  using namespace inja;

  MurphiTestConstant tc{"NrCaches", 3};

  json data = tc;

  auto result = to_string(data);

  ASSERT_STREQ("{\"id\":\"NrCaches\",\"value\":3}", result.c_str());
}

TEST(InjaSuite, TemplateInheritance) {
  using namespace inja;
  auto &env = InjaEnvSingleton::getInstance();

  Template tmp = env.parse_template("murphi_base.tmpl");
  json data;
  data["const_decls"].push_back({{"id", "NrCaches"}, {"value", 3}});
  data["const_decls"].push_back({{"id", "ValCount"}, {"value", 6}});
  auto result = env.render(tmp, data);

  ASSERT_NE(result.find("NrCaches : 3;"), std::string::npos);
  ASSERT_NE(result.find("ValCount : 6;"), std::string::npos);
}

TEST(InjaSuite, IncludeOnTypeId) {
  using namespace inja;
  auto &env = InjaEnvSingleton::getInstance();
  Template tmp = env.parse_template("murphi_base.tmpl");

  json data;
  data["type_decls"].push_back(
      {{"typeId", "record"}, {"id", "Message"}, {"type", getMessageType()}});
  std::string result = env.render(tmp, data);

  ASSERT_NE(result.find("adr : Address;"), std::string::npos);
  ASSERT_NE(result.find("mtype : MessageType;"), std::string::npos);
  ASSERT_NE(result.find("src : Machines;"), std::string::npos);
  ASSERT_NE(result.find("dst : Machines;"), std::string::npos);
  ASSERT_NE(result.find("cl : ClValue;"), std::string::npos);

  std::regex re(R"(\w+ : record(.|\s)*end;)");
  std::smatch sm;
  std::regex_search(result, sm, re);
  ASSERT_TRUE(sm.length() > 0);
}

TEST(InjaSuite, TestPushingJsonObject) {
  using namespace inja;

  json data;
  MurphiTestConstant tst_const{"NrCaches", 3};
  data["const_decls"].push_back(tst_const);

  const auto jsonStr = to_string(data);

  ASSERT_STREQ("{\"const_decls\":[{\"id\":\"NrCaches\",\"value\":3}]}",
               jsonStr.c_str());
}