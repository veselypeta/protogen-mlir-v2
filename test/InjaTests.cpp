#include <gtest/gtest.h>
#include <inja/inja.hpp>
#include <nlohmann/json.hpp>


TEST(InjaSuite, BasicExample){
  using namespace inja;
  json data;
  data["name"] = "world";
  auto result = inja::render("Hello {{ name }}!", data);

  ASSERT_STREQ("Hello world!", result.c_str());
}

TEST(InjaSuite, ComplexJsonTest){
  using namespace inja;
  json data;

  data["name"]["forename"] = "bob";
  data["name"]["surname"] = "smith";

  const std::string templ = "My name is {{ name.forename }} {{ name.surname }}!";

  const auto result = inja::render(templ, data);

  ASSERT_STREQ("My name is bob smith!", result.c_str());
}

TEST(InjaSuite, InjaLoopsTest){
  using namespace inja;
  json data;
  data["constdecls"] = {
      {{"id", "NrCaches"}, { "value", 3 }},
      {{"id", "ValCount"}, { "value", 5 }}
  };

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

TEST(JSONSuite, UnorderedJsonTest){
  using namespace inja;
  json unordJson;
  unordJson["elements"] = json::array();
  unordJson["elements"].push_back({{"C", 1}});
  unordJson["elements"].push_back({{"B", 2}});
  unordJson["elements"].push_back({{"A", 3}});

  auto result = to_string(unordJson);

  ASSERT_STREQ("{\"elements\":[{\"C\":1},{\"B\":2},{\"A\":3}]}", result.c_str());
}


namespace ns{
using namespace inja;

struct MurphiTestConstant{
  std::string id;
  size_t value;
};

void to_json(json &j, const MurphiTestConstant &m) {
  j = json{{"id", m.id}, {"value", m.value}};
}

void from_json(const json &j, MurphiTestConstant &m){
  j.at("id").get_to(m.id);
  j.at("value").get_to(m.value);
}
}

TEST(JSONSuite, StructPushBack){
  using namespace inja;
  using namespace ns;

  MurphiTestConstant tc{"NrCaches", 3};

  json data = tc;

  auto result = to_string(data);

  ASSERT_STREQ("{\"id\":\"NrCaches\",\"value\":3}", result.c_str());
  
}

TEST(InjaSuite, TemplateInheritance){
  using namespace inja;
  Environment env{"../../templates/"};
  env.set_trim_blocks(true);
  env.set_lstrip_blocks(true);

  Template tmp = env.parse_template("murphi_base.tmpl");
  json data;
  data["const_decls"].push_back({{"id", "NrCaches"}, {"value", 3}});
  data["const_decls"].push_back({{"id", "ValCount"}, {"value", 6}});
  auto result = env.render(tmp, data);

  // Template returns something
  ASSERT_STRNE("", result.c_str());
}


TEST(InjaSuite, IncludeOnTypeId){
  using namespace inja;
  Environment env{"../../templates/"};
  env.set_trim_blocks(true);
  env.set_lstrip_blocks(true);
  Template tmp = env.parse_template("murphi_base.tmpl");

  json data;
  data["const_decls"] = json::array();
  data["type_decls"].push_back({
      { "typeid", "record"},
      { "id", "Message" },
      { "decls", {
                      {"src", "Cache"},
                      {"dst", "Cache"},
                      {"access", "Access"}
                }
      }
  }
 );

  auto result = env.render(tmp, data);

  ASSERT_STREQ("", result.c_str());



}