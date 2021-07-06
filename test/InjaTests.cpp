#include <gtest/gtest.h>
#include <inja/inja.hpp>



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