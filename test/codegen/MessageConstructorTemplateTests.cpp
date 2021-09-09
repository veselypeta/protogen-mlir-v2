#include "translation/murphi/codegen/InjaEnvSingleton.h"
#include <gtest/gtest.h>
#include <inja/inja.hpp>


nlohmann::json getIdAndType(const std::string &id, const std::string &typeId,
                            const std::string &type) {
  using namespace nlohmann;
  json data;
  data["id"] = id;
  data["typeId"] = typeId;
  data["type"] = type;
  return data;
}

TEST(CodeGenTests, InjaEnvSingelton) {
  using namespace inja;
  Environment &env = InjaEnvSingleton::getInstance();
  json data;
  auto tmpl = env.parse_template("hello_world.tmpl");
  auto result = env.render(tmpl, data);
  ASSERT_STREQ(result.c_str(), "hello world!");
}

TEST(CodeGenTests, ComplexMessageConstructor) {
  using namespace inja;
  json data;

  // setup some test data
  data["global_message_type"]["fields"].push_back(
      {getIdAndType("cl", "ID", "CL")});

  data["message_constructor_tmpl_data"]["msgId"] = "Resp";
  data["message_constructor_tmpl_data"]["additionalParameters"].push_back(
      getIdAndType("cl", "ID", "CL"));

  // get the environment
  auto env = InjaEnvSingleton::getInstance();

  // parse the template
  Template tmp = env.parse_template("message_constructor.tmpl");

  // render the template
  auto result = env.render(tmp, data);

  // renders the line that it has the additional parameter
  ASSERT_TRUE(result.find("msg.cl := cl;") != std::string::npos);
  ASSERT_TRUE(result.find("msg.cl := undefined;") == std::string::npos);

  // the additional parameter is also rendered
  ASSERT_TRUE(result.find("cl: CL) : Message;") != std::string::npos);
}

TEST(CodeGenTests, SimpleMessageConstructor){
  using namespace inja;
  json data;

  // setup data
  data["global_message_type"]["fields"].push_back(
      {getIdAndType("cl", "ID", "CL")});

  data["message_constructor_tmpl_data"]["msgId"] = "Ack";
  data["message_constructor_tmpl_data"]["additionalParameters"] = json::array();


  auto env = InjaEnvSingleton::getInstance();
  auto tmpl = env.parse_template("message_constructor.tmpl");
  auto result = env.render(tmpl, data);

  ASSERT_TRUE(result.find("dst: Machines) : Message;") != std::string::npos);
  ASSERT_TRUE(result.find("msg.cl := undefined;") != std::string::npos);
}
