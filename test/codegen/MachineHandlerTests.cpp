#include "translation/murphi/codegen/InjaEnvSingleton.h"
#include "translation/murphi/codegen/MurphiCodeGen.h"
#include "translation/utils/JSONValidation.h"
#include <gtest/gtest.h>
#include <inja/inja.hpp>

using namespace inja;
using namespace murphi;
TEST(MachineHandler, BasicTest){
  json data = detail::MachineHandler{"cache"};

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("proc_decl.tmpl");
  auto result = env.render(tmpl, data);

  // verify json
  std::string schema_path =
      std::string(JSONValidation::schema_base_directory) + "gen_ProcDecl.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  ASSERT_TRUE(is_valid);

  ASSERT_STREQ("", result.c_str());
//  ASSERT_FALSE(result.empty());

}