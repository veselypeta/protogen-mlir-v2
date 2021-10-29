#include "translation/murphi/codegen/InjaEnvSingleton.h"
#include "translation/murphi/codegen/MurphiCodeGen.h"
#include <gtest/gtest.h>
#include <inja/inja.hpp>

using namespace inja;
using namespace murphi;
TEST(FunctionTest, RenderTemplate) {
  json data;
  data["id"] = "my_function";
  data["params"] = {
      detail::Formal<detail::ID>{"first_param", {{"first_param_type"}}},
      detail::Formal<detail::ID>{"second_param", {{"second_param_type"}}},
      detail::Formal<detail::ID>{"third_param", {{"third_param_type"}}},
  };
  data["returnType"] = detail::ID{"ret_type_id"};
  data["forwardDecls"] = {
      detail::ForwardDecl<detail::ConstDecl>{"const", {"fwd_const_decl", 55}},
      detail::ForwardDecl<detail::TypeDecl<detail::ID>>{"type", detail::TypeDecl<detail::ID>{"loc_type", detail::ID{"type_id"}}},
      detail::ForwardDecl<detail::VarDecl<detail::ID>>{"var", {"msg", {"Message"}}}
  };

  data["statements"] = {
      detail::Assignment<detail::ExprID, detail::ExprID>{
          {"msg", "object", {"adr"}},
          {"adr"}
      }
  };

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("function_decl.tmpl");
  auto result = env.render(tmpl, data);

  ASSERT_FALSE(result.empty());
}
