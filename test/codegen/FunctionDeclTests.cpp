#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "translation/murphi/codegen/InjaEnvSingleton.h"
#include "translation/murphi/codegen/MurphiCodeGen.h"
#include "translation/utils/JSONValidation.h"
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
      detail::ForwardDecl<detail::TypeDecl<detail::ID>>{
          "type",
          detail::TypeDecl<detail::ID>{"loc_type", detail::ID{"type_id"}}},
      detail::ForwardDecl<detail::VarDecl<detail::ID>>{"var",
                                                       {"msg", {"Message"}}}};

  data["statements"] = {detail::Assignment<detail::Designator<detail::ExprID>, detail::ExprID>{
      {"msg", "object", {"adr"}}, {"adr"}}};

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("function_decl.tmpl");
  auto result = env.render(tmpl, data);

  // verify json
  std::string schema_path =
      std::string(JSONValidation::schema_base_directory) + "gen_ProcDef.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  ASSERT_TRUE(is_valid);
  ASSERT_FALSE(result.empty());
}

TEST(FunctionTest, MessageConstructor) {

  // Setup an MLIR Context to create some dummy operations
  using namespace mlir;
  MLIRContext ctx;
  ctx.getOrLoadDialect<mlir::pcc::PCCDialect>();
  ctx.getOrLoadDialect<mlir::StandardOpsDialect>();
  OpBuilder builder(&ctx);
  ModuleOp theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(theModule.getBody());

  // create a basic Message Constructor Operation
  // say we have msg type Request{}
  // which implicitly has mtype: MsgIdType, IDType, IDType
  std::vector<Type> innerStructTypes{
      pcc::MsgIdType::get(&ctx, "none"), pcc::IDType::get(&ctx),
      pcc::IDType::get(&ctx), pcc::DataType::get(&ctx)};
  std::vector<NamedAttribute> fieldNames{
      std::make_pair(Identifier::get("mtype", &ctx),
                     TypeAttr::get(pcc::MsgIdType::get(&ctx, "none"))),
      std::make_pair(Identifier::get("src", &ctx),
                     TypeAttr::get(pcc::IDType::get(&ctx))),
      std::make_pair(Identifier::get("dst", &ctx),
                     TypeAttr::get(pcc::IDType::get(&ctx))),
      std::make_pair(Identifier::get("cl", &ctx),
                     TypeAttr::get(pcc::DataType::get(&ctx))),
  };

  pcc::StructType structType = pcc::StructType::get(innerStructTypes);
  pcc::MsgDeclOp msgDeclOp = builder.create<pcc::MsgDeclOp>(
      builder.getUnknownLoc(), "Response", structType, fieldNames);

  // now we can finally attempt to render the function generation
  murphi::detail::MessageFactory msgConstrFun{msgDeclOp};

  // convert it to json
  json j = msgConstrFun;

  // render the template
  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("function_decl.tmpl");
  auto result = env.render(tmpl, j);

  // verify against schema
  std::string schema_path =
      std::string(JSONValidation::schema_base_directory) + "gen_ProcDef.json";
  bool is_valid = JSONValidation::validate_json(schema_path, j);
  ASSERT_TRUE(is_valid);

  ASSERT_FALSE(result.empty());
}

TEST(FunctionTests, OrderedSendFunction){

  json j = detail::OrderedSendFunction{"fwd"};
  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("proc_decl.tmpl");
  auto result = env.render(tmpl, j);

  ASSERT_FALSE(result.empty());
}

TEST(FunctionTests, OrderedPopFunction){
  json j = detail::OrderedPopFunction{"fwd"};
  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("proc_decl.tmpl");
  auto result = env.render(tmpl, j);

  ASSERT_FALSE(result.empty());
}

TEST(FunctionTests, UnorderedSendFunction){
  json j = detail::UnorderedSendFunction{"resp"};

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("proc_decl.tmpl");
  auto result = env.render(tmpl, j);

  ASSERT_FALSE(result.empty());
}
