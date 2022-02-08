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

  data["statements"] = {detail::Assignment<detail::Designator, detail::ExprID>{
      {"msg",{detail::Indexer{"object", detail::ExprID{"adr"}}}}, {"adr"}}};

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

  json data = detail::OrderedSendFunction{"fwd"};
  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("proc_decl.tmpl");
  auto result = env.render(tmpl, data);

  ASSERT_FALSE(result.empty());

  // verify json
  std::string schema_path =
      std::string(JSONValidation::schema_base_directory) + "gen_ProcDecl.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  ASSERT_TRUE(is_valid);
}

TEST(FunctionTests, OrderedPopFunction){
  json data = detail::OrderedPopFunction{"fwd"};
  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("proc_decl.tmpl");
  auto result = env.render(tmpl, data);

  ASSERT_FALSE(result.empty());

  // verify json
  std::string schema_path =
      std::string(JSONValidation::schema_base_directory) + "gen_ProcDecl.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  ASSERT_TRUE(is_valid);
}

TEST(FunctionTests, UnorderedSendFunction){
  json data = detail::UnorderedSendFunction{"resp"};

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("proc_decl.tmpl");
  auto result = env.render(tmpl, data);

  ASSERT_FALSE(result.empty());

  // verify json
  std::string schema_path =
      std::string(JSONValidation::schema_base_directory) + "gen_ProcDecl.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  ASSERT_TRUE(is_valid);
}

TEST(FunctionTests, GenericMurphiFunction){
  auto msgFactoryFunction = detail::GenericMurphiFunction{
    "Response",
      {detail::Formal<detail::ID>{detail::c_adr, {detail::ss_address_t}}},
      detail::ID{detail::r_message_t},
      {detail::ForwardDecl<detail::TypeDecl<detail::ID>>{"var", {detail::c_msg, {detail::r_message_t}}}},
      {}
  };

  json data = msgFactoryFunction;

  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("proc_decl.tmpl");
  auto result = env.render(tmpl, data);

  ASSERT_FALSE(result.empty());

  // verify json
  std::string schema_path =
      std::string(JSONValidation::schema_base_directory) + "gen_ProcDecl.json";
  bool is_valid = JSONValidation::validate_json(schema_path, data);
  ASSERT_TRUE(is_valid);

}

TEST(FunctionTests, SetOpAdd){
  auto setOpAdd = detail::SetAdd{{"Machines", 3}};


  json data = setOpAdd;
  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("proc_decl.tmpl");
  auto result = env.render(tmpl, data);
  auto expectedText =
      "procedure add_v_3_Machines(var sv : v_3_Machines; n : Machines);\n"
      "begin\n"
      "    if MultisetCount(i:sv, sv[i] = n) = 0 then\n"
      "        MultisetAdd(n, sv);\n"
      "    endif;\n"
      "end;\n";

  ASSERT_STREQ(result.c_str(), expectedText);
}

TEST(FunctionTests, SetOpCount){
  auto setOpCount = detail::SetCount{{"OBJSET_cache", 5}};

  json data = setOpCount;
  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("proc_decl.tmpl");
  auto result = env.render(tmpl, data);

  auto expectedText =
      "function count_v_5_OBJSET_cache ( var sv : v_5_OBJSET_cache ) : cnt_v_5_OBJSET_cache;\n"
      "begin\n"
      "    return MultisetCount(i:sv, IsMember(sv[i], OBJSET_cache));\n"
      "end;\n";

  ASSERT_STREQ(result.c_str(), expectedText);
}

TEST(FunctionTests, SetOpContains){
  auto setOpContains = detail::SetContains{{"OBJSET_cache", 3}};
  json data = setOpContains;
  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("proc_decl.tmpl");
  auto result = env.render(tmpl, data);

  auto expectedText =
      "function contains_v_3_OBJSET_cache ( var sv : v_3_OBJSET_cache; n : OBJSET_cache ) : boolean;\n"
      "begin\n"
      "    return MultisetCount(i:sv, sv[i] = n) = 1;\n"
      "end;\n";

  ASSERT_STREQ(result.c_str(), expectedText);

}

TEST(FunctionTests, SetOpDelete){
  auto setOpDelete = detail::SetDelete{{"OBJSET_cache", 3}};
  json data = setOpDelete;
  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("proc_decl.tmpl");
  auto result = env.render(tmpl, data);

  auto expectedText =
      "procedure delete_v_3_OBJSET_cache(var sv : v_3_OBJSET_cache; n : OBJSET_cache);\n"
      "begin\n"
      "    if MultisetCount(i:sv, sv[i] = n) = 1 then\n"
      "        MultisetRemovePred(i:sv, sv[i] = n);\n"
      "    endif;\n"
      "end;\n";

  ASSERT_STREQ(result.c_str(), expectedText);
}

TEST(FunctionTests, SetOpClear){
  auto setOpClear = detail::SetClear{{"OBJSET_cache", 3}};
  json data = setOpClear;
  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("proc_decl.tmpl");
  auto result = env.render(tmpl, data);

  auto expectedText =
      "procedure clear_v_3_OBJSET_cache(var sv : v_3_OBJSET_cache);\n"
      "begin\n"
      "    MultisetRemovePred(i:sv, true);\n"
      "end;\n";

  ASSERT_STREQ(result.c_str(), expectedText);
}

TEST(FunctionTests, MulticastSend){
  auto ms = detail::MulticastSend{
    "fwd",
        detail::Set{"Machines", 3}
  };
  json data = ms;
  auto &env = InjaEnvSingleton::getInstance();
  const auto tmpl = env.parse_template("proc_decl.tmpl");
  auto result = env.render(tmpl, data);

  auto expectedText =
      "procedure Multicast_fwd_v_3_Machines(var msg : Message; dst : v_3_Machines);\n"
      "begin\n"
      "    for iSV : Machines do\n"
      "        if iSV != msg.src then\n"
      "            if MultisetCount(i:dst, dst[i] = iSV) = 1 then\n"
      "                msg.dst := iSV;\n"
      "                Send_fwd(msg);\n"
      "            endif;\n"
      "        endif;\n"
      "    endfor;\n"
      "end;\n";

  ASSERT_STREQ(result.c_str(), expectedText);
}
