#include "translation/murphi/codegen/MurphiCodeGen.h"
#include "translation/utils/JSONValidation.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include <gtest/gtest.h>

using namespace murphi;
using namespace JSONValidation;
TEST(MurphiCodeGen, ConstDecl_to_json) {
  detail::ConstDecl decl{"NrCaches", 3};
  json j = decl; // implicitly calls to_json function
  ASSERT_STREQ(std::string(j["id"]).c_str(), "NrCaches");
  ASSERT_EQ(j["value"], 3);
  std::string schema_path =
      std::string(schema_base_directory) + "gen_ConstDecl.json";

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
      std::string(schema_base_directory) + "gen_TypeDecl.json";

  ASSERT_TRUE(validate_json(schema_path, j));
}

TEST(MurphiCodeGen, MurphiType_schema_validation) {
  json valid_json = {{"decls", {"a", "b", "c"}}};
  json invalid_json = {{"decls", {"a", 21, "c"}}};

  std::string schema_path =
      std::string(schema_base_directory) + "gen_MurphiType.json";
  ASSERT_TRUE(validate_json(schema_path, valid_json));
  ASSERT_FALSE(validate_json(schema_path, invalid_json));
}

TEST(MurphiCodeGen, emitNetworkDefinitionJson) {
  json j = detail::emitNetworkDefinitionJson();
  // for each decl - validate that it is a valid type_decl
  std::string schema_path = std::string(schema_base_directory) + "gen_TypeDecl.json";
  for (auto &decl : j) {
    ASSERT_TRUE(validate_json(schema_path, decl));
  }
}


class ModuleOpFixture: public ::testing::Test {
public :
  ModuleOpFixture(): builder{&ctx} {
    ctx.getOrLoadDialect<mlir::pcc::PCCDialect>();
    ctx.getOrLoadDialect<mlir::StandardOpsDialect>();
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(theModule.getBody());
  }

  void add_test_const_decls(){
    builder.create<mlir::pcc::ConstantOp>(builder.getUnknownLoc(), "NrCaches", 4);
  }

  mlir::MLIRContext ctx;
  mlir::OpBuilder builder;
  mlir::ModuleOp theModule;
};

TEST_F(ModuleOpFixture, validate_wo_const_decls){
  MurphiCodeGen codeGen(theModule, llvm::outs());
  ASSERT_TRUE(codeGen.is_json_valid());
  codeGen.generateConstants();
  ASSERT_TRUE(codeGen.is_json_valid());
}

TEST_F(ModuleOpFixture, validate_w_const_decls){
  add_test_const_decls();
  MurphiCodeGen codeGen(theModule, llvm::outs());
  ASSERT_TRUE(codeGen.is_json_valid());
  codeGen.generateConstants();
  ASSERT_TRUE(codeGen.is_json_valid());
}