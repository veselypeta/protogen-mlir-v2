#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "translation/murphi/codegen/MurphiCodeGen.h"
#include "translation/utils/JSONValidation.h"
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

TEST(MurphiCodeGen, Union_to_json) {
  detail::Union testUnion{"my_union", {"aaa", "bbb", "ccc"}};
  json j = testUnion;

  ASSERT_STREQ(std::string(j["id"]).c_str(), "my_union");
  ASSERT_STREQ(std::string(j["typeId"]).c_str(), "union");
  auto elems = j["type"]["listElems"].get<std::vector<std::string>>();
  ASSERT_EQ(elems.size(), 3);
  ASSERT_STREQ(elems.at(0).c_str(), "aaa");
  ASSERT_STREQ(elems.at(1).c_str(), "bbb");
  ASSERT_STREQ(elems.at(2).c_str(), "ccc");

  // json validation
  std::string schema_path =
      std::string(schema_base_directory) + "gen_TypeDecl.json";

  ASSERT_TRUE(validate_json(schema_path, j));
}

TEST(MurphiCodeGen, Union_to_json_too_short) {
  detail::Union testUnion{"my_union", {"aaa"}};
  json j = testUnion;
  // json validation
  std::string schema_path =
      std::string(schema_base_directory) + "gen_TypeDecl.json";

  ASSERT_FALSE(validate_json(schema_path, j));
}

TEST(MurphiCodeGen, MurphiType_schema_validation) {
  json valid_json = {{"decls", {"a", "b", "c"}}};
  json invalid_json = {{"decls", {"a", 21, "c"}}};

  std::string schema_path =
      std::string(schema_base_directory) + "gen_MurphiType.json";
  ASSERT_TRUE(validate_json(schema_path, valid_json));
  ASSERT_FALSE(validate_json(schema_path, invalid_json));
}

TEST(MurphiCodeGen, MurphiRecordType_to_json) {
  detail::Record record = {
      "my_record", {{"id", "ID"}, {"hello", "world"}, {"field", "type"}}};
  json j = record;
  std::string schema_path =
      std::string(schema_base_directory) + "gen_TypeDecl.json";
  ASSERT_TRUE(validate_json(schema_path, j));
}

TEST(MurphiCodeGen, MurphiScalarSetType_to_json_string){
  detail::ScalarSet<std::string> string_ss = {"ss_string", "ss_value"};
  json j = string_ss;
  ASSERT_STREQ(std::string(j["id"]).c_str(), "ss_string" );
  ASSERT_STREQ(std::string(j["typeId"]).c_str(), "scalarset" );
  ASSERT_STREQ(std::string(j["type"]["type"]).c_str(), "ss_value" );

  std::string schema_path =
      std::string(schema_base_directory) + "gen_TypeDecl.json";
  ASSERT_TRUE(validate_json(schema_path, j));
}

TEST(MurphiCodeGen, MurphiScalarSetType_to_json_integer){
  detail::ScalarSet<size_t> integer_ss = {"ss_integer", 4};
  json j = integer_ss;
  ASSERT_STREQ(std::string(j["id"]).c_str(), "ss_integer" );
  ASSERT_STREQ(std::string(j["typeId"]).c_str(), "scalarset" );
  ASSERT_EQ(j["type"]["type"], 4 );

  std::string schema_path =
      std::string(schema_base_directory) + "gen_TypeDecl.json";
  ASSERT_TRUE(validate_json(schema_path, j));
}

TEST(MurphiCodeGen, emitNetworkDefinitionJson) {
  json j = detail::emitNetworkDefinitionJson();
  // for each decl - validate that it is a valid type_decl
  std::string schema_path =
      std::string(schema_base_directory) + "gen_TypeDecl.json";
  for (auto &decl : j) {
    ASSERT_TRUE(validate_json(schema_path, decl));
  }
}

class ModuleOpFixture : public ::testing::Test {
public:
  ModuleOpFixture() : builder{&ctx} {
    ctx.getOrLoadDialect<mlir::pcc::PCCDialect>();
    ctx.getOrLoadDialect<mlir::StandardOpsDialect>();
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(theModule.getBody());
  }

  void add_test_const_decls() {
    builder.create<mlir::pcc::ConstantOp>(builder.getUnknownLoc(), "NrCaches",
                                          4);
  }

  void add_mach_decl_ops() {
    // create a basic cache decl op
    mlir::pcc::IDType idType = mlir::pcc::IDType::get(&ctx);
    std::vector<mlir::Type> elemTypes = {idType};
    mlir::pcc::StructType structType = mlir::pcc::StructType::get(elemTypes);
    mlir::pcc::SetType setType = mlir::pcc::SetType::get(structType, 4);
    // generate the named attributes
    std::vector<mlir::NamedAttribute> namedAttrs;
    namedAttrs.emplace_back(builder.getIdentifier("owner"),
                            mlir::TypeAttr::get(idType));

    builder.create<mlir::pcc::CacheDeclOp>(
        builder.getUnknownLoc(), detail::machines.cache, setType, namedAttrs);

    // create a basic dir decl op

    mlir::pcc::IDType dirOwnerField = mlir::pcc::IDType::get(&ctx);
    std::vector<mlir::Type> dirElemTypes = {dirOwnerField};
    mlir::pcc::StructType dirStructType =
        mlir::pcc::StructType::get(dirElemTypes);
    std::vector<mlir::NamedAttribute> dirNamedAttrs = {
        builder.getNamedAttr("owner", mlir::TypeAttr::get(dirOwnerField))};
    builder.create<mlir::pcc::DirectoryDeclOp>(builder.getUnknownLoc(),
                                               detail::machines.directory,
                                               dirStructType, dirNamedAttrs);
  }

  mlir::MLIRContext ctx;
  mlir::OpBuilder builder;
  mlir::ModuleOp theModule;
};

TEST_F(ModuleOpFixture, validate_wo_const_decls) {
  MurphiCodeGen codeGen(theModule, llvm::outs());
  ASSERT_TRUE(codeGen.is_json_valid());
  codeGen.generateConstants();
  ASSERT_TRUE(codeGen.is_json_valid());
}

TEST_F(ModuleOpFixture, validate_w_const_decls) {
  add_test_const_decls();
  MurphiCodeGen codeGen(theModule, llvm::outs());
  ASSERT_TRUE(codeGen.is_json_valid());
  codeGen.generateConstants();
  ASSERT_TRUE(codeGen.is_json_valid());
}

TEST_F(ModuleOpFixture, validate_add_types) {
  add_test_const_decls();
  add_mach_decl_ops();
  MurphiCodeGen codeGen(theModule, llvm::outs());
  ASSERT_TRUE(codeGen.is_json_valid());
  codeGen.generateConstants();
  codeGen.generateTypes();
  ASSERT_TRUE(codeGen.is_json_valid());
}