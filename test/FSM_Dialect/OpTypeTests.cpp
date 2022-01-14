#include "FSM/FSMOps.h"
#include "OpHelper.h"
#include "mlir/Parser.h"
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::fsm;

TEST(OpTypes, IDTypeConstr) {
  OpHelper helper;
  Location unknLoc = helper.builder.getUnknownLoc();

  IDType idType = IDType::get(&helper.ctx);

  // make a machine op
  FunctionType fType = helper.builder.getFunctionType(llvm::None, llvm::None);
  MachineOp theCache =
      helper.builder.create<MachineOp>(unknLoc, "cache", fType);
  Block *cacheEntry = theCache.addEntryBlock();
  helper.builder.setInsertionPointToStart(cacheEntry);

  // make a variable op which returns this type
  VariableOp varOp =
      helper.builder.create<VariableOp>(unknLoc, idType, nullptr, "idTypeVar");

  std::string str;
  llvm::raw_string_ostream stream(str);
  varOp.print(stream);

  auto expctOut = "%idTypeVar = fsm.variable \"idTypeVar\" : !fsm.id";

  EXPECT_STREQ(expctOut, str.c_str());
}

TEST(OpTypes, IDTypeParse) {
  OpHelper helper;
  llvm::StringRef OpText =
      "fsm.machine @cache (){\n"
      "  %idTypeVar = fsm.variable \"idTypeVar\" : !fsm.id\n"
      "}\n";

  auto result = parseSourceString(OpText, &helper.ctx);

  result->walk([](VariableOp vOp){
    EXPECT_TRUE(vOp.getType().isa<IDType>());
    EXPECT_EQ(vOp.name(), "idTypeVar");
    EXPECT_FALSE(vOp.initValue().hasValue());
  });

}


TEST(OpTypes, DataTypeConstr){
  OpHelper helper;
  Location unknLoc = helper.builder.getUnknownLoc();

  DataType dataType = DataType::get(&helper.ctx);

  // make a machine op
  FunctionType fType = helper.builder.getFunctionType(llvm::None, llvm::None);
  MachineOp theCache =
      helper.builder.create<MachineOp>(unknLoc, "cache", fType);
  Block *cacheEntry = theCache.addEntryBlock();
  helper.builder.setInsertionPointToStart(cacheEntry);

  // make a variable op which returns this type
  VariableOp varOp =
      helper.builder.create<VariableOp>(unknLoc, dataType, nullptr, "idTypeVar");

  std::string str;
  llvm::raw_string_ostream stream(str);
  varOp.print(stream);

  auto expctOut = "%idTypeVar = fsm.variable \"idTypeVar\" : !fsm.data";

  EXPECT_STREQ(expctOut, str.c_str());
}

TEST(OpTypes, DataTypeParse) {
  OpHelper helper;
  llvm::StringRef OpText =
      "fsm.machine @cache (){\n"
      "  %idTypeVar = fsm.variable \"idTypeVar\" : !fsm.data\n"
      "}\n";

  auto result = parseSourceString(OpText, &helper.ctx);

  result->walk([](VariableOp vOp){
    EXPECT_TRUE(vOp.getType().isa<DataType>());
    EXPECT_EQ(vOp.name(), "idTypeVar");
    EXPECT_FALSE(vOp.initValue().hasValue());
  });
}

TEST(OpTypes, MsgTypeConstr){
  OpHelper helper;
  Location unknLoc = helper.builder.getUnknownLoc();

  MsgType msgType = MsgType::get(&helper.ctx);

  // make a machine op
  FunctionType fType = helper.builder.getFunctionType(llvm::None, llvm::None);
  MachineOp theCache =
      helper.builder.create<MachineOp>(unknLoc, "cache", fType);
  Block *cacheEntry = theCache.addEntryBlock();
  helper.builder.setInsertionPointToStart(cacheEntry);

  // make a variable op which returns this type
  VariableOp varOp =
      helper.builder.create<VariableOp>(unknLoc, msgType, nullptr, "idTypeVar");

  std::string str;
  llvm::raw_string_ostream stream(str);
  varOp.print(stream);

  auto expctOut = "%idTypeVar = fsm.variable \"idTypeVar\" : !fsm.msg";

  EXPECT_STREQ(expctOut, str.c_str());
}

TEST(OpTypes, MsgTypeParse) {
  OpHelper helper;
  llvm::StringRef OpText =
      "fsm.machine @cache (){\n"
      "  %idTypeVar = fsm.variable \"idTypeVar\" : !fsm.msg\n"
      "}\n";

  auto result = parseSourceString(OpText, &helper.ctx);

  result->walk([](VariableOp vOp){
    EXPECT_TRUE(vOp.getType().isa<MsgType>());
    EXPECT_EQ(vOp.name(), "idTypeVar");
    EXPECT_FALSE(vOp.initValue().hasValue());
  });
}