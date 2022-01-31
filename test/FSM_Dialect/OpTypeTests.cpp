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

  result->walk([](VariableOp vOp) {
    EXPECT_TRUE(vOp.getType().isa<IDType>());
    EXPECT_EQ(vOp.name(), "idTypeVar");
    EXPECT_FALSE(vOp.initValue().hasValue());
  });
}

TEST(OpTypes, DataTypeConstr) {
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
  VariableOp varOp = helper.builder.create<VariableOp>(unknLoc, dataType,
                                                       nullptr, "idTypeVar");

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

  result->walk([](VariableOp vOp) {
    EXPECT_TRUE(vOp.getType().isa<DataType>());
    EXPECT_EQ(vOp.name(), "idTypeVar");
    EXPECT_FALSE(vOp.initValue().hasValue());
  });
}

TEST(OpTypes, MsgTypeConstr) {
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

  result->walk([](VariableOp vOp) {
    EXPECT_TRUE(vOp.getType().isa<MsgType>());
    EXPECT_EQ(vOp.name(), "idTypeVar");
    EXPECT_FALSE(vOp.initValue().hasValue());
  });
}

TEST(OpTypes, StateTypeConstr) {
  OpHelper helper;
  Location unknLoc = helper.builder.getUnknownLoc();

  StateType stateType = StateType::get(&helper.ctx);

  // make a machine op
  FunctionType fType = helper.builder.getFunctionType(llvm::None, llvm::None);
  MachineOp theCache =
      helper.builder.create<MachineOp>(unknLoc, "cache", fType);
  Block *cacheEntry = theCache.addEntryBlock();
  helper.builder.setInsertionPointToStart(cacheEntry);

  // make a variable op which returns this type
  VariableOp varOp = helper.builder.create<VariableOp>(unknLoc, stateType,
                                                       nullptr, "idTypeVar");

  std::string str;
  llvm::raw_string_ostream stream(str);
  varOp.print(stream);

  auto expctOut = "%idTypeVar = fsm.variable \"idTypeVar\" : !fsm.state";

  EXPECT_STREQ(expctOut, str.c_str());
}

TEST(OpTypes, StateTypeParse) {
  OpHelper helper;
  llvm::StringRef OpText =
      "fsm.machine @cache (){\n"
      "  %idTypeVar = fsm.variable \"idTypeVar\" : !fsm.state\n"
      "}\n";

  auto result = parseSourceString(OpText, &helper.ctx);

  result->walk([](VariableOp vOp) {
    EXPECT_TRUE(vOp.getType().isa<StateType>());
    EXPECT_EQ(vOp.name(), "idTypeVar");
    EXPECT_FALSE(vOp.initValue().hasValue());
  });
}

TEST(OpTypes, RangeType){
  OpHelper helper;
  Location unknLoc = helper.builder.getUnknownLoc();

  RangeType rangeType = RangeType::get(&helper.ctx, 0, 3);

  // make a machine op
  FunctionType fType = helper.builder.getFunctionType(llvm::None, llvm::None);
  MachineOp theCache =
      helper.builder.create<MachineOp>(unknLoc, "cache", fType);
  Block *cacheEntry = theCache.addEntryBlock();
  helper.builder.setInsertionPointToStart(cacheEntry);

  // make a variable op which returns this type
  VariableOp varOp = helper.builder.create<VariableOp>(unknLoc, rangeType,
                                                       nullptr, "idTypeVar");

  std::string str;
  llvm::raw_string_ostream stream(str);
  varOp.print(stream);

  auto expctOut = "%idTypeVar = fsm.variable \"idTypeVar\" : !fsm.range<0, 3>";

  EXPECT_STREQ(expctOut, str.c_str());
}

TEST(OpTypes, RangeTypeParse) {
  OpHelper helper;
  llvm::StringRef OpText =
      "fsm.machine @cache (){\n"
      "  %idTypeVar = fsm.variable \"idTypeVar\" : !fsm.range<0, 3>\n"
      "}\n";

  auto result = parseSourceString(OpText, &helper.ctx);

  result->walk([](VariableOp vOp) {
    EXPECT_TRUE(vOp.getType().isa<RangeType>());
    EXPECT_EQ(vOp.name(), "idTypeVar");
    EXPECT_FALSE(vOp.initValue().hasValue());
    RangeType r = vOp.getType().cast<RangeType>();
    EXPECT_EQ(r.getStart(), 0);
    EXPECT_EQ(r.getEnd(), 3);
  });
}

TEST(OpTypes, SetType){
  OpHelper helper;
  Location unknLoc = helper.builder.getUnknownLoc();

  SetType setType = SetType::get(IDType::get(&helper.ctx), 3);

  // make a machine op
  FunctionType fType = helper.builder.getFunctionType(llvm::None, llvm::None);
  MachineOp theCache =
      helper.builder.create<MachineOp>(unknLoc, "cache", fType);
  Block *cacheEntry = theCache.addEntryBlock();
  helper.builder.setInsertionPointToStart(cacheEntry);

  // make a variable op which returns this type
  VariableOp varOp = helper.builder.create<VariableOp>(unknLoc, setType,
                                                       nullptr, "idTypeVar");

  std::string str;
  llvm::raw_string_ostream stream(str);
  varOp.print(stream);

  auto expctOut = "%idTypeVar = fsm.variable \"idTypeVar\" : !fsm.set<!fsm.id, 3>";

  EXPECT_STREQ(expctOut, str.c_str());
}

TEST(OpTypes, SetTypeParse) {
  OpHelper helper;
  llvm::StringRef OpText =
      "fsm.machine @cache (){\n"
      "  %idTypeVar = fsm.variable \"idTypeVar\" : !fsm.set<!fsm.id, 3>\n"
      "}\n";

  auto result = parseSourceString(OpText, &helper.ctx);

  result->walk([](VariableOp vOp) {
    EXPECT_TRUE(vOp.getType().isa<SetType>());
    EXPECT_EQ(vOp.name(), "idTypeVar");
    EXPECT_FALSE(vOp.initValue().hasValue());
    SetType r = vOp.getType().cast<SetType>();
    EXPECT_TRUE(r.getElementType().isa<IDType>());
    EXPECT_EQ(r.getNumElements(), 3);
  });
}
