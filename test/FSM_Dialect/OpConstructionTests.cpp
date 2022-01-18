//
// Created by petr on 10/01/2022.
//
#include "FSM/FSMOps.h"
#include "OpHelper.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Parser.h"
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::fsm;

TEST(OpConstructionTests, PrintMachineOp) {

  OpHelper helper;

  FunctionType type = helper.builder.getFunctionType(llvm::None, llvm::None);

  auto machineOp = helper.builder.create<MachineOp>(
      helper.builder.getUnknownLoc(), "cache", type);
  Block *entry = machineOp.addEntryBlock();

  //// add a constant to the body
  // set the insertion point to inside the body of the machine
  helper.builder.setInsertionPointToStart(entry);

  // create the constant op
  auto initAttr = helper.builder.getI64IntegerAttr(21);
  helper.builder.create<VariableOp>(helper.builder.getUnknownLoc(),
                                    initAttr.getType(), initAttr, "some_count");

  // print operation to string
  std::string str;
  llvm::raw_string_ostream stream{str};
  machineOp.print(stream);

  auto expectedStr = "fsm.machine @cache() {\n"
                     "  %some_count = fsm.variable \"some_count\" {initValue = "
                     "21 : i64} : i64\n"
                     "}";

  ASSERT_STREQ(str.c_str(), expectedStr);
  ASSERT_TRUE(succeeded(helper.module.verify()));
}

TEST(OpConstructionTests, ParseMachineOp) {
  OpHelper helper;
  llvm::StringRef machineOp =
      "fsm.machine @cache() {\n  %some_count = fsm.variable "
      "\"some_count\" {initValue = 21 : i64} : i64\n}";
  auto result = parseSourceString<ModuleOp>(machineOp, &helper.ctx);
  result->walk([&](MachineOp op) {
    ASSERT_STREQ(op.sym_name().str().c_str(), "cache");
  });
  result->walk([&](VariableOp op) {
    ASSERT_EQ(op.initValue().getValue().cast<IntegerAttr>().getInt(), 21);
  });
  ASSERT_TRUE(succeeded(helper.module.verify()));
}

TEST(OpConstructionTests, StateOp) {
  OpHelper helper;

  // set up the test

  // first build the machine op
  auto machType = helper.builder.getFunctionType(llvm::None, llvm::None);
  auto mach = helper.builder.create<MachineOp>(helper.builder.getUnknownLoc(),
                                               "cache", machType);
  // set the insertion point
  Block *entry = mach.addEntryBlock();
  helper.builder.setInsertionPointToStart(entry);

  // create the state op
  auto stateOp =
      helper.builder.create<StateOp>(helper.builder.getUnknownLoc(), "S");
  Block *sEntry = stateOp.addEntryBlock();
  helper.builder.setInsertionPointToStart(sEntry);
  // add the constant
  helper.builder.create<ConstantOp>(helper.builder.getUnknownLoc(),
                                    helper.builder.getI64IntegerAttr(21));

  // print operation to string
  std::string str;
  llvm::raw_string_ostream stream{str};
  mach.print(stream);
  auto expectedText = "fsm.machine @cache() {\n"
                      "  fsm.state @S transitions  {\n"
                      "    %c21_i64 = constant 21 : i64\n"
                      "  }\n"
                      "}";

  ASSERT_STREQ(expectedText, str.c_str());
  ASSERT_TRUE(succeeded(helper.module.verify()));
}

TEST(OpConstructionTests, ParseStateOp) {
  OpHelper helper;
  llvm::StringRef opText = "fsm.machine @cache() {\n"
                           "  fsm.state @S transitions  {\n"
                           "    %0 = std.constant 21 : i64\n "
                           "  }\n"
                           "}";

  auto result = parseSourceString(opText, &helper.ctx);

  result->walk([](StateOp op) { ASSERT_TRUE(op.sym_name() == "S"); });

  ASSERT_TRUE(succeeded(result->verify()));
  ASSERT_TRUE(succeeded(helper.module.verify()));
}

TEST(OpConstructionTests, PrintTransitionOp) {
  OpHelper helper;
  // set up the test
  Location unkLoc = helper.builder.getUnknownLoc();

  // create the necessary ops
  FunctionType machType =
      helper.builder.getFunctionType(llvm::None, llvm::None);
  MachineOp machOp =
      helper.builder.create<MachineOp>(unkLoc, "cache", machType);
  Block *entry = machOp.addEntryBlock();
  helper.builder.setInsertionPointToStart(entry);

  StateOp sState = helper.builder.create<StateOp>(unkLoc, "S");
  Block *sEntry = sState.addEntryBlock();
  helper.builder.setInsertionPointToStart(sEntry);

  // create the transition Op
  FunctionType transFnType =
      helper.builder.getFunctionType(llvm::None, llvm::None);
  TransitionOp transOp = helper.builder.create<TransitionOp>(
      unkLoc, "load", transFnType, helper.builder.getSymbolRefAttr("M"));
  Block *tEntry = transOp.addEntryBlock();
  helper.builder.setInsertionPointToStart(tEntry);

  // create a dummy op
  helper.builder.create<ConstantOp>(unkLoc,
                                    helper.builder.getI64IntegerAttr(21));

  // print operation to string
  std::string str;
  llvm::raw_string_ostream stream{str};
  machOp.print(stream);

  auto expctStr = "fsm.machine @cache() {\n"
                  "  fsm.state @S transitions  {\n"
                  "    fsm.transition @load() attributes {nextState = @M} {\n"
                  "      %c21_i64 = constant 21 : i64\n"
                  "    }\n"
                  "  }\n"
                  "}";

  ASSERT_STREQ(expctStr, str.c_str());
  ASSERT_TRUE(succeeded(helper.module.verify()));
}

TEST(OpConstructionTests, ParseTransitionOp) {
  OpHelper helper;
  llvm::StringRef opText =
      "fsm.machine @cache() {\n"
      "  fsm.state @S transitions  {\n"
      "    fsm.transition @load() attributes {nextState = @M} {\n"
      "      %c21_i64 = constant 21 : i64\n"
      "    }\n"
      "  }\n"
      "}";

  auto result = parseSourceString(opText, &helper.ctx);

  result->walk([](TransitionOp op) {
    ASSERT_TRUE(op.nextState().hasValue());
    ASSERT_TRUE(op.nextState().getValue().getLeafReference() == "M");
    ASSERT_TRUE(op.sym_name() == "load");
  });

  auto logRes = result->verify();
  ASSERT_TRUE(succeeded(logRes));
  ASSERT_TRUE(succeeded(helper.module.verify()));
}

TEST(OpConstructionTests, SymbolTableTest) {
  OpHelper helper;
  Location unknLoc = helper.builder.getUnknownLoc();

  FunctionType machType =
      helper.builder.getFunctionType(llvm::None, llvm::None);
  MachineOp machOp =
      helper.builder.create<MachineOp>(unknLoc, "cache", machType);
  Block *entry = machOp.addEntryBlock();
  helper.builder.setInsertionPointToStart(entry);

  helper.builder.create<VariableOp>(unknLoc, helper.builder.getI64Type(),
                                    helper.builder.getI64IntegerAttr(21),
                                    "some_value");

  // create a state
  StateOp sState = helper.builder.create<StateOp>(unknLoc, "S");
  Block *sEntry = sState.addEntryBlock();
  helper.builder.setInsertionPointToStart(sEntry);

  // create a transition
  FunctionType sLoadFnType =
      helper.builder.getFunctionType(llvm::None, llvm::None);
  TransitionOp sLoadTran = helper.builder.create<TransitionOp>(
      unknLoc, "load", sLoadFnType, helper.builder.getSymbolRefAttr("M"));
  Block *sLoadTransBlock = sLoadTran.addEntryBlock();
  helper.builder.setInsertionPointToStart(sLoadTransBlock);

  // add a dummy op
  helper.builder.create<ConstantOp>(unknLoc,
                                    helper.builder.getI64IntegerAttr(21));

  // create a new state
  helper.builder.setInsertionPointAfter(sState);
  StateOp mState = helper.builder.create<StateOp>(unknLoc, "M");
  Block *mEntry = mState.addEntryBlock();
  helper.builder.setInsertionPointToStart(mEntry);

  // add a transition to M state
  FunctionType mLoadFnType =
      helper.builder.getFunctionType(llvm::None, llvm::None);
  TransitionOp mLoadTran = helper.builder.create<TransitionOp>(
      unknLoc, "load", mLoadFnType, helper.builder.getSymbolRefAttr("S"));
  Block *mLoadTranEntry = mLoadTran.addEntryBlock();
  helper.builder.setInsertionPointToStart(mLoadTranEntry);
  // add a dummy op
  helper.builder.create<ConstantOp>(unknLoc,
                                    helper.builder.getI64IntegerAttr(33));

  /// Now we want to check how we can use Symbol Table to get the information
  /// we need about transactions

  // 1 - get the cache
  auto cacheOp = helper.module.lookupSymbol<MachineOp>("cache");
  ASSERT_NE(cacheOp, nullptr);

  // 2 - get the S state within the cache
  auto sStateLookUp = cacheOp.lookupSymbol<StateOp>("S");
  ASSERT_NE(sStateLookUp, nullptr);

  // 3 - find a transition
  auto sLoadTransLookUp = sStateLookUp.lookupSymbol<TransitionOp>("load");
  ASSERT_NE(sLoadTransLookUp, nullptr);
  ASSERT_TRUE(sLoadTransLookUp.nextState().getValue().getLeafReference() ==
              "M");
  // lookup a symbol by a symbol ref attr from another op
  auto lookup = cacheOp.lookupSymbol<StateOp>(sLoadTransLookUp.nextStateAttr());
  ASSERT_NE(lookup, nullptr);

  // 4 - try and lookup the state linked from
  auto nextStateLookUp = cacheOp.lookupSymbol<StateOp>(
      sLoadTransLookUp.nextStateAttr().getLeafReference());
  ASSERT_NE(nextStateLookUp, nullptr);

  // 5 - go directly to a transition from cache Op
  auto symbolRef = helper.builder.getSymbolRefAttr(
      "S", {helper.builder.getSymbolRefAttr("load")});

  auto dirTrans = cacheOp.lookupSymbol<TransitionOp>(symbolRef);
  ASSERT_NE(dirTrans, nullptr);

  ASSERT_TRUE(succeeded(helper.module.verify()));
}

TEST(OpConstructionTests, MessageOpConstruction) {
  OpHelper helper;
  Location unknLoc = helper.builder.getUnknownLoc();

  // create some dummy constant ops to be passed in as operands
  auto c1 = helper.builder.create<ConstantOp>(
      unknLoc, helper.builder.getI64IntegerAttr(1));
  auto c2 = helper.builder.create<ConstantOp>(
      unknLoc, helper.builder.getI64IntegerAttr(2));
  auto c3 = helper.builder.create<ConstantOp>(
      unknLoc, helper.builder.getI64IntegerAttr(3));

  ValueRange vr = {c1, c2, c3};

  helper.builder.create<MessageOp>(unknLoc, MsgType::get(&helper.ctx),
                                   helper.builder.getSymbolRefAttr("Resp"),
                                   helper.builder.getStringAttr("Inv"), vr);

  auto expctText = "module  {\n"
                   "  %c1_i64 = constant 1 : i64\n"
                   "  %c2_i64 = constant 2 : i64\n"
                   "  %c3_i64 = constant 3 : i64\n"
                   "  %0 = fsm.message @Resp \"Inv\" %c1_i64, %c2_i64, %c3_i64 "
                   ": i64, i64, i64 -> !fsm.msg\n"
                   "}\n";

  // print operation to string
  std::string str;
  llvm::raw_string_ostream stream{str};
  helper.module.print(stream);

  ASSERT_STREQ(str.c_str(), expctText);
  ASSERT_TRUE(succeeded(helper.module.verify()));
}

TEST(OpConstructionTests, MessageOpParse) {
  OpHelper helper;
  llvm::StringRef parseText =
      "module  {\n"
      "  %c1_i64 = constant 1 : i64\n"
      "  %c2_i64 = constant 2 : i64\n"
      "  %c3_i64 = constant 3 : i64\n"
      "  %0 = fsm.message @Resp \"Inv\" %c1_i64, %c2_i64, %c3_i64 "
      ": i64, i64, i64 -> !fsm.msg\n"
      "}\n";

  auto result = parseSourceString(parseText, &helper.ctx);

  result->walk([](MessageOp msgOp) {
    EXPECT_EQ(msgOp.msgTypeAttr().getLeafReference(), "Resp");
    EXPECT_EQ(msgOp.msgNameAttr().getValue(), "Inv");
    EXPECT_EQ(msgOp.inputs().size(), 3);
    EXPECT_TRUE(msgOp.getResult().getType().isa<MsgType>());
  });
  ASSERT_TRUE(succeeded(helper.module.verify()));
}

TEST(OpConstructionTests, MessageOpInvResultType) {
  OpHelper helper;
  llvm::StringRef parseText =
      "module  {\n"
      "  %c1_i64 = constant 1 : i64\n"
      "  %c2_i64 = constant 2 : i64\n"
      "  %c3_i64 = constant 3 : i64\n"
      "  %0 = fsm.message @Resp \"Inv\" %c1_i64, %c2_i64, %c3_i64 "
      ": i64, i64, i64 -> i64\n"
      "}\n";

  // parsing should fail and return empty result
  auto result = parseSourceString(parseText, &helper.ctx);
  ASSERT_EQ(*result, nullptr);
}

TEST(OpConstructionTests, TransitionOpWithMessageArgument) {
  OpHelper helper;
  Location unknLoc = helper.builder.getUnknownLoc();

  // cache op
  FunctionType cacheFnType =
      helper.builder.getFunctionType(llvm::None, llvm::None);
  MachineOp cacheOp =
      helper.builder.create<MachineOp>(unknLoc, "cache", cacheFnType);
  Block *cacheEntry = cacheOp.addEntryBlock();
  helper.builder.setInsertionPointToStart(cacheEntry);

  // simple state
  StateOp sState = helper.builder.create<StateOp>(unknLoc, "S");
  Block *stateEntry = sState.addEntryBlock();
  helper.builder.setInsertionPointToStart(stateEntry);

  // simple transition with msg argument
  FunctionType transType =
      helper.builder.getFunctionType({MsgType::get(&helper.ctx)}, llvm::None);
  TransitionOp transOp = helper.builder.create<TransitionOp>(
      unknLoc, "store", transType, /*nextStateAttr*/ nullptr);
  Block *transEntry = transOp.addEntryBlock();
  helper.builder.setInsertionPointToStart(transEntry);

  // access a member of the msg
  helper.builder.create<AccessOp>(
      unknLoc, DataType::get(helper.builder.getContext()),
      transOp.getArgument(0), helper.builder.getStringAttr("cl"));

  // print out the IR
  std::string str;
  llvm::raw_string_ostream sstream(str);
  transOp.print(sstream);

  const char *expectedText =
      "fsm.transition @store(%arg0: !fsm.msg) {\n"
      "  %0 = fsm.access {memberId = \"cl\"} %arg0 : !fsm.msg -> !fsm.data\n"
      "}";

  EXPECT_STREQ(str.c_str(), expectedText);
  ASSERT_TRUE(succeeded(helper.module.verify()));
}

TEST(OpConstructionTests, AccessOpCreate) {
  OpHelper helper;
  Location unknLoc = helper.builder.getUnknownLoc();

  FunctionType fnType =
      helper.builder.getFunctionType({MsgType::get(&helper.ctx)}, {});
  MachineOp cacheOp =
      helper.builder.create<MachineOp>(unknLoc, "cache", fnType);
  Block *cacheEntry = cacheOp.addEntryBlock();
  helper.builder.setInsertionPointToStart(cacheEntry);

  // create some access Ops
  helper.builder.create<AccessOp>(unknLoc, DataType::get(&helper.ctx),
                                  cacheOp.getArgument(0),
                                  helper.builder.getStringAttr("cl"));

  helper.builder.create<AccessOp>(unknLoc, IDType::get(&helper.ctx),
                                  cacheOp.getArgument(0),
                                  helper.builder.getStringAttr("src"));

  // print the module
  std::string str;
  llvm::raw_string_ostream sstream(str);
  cacheOp.print(sstream);

  const auto expectedText =
      "fsm.machine @cache(%arg0: !fsm.msg) {\n"
      "  %0 = fsm.access {memberId = \"cl\"} %arg0 : !fsm.msg -> !fsm.data\n"
      "  %1 = fsm.access {memberId = \"src\"} %arg0 : !fsm.msg -> !fsm.id\n"
      "}";
  ASSERT_STREQ(str.c_str(), expectedText);
}

TEST(OpConstructionTests, AccessOpParse) {
  OpHelper helper;
  const llvm::StringRef opText =
      "fsm.machine @cache(%arg0: !fsm.msg) {\n"
      "  %0 = fsm.access {memberId = \"cl\"} %arg0 : !fsm.msg -> !fsm.data\n"
      "  %1 = fsm.access {memberId = \"src\"} %arg0 : !fsm.msg -> !fsm.id\n"
      "}";
  auto result = parseSourceString(opText, &helper.ctx);
  ASSERT_NE(*result, nullptr); // parsing succeeded

  result->walk([](AccessOp op) {
    if (op.memberId() == "cl") {
      ASSERT_TRUE(op.getResult().getType().isa<DataType>());
      ASSERT_TRUE(op.msg().getType().isa<MsgType>());
    } else {
      ASSERT_EQ(op.memberId(), "src");
      ASSERT_TRUE(op.getResult().getType().isa<IDType>());
      ASSERT_TRUE(op.msg().getType().isa<MsgType>());
    }
  });

  ASSERT_TRUE(succeeded(result->verify()));
}

TEST(OpConstructionTests, AccessOpWithInvArgumentType) {
  OpHelper helper;
  const llvm::StringRef opText =
      "fsm.machine @cache(%arg0: !fsm.data) {\n"
      "  %0 = fsm.access {memberId = \"cl\"} %arg0 : !fsm.data -> !fsm.data\n"
      "  %1 = fsm.access {memberId = \"src\"} %arg0 : !fsm.data -> !fsm.id\n"
      "}";
  auto result = parseSourceString(opText, &helper.ctx);
  ASSERT_EQ(*result, nullptr); // parsing failed
}

TEST(OpConstructionTests, UpdateOp) {
  OpHelper helper;
  Location unknLoc = helper.builder.getUnknownLoc();

  FunctionType cacheFnType = helper.builder.getFunctionType({}, {});
  MachineOp cacheOp =
      helper.builder.create<MachineOp>(unknLoc, "cache", cacheFnType);
  Block *cacheEntry = cacheOp.addEntryBlock();
  helper.builder.setInsertionPointToStart(cacheEntry);

  // create a variable op
  auto intAttr = helper.builder.getI64IntegerAttr(22);
  VariableOp varOp = helper.builder.create<VariableOp>(
      unknLoc, intAttr.getType(), intAttr, "State");

  // create a constant op
  ConstantOp ssaVal = helper.builder.create<ConstantOp>(
      unknLoc, helper.builder.getI64IntegerAttr(33));

  // create the update op
  helper.builder.create<UpdateOp>(unknLoc, varOp, ssaVal);

  std::string str;
  llvm::raw_string_ostream sstream(str);
  cacheOp.print(sstream);

  auto expectedText =
      "fsm.machine @cache() {\n"
      "  %State = fsm.variable \"State\" {initValue = 22 : i64} : i64\n"
      "  %c33_i64 = constant 33 : i64\n"
      "  fsm.update %State, %c33_i64 : i64\n"
      "}";
  ASSERT_STREQ(str.c_str(), expectedText);
}

TEST(OpConstructionTests, UpdateOpParse) {
  OpHelper helper;
  llvm::StringRef opText =
      "fsm.machine @cache() {\n"
      "  %State = fsm.variable \"State\" {initValue = 22 : i64} : i64\n"
      "  %c33_i64 = constant 33 : i64\n"
      "  fsm.update %State, %c33_i64 : i64\n"
      "}";

  auto result = parseSourceString(opText, &helper.ctx);
  ASSERT_NE(*result, nullptr);
  ASSERT_TRUE(succeeded(result->verify()));
}

TEST(OpConstructionTests, NoOpPrint) {
  OpHelper helper;
  Location unknLoc = helper.builder.getUnknownLoc();

  helper.builder.create<NOPOp>(unknLoc);

  std::string str;
  llvm::raw_string_ostream sstream(str);
  helper.module.print(sstream);

  auto expectedOpText = "module  {\n"
                        "  fsm.nop\n"
                        "}\n";
  ASSERT_STREQ(str.c_str(), expectedOpText);
}

TEST(OpConstructionTests, NoOpParse) {
  OpHelper helper;
  llvm::StringRef opText = "module  {\n"
                           "  fsm.nop\n"
                           "}\n";
  auto result = parseSourceString(opText, &helper.ctx);
  ASSERT_NE(*result, nullptr);
}

TEST(OpConstructionTests, ReferenceOpPrint) {
  OpHelper helper;
  Location unknLoc = helper.builder.getUnknownLoc();

  helper.builder.create<ReferenceOp>(unknLoc, IDType::get(&helper.ctx),
                                     helper.builder.getSymbolRefAttr("cache"));

  std::string str;
  llvm::raw_string_ostream sstream(str);
  helper.module.print(sstream);

  auto expectedText = "module  {\n"
                      "  %0 = fsm.ref @cache\n"
                      "}\n";
  ASSERT_STREQ(str.c_str(), expectedText);
}

TEST(OpConstructionTests, ReferenceOpParse) {
  OpHelper helper;
  llvm::StringRef opText = "module  {\n"
                           "  %0 = fsm.ref @cache\n"
                           "}\n";
  auto result = parseSourceString(opText, &helper.ctx);
  ASSERT_NE(*result, nullptr);
  result->walk([](ReferenceOp refOp) {
    EXPECT_EQ(refOp.reference().getLeafReference(), "cache");
    EXPECT_TRUE(refOp.result().getType().isa<IDType>());
  });
}

TEST(OpConstructionTests, IfOpPrintWithoutElse) {
  OpHelper helper;
  Location unknLoc = helper.builder.getUnknownLoc();

  ConstantOp trueValue = helper.builder.create<ConstantOp>(
      unknLoc, helper.builder.getBoolAttr(true));
  IfOp ifOp = helper.builder.create<IfOp>(unknLoc, trueValue, false);
  helper.builder.setInsertionPointToStart(ifOp.thenBlock());
  helper.builder.create<NOPOp>(unknLoc);

  std::string str;
  llvm::raw_string_ostream sstream(str);
  helper.module.print(sstream);

  auto expectedText = "module  {\n"
                      "  %true = constant true\n"
                      "  fsm.if %true {\n"
                      "    fsm.nop\n"
                      "  }\n"
                      "}\n";
  ASSERT_STREQ(str.c_str(), expectedText);
}

TEST(OpConstructionTests, IfOpParseWithoutElse) {
  OpHelper helper;
  llvm::StringRef opText = "module  {\n"
                           "  %true = constant true\n"
                           "  fsm.if %true {\n"
                           "    fsm.nop\n"
                           "  }\n"
                           "}\n";

  auto result = parseSourceString(opText, &helper.ctx);
  ASSERT_NE(*result, nullptr);
  result->walk([](IfOp ifOp){
    EXPECT_FALSE(ifOp.thenBlock()->empty());
    EXPECT_EQ(ifOp.elseBlock(), nullptr);
  });
}

TEST(OpConstructionTests, IfOpPrintWithElseRegion){
  OpHelper helper;
  Location unknLoc = helper.builder.getUnknownLoc();

  ConstantOp trueValue = helper.builder.create<ConstantOp>(
      unknLoc, helper.builder.getBoolAttr(true));
  IfOp ifOp = helper.builder.create<IfOp>(unknLoc, trueValue, true);
  helper.builder.setInsertionPointToStart(ifOp.thenBlock());
  helper.builder.create<NOPOp>(unknLoc);
  helper.builder.setInsertionPointToStart(ifOp.elseBlock());
  helper.builder.create<NOPOp>(unknLoc);

  std::string str;
  llvm::raw_string_ostream sstream(str);
  helper.module.print(sstream);

  auto expectedText = "module  {\n"
                      "  %true = constant true\n"
                      "  fsm.if %true {\n"
                      "    fsm.nop\n"
                      "  } else {\n"
                      "    fsm.nop\n"
                      "  }\n"
                      "}\n";
  ASSERT_STREQ(str.c_str(), expectedText);
}

TEST(OpConstructionTests, IfOpParseWithElse) {
  OpHelper helper;
  llvm::StringRef opText  = "module  {\n"
                           "  %true = constant true\n"
                           "  fsm.if %true {\n"
                           "    fsm.nop\n"
                           "  } else {\n"
                           "    fsm.nop\n"
                           "  }\n"
                           "}\n";

  auto result = parseSourceString(opText, &helper.ctx);
  ASSERT_NE(*result, nullptr);
  result->walk([](IfOp ifOp){
    EXPECT_FALSE(ifOp.thenBlock()->empty());
    EXPECT_FALSE(ifOp.elseBlock()->empty());
  });
}