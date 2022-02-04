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

  std::vector<Value> vr = {c1, c2, c3};
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
      "  fsm.update %State, %c33_i64 : i64, i64\n"
      "}";
  ASSERT_STREQ(str.c_str(), expectedText);
}

TEST(OpConstructionTests, UpdateOpParse) {
  OpHelper helper;
  llvm::StringRef opText =
      "fsm.machine @cache() {\n"
      "  %State = fsm.variable \"State\" {initValue = 22 : i64} : i64\n"
      "  %c33_i64 = constant 33 : i64\n"
      "  fsm.update %State, %c33_i64 : i64, i64\n"
      "}";

  auto result = parseSourceString(opText, &helper.ctx);
  ASSERT_NE(*result, nullptr);
  ASSERT_TRUE(succeeded(result->verify()));
}

TEST(OpConstrutionTests, UpdateOpRangeVarWithInt) {
  OpHelper helper;
  llvm::StringRef opText =
      "fsm.machine @cache() {\n"
      "  %my_range = fsm.variable \"range\" {initValue = 0 : i64} : "
      "!fsm.range<0,3>\n"
      "  %c2_i64 = constant 2 : i64\n"
      "  fsm.update %my_range, %c2_i64 : !fsm.range<0,3>, i64\n"
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
  result->walk([](IfOp ifOp) {
    EXPECT_FALSE(ifOp.thenBlock()->empty());
    EXPECT_EQ(ifOp.elseBlock(), nullptr);
  });
}

TEST(OpConstructionTests, IfOpPrintWithElseRegion) {
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
  llvm::StringRef opText = "module  {\n"
                           "  %true = constant true\n"
                           "  fsm.if %true {\n"
                           "    fsm.nop\n"
                           "  } else {\n"
                           "    fsm.nop\n"
                           "  }\n"
                           "}\n";

  auto result = parseSourceString(opText, &helper.ctx);
  ASSERT_NE(*result, nullptr);
  result->walk([](IfOp ifOp) {
    EXPECT_FALSE(ifOp.thenBlock()->empty());
    EXPECT_FALSE(ifOp.elseBlock()->empty());
  });
}

TEST(OpConstructionTests, BreakOpPrint) {
  OpHelper helper;
  helper.builder.create<BreakOp>(helper.builder.getUnknownLoc());

  std::string str;
  llvm::raw_string_ostream sstream(str);
  helper.module.print(sstream);

  auto expectedText = "module  {\n"
                      "  fsm.break\n"
                      "}\n";
  ASSERT_STREQ(str.c_str(), expectedText);
}

TEST(OpConstructionTests, BreakOpParse) {
  OpHelper helper;
  llvm::StringRef opText = "module  {\n"
                           "  fsm.break\n"
                           "}\n";
  auto result = parseSourceString(opText, &helper.ctx);
  ASSERT_NE(*result, nullptr);
}

TEST(OpConstructionTests, NetworkOpPrint) {
  OpHelper helper;
  helper.builder.create<NetworkOp>(helper.builder.getUnknownLoc(),
                                   NetworkType::get(&helper.ctx), "ordered",
                                   helper.builder.getSymbolRefAttr("fwd"));
  std::string str;
  llvm::raw_string_ostream sstream(str);
  helper.module.print(sstream);
  auto expectedText = "module  {\n"
                      "  %fwd = fsm.network @fwd \"ordered\"\n"
                      "}\n";
  EXPECT_STREQ(expectedText, str.c_str());
}

TEST(OpConstructionTests, NetworkOpParse) {
  OpHelper helper;
  auto sourceText = "module  {\n"
                    "  %fwd = fsm.network @fwd \"ordered\"\n"
                    "}\n";
  auto result = parseSourceString(sourceText, &helper.ctx);
  ASSERT_NE(*result, nullptr);
  result->walk([](NetworkOp op) {
    EXPECT_EQ(op.sym_name().getLeafReference(), "fwd");
    EXPECT_EQ(op.ordering(), "ordered");
    EXPECT_TRUE(op.getType().isa<NetworkType>());
  });
}

TEST(OpConstructionTests, NetworkOpInvalidOrdering) {
  OpHelper helper;
  auto netop = helper.builder.create<NetworkOp>(
      helper.builder.getUnknownLoc(), NetworkType::get(&helper.ctx), "blah",
      helper.builder.getSymbolRefAttr("fwd"));

  EXPECT_TRUE(failed(netop.verify()));
}

TEST(OpConstructionTests, SendOp) {
  OpHelper helper;
  Location unknLoc = helper.builder.getUnknownLoc();

  // create a network
  auto network = helper.builder.create<NetworkOp>(
      unknLoc, NetworkType::get(&helper.ctx), "ordered",
      helper.builder.getSymbolRefAttr("testnet"));

  auto cache = helper.builder.create<MachineOp>(
      unknLoc, "cache", helper.builder.getFunctionType({}, {}));
  Block *cacheEntry = cache.addEntryBlock();
  helper.builder.setInsertionPointToStart(cacheEntry);

  // create a basic state
  auto state1 = helper.builder.create<StateOp>(unknLoc, "s1");
  Block *s1Entry = state1.addEntryBlock();
  helper.builder.setInsertionPointToStart(s1Entry);

  // create a transition with a msg param
  FunctionType t1Type =
      helper.builder.getFunctionType({MsgType::get(&helper.ctx)}, {});
  auto trans1 =
      helper.builder.create<TransitionOp>(unknLoc, "t1", t1Type, nullptr);
  Block *t1Entry = trans1.addEntryBlock();
  helper.builder.setInsertionPointToStart(t1Entry);

  // create a send op
  helper.builder.create<SendOp>(unknLoc, network, trans1.getArgument(0));

  // print out the result
  std::string str;
  llvm::raw_string_ostream sstream(str);
  helper.module.print(sstream);

  auto expectedText = "module  {\n"
                      "  %testnet = fsm.network @testnet \"ordered\"\n"
                      "  fsm.machine @cache() {\n"
                      "    fsm.state @s1 transitions  {\n"
                      "      fsm.transition @t1(%arg0: !fsm.msg) {\n"
                      "        fsm.send %testnet %arg0\n"
                      "      }\n"
                      "    }\n"
                      "  }\n"
                      "}\n";
  EXPECT_STREQ(str.c_str(), expectedText);
}

TEST(OpConstructionTests, SendOpParse) {
  OpHelper helper;
  auto opText = "module  {\n"
                "  %testnet = fsm.network @testnet \"ordered\"\n"
                "  fsm.machine @cache() {\n"
                "    fsm.state @s1 transitions  {\n"
                "      fsm.transition @t1(%arg0: !fsm.msg) {\n"
                "        fsm.send %testnet %arg0\n"
                "      }\n"
                "    }\n"
                "  }\n"
                "}\n";

  auto result = parseSourceString(opText, &helper.ctx);
  ASSERT_NE(*result, nullptr);

  result->walk([](SendOp sendOp) {
    EXPECT_TRUE(sendOp.message().getType().isa<MsgType>());
    EXPECT_TRUE(sendOp.network().getType().isa<NetworkType>());
  });
}

TEST(OpConstructionTests, DeferMsg) {
  OpHelper helper;
  Location unknLoc = helper.builder.getUnknownLoc();

  // create a network
  /*auto network = */ helper.builder.create<NetworkOp>(
      unknLoc, NetworkType::get(&helper.ctx), "ordered",
      helper.builder.getSymbolRefAttr("testnet"));

  auto cache = helper.builder.create<MachineOp>(
      unknLoc, "cache", helper.builder.getFunctionType({}, {}));
  Block *cacheEntry = cache.addEntryBlock();
  helper.builder.setInsertionPointToStart(cacheEntry);

  // create a basic state
  auto state1 = helper.builder.create<StateOp>(unknLoc, "s1");
  Block *s1Entry = state1.addEntryBlock();
  helper.builder.setInsertionPointToStart(s1Entry);

  // create a transition with a msg param
  FunctionType t1Type =
      helper.builder.getFunctionType({MsgType::get(&helper.ctx)}, {});
  auto trans1 =
      helper.builder.create<TransitionOp>(unknLoc, "t1", t1Type, nullptr);
  Block *t1Entry = trans1.addEntryBlock();
  helper.builder.setInsertionPointToStart(t1Entry);

  // create a send op
  helper.builder.create<DeferMsg>(unknLoc, trans1.getArgument(0));

  // print out the result
  std::string str;
  llvm::raw_string_ostream sstream(str);
  helper.module.print(sstream);

  auto expectedText = "module  {\n"
                      "  %testnet = fsm.network @testnet \"ordered\"\n"
                      "  fsm.machine @cache() {\n"
                      "    fsm.state @s1 transitions  {\n"
                      "      fsm.transition @t1(%arg0: !fsm.msg) {\n"
                      "        fsm.defer %arg0\n"
                      "      }\n"
                      "    }\n"
                      "  }\n"
                      "}\n";
  EXPECT_STREQ(str.c_str(), expectedText);
}

TEST(OpConstructionTests, DeferMsgParse) {
  OpHelper helper;
  auto opText = "module  {\n"
                "  %testnet = fsm.network @testnet \"ordered\"\n"
                "  fsm.machine @cache() {\n"
                "    fsm.state @s1 transitions  {\n"
                "      fsm.transition @t1(%arg0: !fsm.msg) {\n"
                "        fsm.defer %arg0\n"
                "      }\n"
                "    }\n"
                "  }\n"
                "}\n";

  auto result = parseSourceString(opText, &helper.ctx);
  ASSERT_NE(*result, nullptr);

  result->walk([](DeferMsg deferMsg) {
    EXPECT_TRUE(deferMsg.message().getType().isa<MsgType>());
  });
}

TEST(OpConstructionTests, SendDeferMsg) {
  OpHelper helper;
  Location unknLoc = helper.builder.getUnknownLoc();

  // create a network
  /*auto network = */ helper.builder.create<NetworkOp>(
      unknLoc, NetworkType::get(&helper.ctx), "ordered",
      helper.builder.getSymbolRefAttr("testnet"));

  auto cache = helper.builder.create<MachineOp>(
      unknLoc, "cache", helper.builder.getFunctionType({}, {}));
  Block *cacheEntry = cache.addEntryBlock();
  helper.builder.setInsertionPointToStart(cacheEntry);

  // create a basic state
  auto state1 = helper.builder.create<StateOp>(unknLoc, "s1");
  Block *s1Entry = state1.addEntryBlock();
  helper.builder.setInsertionPointToStart(s1Entry);

  // create a transition with a msg param
  FunctionType t1Type =
      helper.builder.getFunctionType({MsgType::get(&helper.ctx)}, {});
  auto trans1 =
      helper.builder.create<TransitionOp>(unknLoc, "t1", t1Type, nullptr);
  Block *t1Entry = trans1.addEntryBlock();
  helper.builder.setInsertionPointToStart(t1Entry);

  // create a send op
  helper.builder.create<SendDeferMsg>(unknLoc, trans1.getArgument(0));

  // print out the result
  std::string str;
  llvm::raw_string_ostream sstream(str);
  helper.module.print(sstream);

  auto expectedText = "module  {\n"
                      "  %testnet = fsm.network @testnet \"ordered\"\n"
                      "  fsm.machine @cache() {\n"
                      "    fsm.state @s1 transitions  {\n"
                      "      fsm.transition @t1(%arg0: !fsm.msg) {\n"
                      "        fsm.defer_send %arg0\n"
                      "      }\n"
                      "    }\n"
                      "  }\n"
                      "}\n";
  EXPECT_STREQ(str.c_str(), expectedText);
}

TEST(OpConstructionTests, SendDeferMsgParse) {
  OpHelper helper;
  auto opText = "module  {\n"
                "  %testnet = fsm.network @testnet \"ordered\"\n"
                "  fsm.machine @cache() {\n"
                "    fsm.state @s1 transitions  {\n"
                "      fsm.transition @t1(%arg0: !fsm.msg) {\n"
                "        fsm.defer_send %arg0\n"
                "      }\n"
                "    }\n"
                "  }\n"
                "}\n";

  auto result = parseSourceString(opText, &helper.ctx);
  ASSERT_NE(*result, nullptr);

  result->walk([](SendDeferMsg sendDeferMsg) {
    EXPECT_TRUE(sendDeferMsg.message().getType().isa<MsgType>());
  });
}

TEST(OpConstructionTests, ConstOp) {
  OpHelper helper;
  helper.builder.create<ConstOp>(helper.builder.getUnknownLoc(),
                                 StateType::get(&helper.ctx),
                                 helper.builder.getStringAttr("S"));

  std::string str;
  llvm::raw_string_ostream sstream(str);
  helper.module.print(sstream);

  auto expectedText = "module  {\n"
                      "  %0 = fsm.constant {value = \"S\"} : !fsm.state\n"
                      "}\n";
  EXPECT_STREQ(str.c_str(), expectedText);
}

TEST(OpConstructionTests, ConstOpParse) {
  OpHelper helper;
  auto opText = "module  {\n"
                "  %0 = fsm.constant {value = \"S\"} : !fsm.state\n"
                "}\n";

  auto result = parseSourceString(opText, &helper.ctx);
  ASSERT_NE(*result, nullptr);
  result->walk([](ConstOp cop) {
    ASSERT_TRUE(cop.value().isa<StringAttr>());
    EXPECT_EQ(cop.value().cast<StringAttr>().getValue(), "S");
    EXPECT_TRUE(cop.getType().isa<StateType>());
  });
}

TEST(OpConstructionTests, MessageDecl) {
  OpHelper helper;
  Location unknLoc = helper.builder.getUnknownLoc();
  auto msgDecl = helper.builder.create<MessageDecl>(unknLoc, "Resp");
  auto msgDeclEntry = msgDecl.addEntryBlock();
  helper.builder.setInsertionPointToStart(msgDeclEntry);
  helper.builder.create<NOPOp>(unknLoc);

  std::string str;
  llvm::raw_string_ostream sstream(str);
  helper.module.print(sstream);

  auto expectedText = "module  {\n"
                      "  fsm.m_decl @Resp decls  {\n"
                      "    fsm.nop\n"
                      "  }\n"
                      "}\n";
  EXPECT_STREQ(str.c_str(), expectedText);
}

TEST(OpConstructionTests, MessageDeclParse) {
  OpHelper helper;
  auto opText = "module  {\n"
                "  fsm.m_decl @Resp decls  {\n"
                "    fsm.nop\n"
                "  }\n"
                "}\n";
  auto result = parseSourceString(opText, &helper.ctx);
  ASSERT_NE(*result, nullptr);
  result->walk([](MessageDecl messageDecl) {
    EXPECT_EQ(messageDecl.sym_name(), "Resp");
  });
}

TEST(OpConstructionTests, MessageVariable) {
  OpHelper helper;
  Location unknLoc = helper.builder.getUnknownLoc();
  auto msgDecl = helper.builder.create<MessageDecl>(unknLoc, "Resp");
  auto msgDeclEntry = msgDecl.addEntryBlock();
  helper.builder.setInsertionPointToStart(msgDeclEntry);
  helper.builder.create<MessageVariable>(unknLoc, DataType::get(&helper.ctx),
                                         "cl");

  std::string str;
  llvm::raw_string_ostream sstream(str);
  helper.module.print(sstream);

  auto expectedText = "module  {\n"
                      "  fsm.m_decl @Resp decls  {\n"
                      "    %cl = fsm.m_var @cl : !fsm.data\n"
                      "  }\n"
                      "}\n";
  EXPECT_STREQ(str.c_str(), expectedText);
}

TEST(OpConstructionTests, MessageVariableParse) {
  OpHelper helper;
  auto opText = "module  {\n"
                "  fsm.m_decl @Resp decls  {\n"
                "    %cl = fsm.m_var @cl : !fsm.data\n"
                "  }\n"
                "}\n";
  auto result = parseSourceString(opText, &helper.ctx);
  ASSERT_NE(*result, nullptr);
  result->walk([](MessageVariable var) {
    EXPECT_EQ(var.sym_name(), "cl");
    EXPECT_TRUE(var.getType().isa<DataType>());
  });
}

TEST(OpConstructionTests, ComparisonOp) {
  OpHelper helper;
  Location unknLoc = helper.builder.getUnknownLoc();

  // create a network
  /*auto network = */ helper.builder.create<NetworkOp>(
      unknLoc, NetworkType::get(&helper.ctx), "ordered",
      helper.builder.getSymbolRefAttr("testnet"));

  auto cache = helper.builder.create<MachineOp>(
      unknLoc, "cache", helper.builder.getFunctionType({}, {}));
  Block *cacheEntry = cache.addEntryBlock();
  helper.builder.setInsertionPointToStart(cacheEntry);

  // create a basic state
  auto state1 = helper.builder.create<StateOp>(unknLoc, "s1");
  Block *s1Entry = state1.addEntryBlock();
  helper.builder.setInsertionPointToStart(s1Entry);

  // create a transition with a msg param
  FunctionType t1Type =
      helper.builder.getFunctionType({MsgType::get(&helper.ctx)}, {});
  auto trans1 =
      helper.builder.create<TransitionOp>(unknLoc, "t1", t1Type, nullptr);
  Block *t1Entry = trans1.addEntryBlock();
  helper.builder.setInsertionPointToStart(t1Entry);

  // create a send op
  auto msgSrc = helper.builder.create<AccessOp>(
      unknLoc, MsgType::get(&helper.ctx), trans1.getArgument(0),
      helper.builder.getStringAttr("src"));

  auto msgDst = helper.builder.create<AccessOp>(
      unknLoc, MsgType::get(&helper.ctx), trans1.getArgument(0),
      helper.builder.getStringAttr("dst"));

  helper.builder.create<CompareOp>(unknLoc, helper.builder.getI1Type(), msgSrc,
                                   msgDst, "=");

  // print out the result
  std::string str;
  llvm::raw_string_ostream sstream(str);
  helper.module.print(sstream);

  auto expectedText =
      "module  {\n"
      "  %testnet = fsm.network @testnet \"ordered\"\n"
      "  fsm.machine @cache() {\n"
      "    fsm.state @s1 transitions  {\n"
      "      fsm.transition @t1(%arg0: !fsm.msg) {\n"
      "        %0 = fsm.access {memberId = \"src\"} %arg0 : !fsm.msg -> "
      "!fsm.msg\n"
      "        %1 = fsm.access {memberId = \"dst\"} %arg0 : !fsm.msg -> "
      "!fsm.msg\n"
      "        %2 = fsm.comp \"=\" %0, %1 : !fsm.msg, !fsm.msg\n"
      "      }\n"
      "    }\n"
      "  }\n"
      "}\n";
  EXPECT_STREQ(str.c_str(), expectedText);
}

TEST(OpConstructionTests, AddOp) {
  OpHelper helper;
  Location unknLoc = helper.builder.getUnknownLoc();

  // create a network
  /*auto network = */ helper.builder.create<NetworkOp>(
      unknLoc, NetworkType::get(&helper.ctx), "ordered",
      helper.builder.getSymbolRefAttr("testnet"));

  auto cache = helper.builder.create<MachineOp>(
      unknLoc, "cache", helper.builder.getFunctionType({}, {}));
  Block *cacheEntry = cache.addEntryBlock();
  helper.builder.setInsertionPointToStart(cacheEntry);

  // create a basic state
  auto state1 = helper.builder.create<StateOp>(unknLoc, "s1");
  Block *s1Entry = state1.addEntryBlock();
  helper.builder.setInsertionPointToStart(s1Entry);

  // create a transition with a msg param
  FunctionType t1Type =
      helper.builder.getFunctionType({MsgType::get(&helper.ctx)}, {});
  auto trans1 =
      helper.builder.create<TransitionOp>(unknLoc, "t1", t1Type, nullptr);
  Block *t1Entry = trans1.addEntryBlock();
  helper.builder.setInsertionPointToStart(t1Entry);

  // create two constant
  auto c1 = helper.builder.create<ConstOp>(unknLoc, helper.builder.getI64Type(),
                                           helper.builder.getStringAttr("1"));
  auto c2 = helper.builder.create<ConstOp>(unknLoc, helper.builder.getI64Type(),
                                           helper.builder.getStringAttr("1"));

  // create the add op
  helper.builder.create<AddOp>(unknLoc, helper.builder.getI64Type(), c1, c2);

  // print out the result
  std::string str;
  llvm::raw_string_ostream sstream(str);
  helper.module.print(sstream);

  auto expectedText = "module  {\n"
                      "  %testnet = fsm.network @testnet \"ordered\"\n"
                      "  fsm.machine @cache() {\n"
                      "    fsm.state @s1 transitions  {\n"
                      "      fsm.transition @t1(%arg0: !fsm.msg) {\n"
                      "        %0 = fsm.constant {value = \"1\"} : i64\n"
                      "        %1 = fsm.constant {value = \"1\"} : i64\n"
                      "        %2 = fsm.add %0, %1 : i64, i64\n"
                      "      }\n"
                      "    }\n"
                      "  }\n"
                      "}\n";
  EXPECT_STREQ(str.c_str(), expectedText);
}

TEST(OpConstructionTests, AddOpParse) {
  OpHelper helper;
  auto opText = "module  {\n"
                "  %testnet = fsm.network @testnet \"ordered\"\n"
                "  fsm.machine @cache() {\n"
                "    fsm.state @s1 transitions  {\n"
                "      fsm.transition @t1(%arg0: !fsm.msg) {\n"
                "        %0 = fsm.constant {value = \"1\"} : i64\n"
                "        %1 = fsm.constant {value = \"1\"} : i64\n"
                "        %2 = fsm.add %0, %1 : i64, i64\n"
                "      }\n"
                "    }\n"
                "  }\n"
                "}\n";
  auto result = parseSourceString(opText, &helper.ctx);
  ASSERT_NE(*result, nullptr);
  result->walk([](AddOp addOp) {
    EXPECT_TRUE(addOp.getType().isa<IntegerType>());
    EXPECT_TRUE(addOp.lhs().getType().isa<IntegerType>());
    EXPECT_TRUE(addOp.rhs().getType().isa<IntegerType>());
  });
}

TEST(OpConstructionTests, SetAddOp) {
  OpHelper helper;
  Location unknLoc = helper.builder.getUnknownLoc();

  // create a network
  /*auto network = */ helper.builder.create<NetworkOp>(
      unknLoc, NetworkType::get(&helper.ctx), "ordered",
      helper.builder.getSymbolRefAttr("testnet"));

  auto cache = helper.builder.create<MachineOp>(
      unknLoc, "cache", helper.builder.getFunctionType({}, {}));
  Block *cacheEntry = cache.addEntryBlock();
  helper.builder.setInsertionPointToStart(cacheEntry);

  // create a set variable
  auto setVar = helper.builder.create<VariableOp>(
      unknLoc, SetType::get(IDType::get(&helper.ctx), 3), nullptr, "set");

  // create a basic state
  auto state1 = helper.builder.create<StateOp>(unknLoc, "s1");
  Block *s1Entry = state1.addEntryBlock();
  helper.builder.setInsertionPointToStart(s1Entry);

  // create a transition with a msg param
  FunctionType t1Type =
      helper.builder.getFunctionType({MsgType::get(&helper.ctx)}, {});
  auto trans1 =
      helper.builder.create<TransitionOp>(unknLoc, "t1", t1Type, nullptr);
  Block *t1Entry = trans1.addEntryBlock();
  helper.builder.setInsertionPointToStart(t1Entry);

  // access the cl
  auto access = helper.builder.create<AccessOp>(unknLoc, IDType::get(&helper.ctx), trans1.getArgument(0), "src");

  // set add op
  helper.builder.create<SetAdd>(unknLoc, setVar, access);

  // print out the result
  std::string str;
  llvm::raw_string_ostream sstream(str);
  helper.module.print(sstream);

  auto expectedText =
      "module  {\n"
      "  %testnet = fsm.network @testnet \"ordered\"\n"
      "  fsm.machine @cache() {\n"
      "    %set = fsm.variable \"set\" : !fsm.set<!fsm.id, 3>\n"
      "    fsm.state @s1 transitions  {\n"
      "      fsm.transition @t1(%arg0: !fsm.msg) {\n"
      "        %0 = fsm.access {memberId = \"src\"} %arg0 : !fsm.msg -> !fsm.id\n"
      "        fsm.set_add %set, %0 : !fsm.set<!fsm.id, 3>, !fsm.id\n"
      "      }\n"
      "    }\n"
      "  }\n"
      "}\n";
  EXPECT_STREQ(str.c_str(), expectedText);
}


TEST(OpConstructionTests, SetAddParse) {
  OpHelper helper;
  auto opText  =
      "module  {\n"
      "  %testnet = fsm.network @testnet \"ordered\"\n"
      "  fsm.machine @cache() {\n"
      "    %set = fsm.variable \"set\" : !fsm.set<!fsm.id, 3>\n"
      "    fsm.state @s1 transitions  {\n"
      "      fsm.transition @t1(%arg0: !fsm.msg) {\n"
      "        %0 = fsm.access {memberId = \"src\"} %arg0 : !fsm.msg -> !fsm.id\n"
      "        fsm.set_add %set, %0 : !fsm.set<!fsm.id, 3>, !fsm.id\n"
      "      }\n"
      "    }\n"
      "  }\n"
      "}\n";
  auto result = parseSourceString(opText, &helper.ctx);
  ASSERT_NE(*result, nullptr);
  result->walk([](SetAdd addOp) {
    EXPECT_TRUE(addOp.theSet().getType().isa<SetType>());
    EXPECT_TRUE(addOp.value().getType().isa<IDType>());
  });
}
