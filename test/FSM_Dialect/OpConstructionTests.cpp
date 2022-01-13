//
// Created by petr on 10/01/2022.
//
#include "FSM/FSMOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Parser.h"
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::fsm;

// Helper class to construct ops
class OpHelper {
public:
  OpHelper() : builder{&ctx} {
    ctx.getOrLoadDialect<FSMDialect>();
    ctx.getOrLoadDialect<StandardOpsDialect>();
    module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());
  }

  MLIRContext ctx;
  OpBuilder builder;
  ModuleOp module;
};

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
    ASSERT_EQ(op.initValue().cast<IntegerAttr>().getInt(), 21);
  });
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
  TransitionOp transOp = helper.builder.create<TransitionOp>(
      unkLoc, helper.builder.getStringAttr("load"),
      helper.builder.getSymbolRefAttr("M"));
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
                  "    fsm.transition {nextState = @M} action @load  {\n"
                  "      %c21_i64 = constant 21 : i64\n"
                  "    }\n"
                  "  }\n"
                  "}";

  ASSERT_STREQ(expctStr, str.c_str());
}

TEST(OpConstructionTests, ParseTransitionOp) {
  OpHelper helper;
  llvm::StringRef opText =
      "fsm.machine @cache() {\n"
      "  fsm.state @S transitions  {\n"
      "    fsm.transition {nextState = @M} action @load  {\n"
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
  TransitionOp sLoadTran = helper.builder.create<TransitionOp>(
      unknLoc, helper.builder.getStringAttr("load"),
      helper.builder.getSymbolRefAttr("M"));
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
  TransitionOp mLoadTran = helper.builder.create<TransitionOp>(
      unknLoc, helper.builder.getStringAttr("load"),
      helper.builder.getSymbolRefAttr("S"));
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

  // 6 - works with colons? - NO


}
